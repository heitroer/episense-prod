"""Avalia as previsões emitidas pelo dashboard e audita a cobertura dos vintages."""
import argparse
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.epiweeks import add_epiweeks, date_to_epiweek, epiweek_to_date
from models.forecast_audit import config_fingerprint, QUANTILES, forecast_rows, load_inputs, model_fingerprint
from models.metrics import paired_forecast_metrics


def week_date(week):
    return epiweek_to_date(int(week[:4]), int(week[4:]))


def dated_observations(directory, cutoff):
    """Preserva o último valor disponível em cada coleta até o dia pedido."""
    observations = {}
    for path in sorted(directory.glob("infodengue_cg_????????.csv")):
        try:
            collected = pd.to_datetime(path.stem.rsplit("_", 1)[-1], format="%Y%m%d").date()
        except ValueError:
            continue
        if collected > cutoff:
            continue
        frame = pd.read_csv(path, dtype={"SE": str})
        for row in frame.itertuples(index=False):
            if week_date(row.SE).date() <= collected and np.isfinite(row.casos) and row.casos >= 0:
                observations[row.SE] = {"cases": float(row.casos), "collected_on": str(collected)}
    return observations


def reconstruct(cases_folder, weather_folder, cutoff):
    observations = dated_observations(cases_folder, cutoff)
    rows = pd.DataFrame(
        [{"SE": week, **observation} for week, observation in sorted(observations.items())],
        columns=["SE", "cases", "collected_on"],
    )
    weather_files = []
    for path in weather_folder.glob("openmeteo_weekly_*.csv"):
        try:
            collected = pd.to_datetime(path.stem.rsplit("_", 1)[1], format="%Y%m%d").date()
        except ValueError:
            continue
        if collected <= cutoff:
            weather_files.append((collected, path))
    if weather_files:
        collected, path = max(weather_files)
        weather = pd.read_csv(path, dtype={"SE": str})
        start = pd.to_datetime(weather["week_start"])
        end = pd.to_datetime(weather["week_end"])
        weather = weather.loc[((end - start).dt.days == 6) & (end.dt.date <= cutoff)].copy()
        weather["weather_collected_on"] = str(collected)
        rows = rows.merge(weather, on="SE", how="left", validate="one_to_one")
    return rows


def operational_monitor(entries, vintage, *, today, maturity_weeks, first_origin):
    """Verifica a janela de emissão e as duas observações esperadas por origem."""
    latest = add_epiweeks(date_to_epiweek(today), -1)
    by_origin = defaultdict(list)
    for entry in entries:
        if entry.get("source") == "issued" and entry.get("issuance_id"):
            issued = pd.Timestamp(entry["generated_at"]).tz_convert("America/Campo_Grande").date()
            if issued <= today:
                by_origin[entry["origin_se"]].append((issued, entry))

    missing, open_origins, issues, targets = [], [], [], []
    observations = vintage.set_index("SE").to_dict("index")
    origin = add_epiweeks(first_origin, 0)
    expected = issued_count = 0
    while origin <= latest:
        expected += 1
        records = by_origin[origin]
        if not records:
            (open_origins if origin == latest else missing).append(origin)
        elif len(records) > 1:
            issues.append({"origin_se": origin, "issue": "duplicate_official_issuance"})
        else:
            issued_count += 1
            issued, entry = records[0]
            audit = entry.get("audit") or {}
            for h in (1, 2):
                target = add_epiweeks(origin, h)
                start = week_date(target).date()
                due = (week_date(target) + pd.Timedelta(weeks=maturity_weeks + 1)).date()
                lead = (start - issued).days
                rows = [r for r in entry.get("forecast", []) if r["h"] == h and r["target_se"] == target]
                shadows = [r for r in entry.get("persistence_shadow", [])
                           if r["h"] == h and r["target_se"] == target]
                case = observations.get(target, {})
                collected = case.get("collected_on")
                observed_week = audit.get("persistence_week")
                complete = len(rows) == len(shadows) == 1 and bool(observed_week) and observed_week <= origin
                if complete:
                    values = [rows[0].get(q) for q in QUANTILES.values()]
                    complete = (all(v is not None and math.isfinite(v) and v >= 0 for v in values)
                                and values == sorted(values) and shadows[0].get("observed_se") == observed_week
                                and shadows[0].get("cases") is not None
                                and shadows[0]["cases"] == audit.get("persistence")
                                and math.isfinite(shadows[0]["cases"]) and shadows[0]["cases"] >= 0)
                if not complete:
                    issues.append({"origin_se": origin, "h": h, "issue": "invalid_paired_emission"})
                if not complete:
                    status = "invalid_pair"
                elif lead <= -7:
                    status = "issued_after_target"
                elif today < due:
                    status = "awaiting_maturity"
                elif collected and collected >= due.isoformat():
                    status = "evaluable"
                else:
                    status = "missing_mature_collection"
                targets.append({"origin_se": origin, "h": h, "target_se": target, "issued_on": str(issued),
                                "lead_days": lead, "mature_from": str(due), "latest_case_collection": collected,
                                "status": status})
        origin = add_epiweeks(origin, 1)
    return {"first_origin": first_origin, "latest_expected_origin": latest, "expected_origins": expected,
            "issued_origins": issued_count, "missed_origins": missing, "open_origins": open_origins,
            "archive_issues": issues, "targets": targets}


def evaluate(entries, actuals, *, today, maturity_weeks):
    """Compara persistência e modelo nas mesmas semanas e separa o nowcast."""
    h8_published, h8_candidate = [], []
    h8_counts = defaultdict(int)
    pairs = defaultdict(list)
    shadow_pairs = defaultdict(list)
    shadow_counts = {
        "forecast": {h: {"evaluable": 0, "awaiting_maturity": 0, "missing_outcome": 0}
                     for h in (1, 2)},
        "nowcast": {h: {"evaluable": 0, "awaiting_maturity": 0, "missing_outcome": 0}
                    for h in (1, 2)},
    }
    eligible = 0
    entries = [e for e in entries if e.get("source") == "issued"
               and pd.Timestamp(e["generated_at"]).tz_convert("America/Campo_Grande").date() <= today]
    official = {entry["origin_se"] for entry in entries if entry.get("issuance_id")}
    if len(official) != sum(bool(e.get("issuance_id")) for e in entries):
        raise ValueError("Arquivo contém múltiplas emissões oficiais para a mesma origem")
    for entry in entries:
        audit = entry.get("audit")
        if entry.get("source") != "issued" or not audit:
            continue
        if entry["origin_se"] in official and not entry.get("issuance_id"):
            continue
        persistence = audit.get("persistence")
        if persistence is None or not math.isfinite(persistence) or persistence < 0:
            continue
        issued = pd.Timestamp(entry["generated_at"]).tz_convert("America/Campo_Grande").date()
        eligible += 1
        shadows = {(row["h"], row["target_se"]): row for row in entry.get("persistence_shadow", [])}
        for row in entry["forecast"]:
            horizon, target = row["h"], row["target_se"]
            start = week_date(target)
            end = (start + pd.Timedelta(weeks=1)).date()
            if issued >= end:
                continue
            lead = "forecast" if issued < start.date() else "nowcast"
            mature = (start + pd.Timedelta(weeks=1 + maturity_weeks)).date() <= today
            actual = actuals.get(target)
            shadow = shadows.get((horizon, target))
            if shadow is not None and horizon in (1, 2):
                value = shadow["cases"]
                if value != persistence or shadow["observed_se"] != audit.get("persistence_week"):
                    raise ValueError(f"Persistência divergente da emissão {entry.get('issuance_id')}")
                status = "awaiting_maturity" if not mature else (
                    "missing_outcome" if actual is None else "evaluable"
                )
                shadow_counts[lead][horizon][status] += 1
                if status == "evaluable":
                    shadow_pairs[(lead, horizon)].append((actual, value, row))
            if horizon == 8 and lead == "forecast" and "h8_shadow" in entry:
                candidate = entry["h8_shadow"]
                if candidate is None:
                    h8_counts["missing_candidate"] += 1
                else:
                    from models.h8_shadow import VARIANT
                    if (candidate.get("variant") != VARIANT or candidate.get("target_se") != target
                            or candidate.get("h") != 8):
                        raise ValueError("Variante H8 incompatível com a emissão")
                    if not mature:
                        h8_counts["awaiting_maturity"] += 1
                    elif actual is None:
                        h8_counts["missing_outcome"] += 1
                    else:
                        h8_counts["evaluable"] += 1
                        h8_candidate.append((actual, persistence, candidate))
                        h8_published.append((actual, persistence, row))
            if lead == "forecast" and mature and actual is not None:
                pairs[horizon].append((actual, persistence, row))

    def summarize(items):
        actual = np.array([item[0] for item in items])
        baseline = np.array([item[1] for item in items])
        quantiles = [[item[2].get(key) for key in QUANTILES.values()] for item in items]
        result = paired_forecast_metrics(actual, quantiles, baseline)
        excluded_keys = {"n_eligible", "n_scored", "n_excluded", "persistence_mae", "skill_mae_persistence"}
        return {
            "n": result["n_scored"],
            "model": {key: value for key, value in result.items() if key not in excluded_keys},
            "n_eligible": result["n_eligible"],
            "n_excluded": result["n_excluded"],
            "persistence_mae": result.get("persistence_mae"),
            "mae_skill": result.get("skill_mae_persistence"),
        }

    return {
        "eligible_origins": eligible,
        "h8_q95_shadow": {"counts": dict(h8_counts),
                          "published": summarize(h8_published) if h8_published else None,
                          "candidate": summarize(h8_candidate) if h8_candidate else None},
        "metrics_by_horizon": {h: summarize(items) for h, items in sorted(pairs.items())},
        "persistence_shadow": {
            lead: {
                "counts_by_horizon": counts,
                "metrics_by_horizon": {
                    h: summarize(items) for (bucket, h), items in sorted(shadow_pairs.items()) if bucket == lead
                },
            }
            for lead, counts in shadow_counts.items()
        },
    }


def final_holdout(entries, cases_folder, today):
    """Fixed prospective cohort. Development folds never enter the final score."""
    protocol = json.loads((ROOT / "config/prospective_holdout.json").read_text())
    horizons = protocol["horizons"]
    origins = [add_epiweeks(protocol["first_origin"], i) for i in range(protocol["weeks"])]
    last_target = add_epiweeks(origins[-1], max(horizons))
    # Allow a full collection week after the last outcome reaches maturity.
    close = week_date(add_epiweeks(last_target, 2 + protocol["maturity_weeks"])).date()
    result = {"protocol": protocol, "last_origin": origins[-1], "score_on": str(close)}
    if today < close:
        return {**result, "status": "awaiting_close"}

    actuals = dated_observations(cases_folder, close)
    official = defaultdict(list)
    for e in entries:
        if e.get("source") == "issued" and e.get("issuance_id") and e["origin_se"] in origins:
            official[e["origin_se"]].append(e)
    missing = []
    rows = defaultdict(list)
    late = defaultdict(int)
    for origin in origins:
        group = official[origin]
        if len(group) != 1:
            missing.append(origin)
            continue
        entry = group[0]
        audit = entry.get("audit") or {}
        issued = pd.Timestamp(entry["generated_at"]).tz_convert("America/Campo_Grande").date()
        if (issued < week_date(origin).date() or issued > close
                or any(audit.get(key) != protocol[key] for key in
                       ("release", "model_sha256", "inference_sha256"))
                or config_fingerprint(audit.get("config")) != protocol["config_sha256"]):
            missing.append(origin)
            continue
        for h in horizons:
            target = add_epiweeks(origin, h)
            if issued >= week_date(target).date():
                late[h] += 1
                continue
            forecasts = [r for r in entry["forecast"] if r["h"] == h and r["target_se"] == target]
            observation = actuals.get(target)
            due = week_date(add_epiweeks(target, 1 + protocol["maturity_weeks"])).date()
            if len(forecasts) == 1 and observation and observation["collected_on"] >= str(due):
                rows[h].append((observation["cases"], [forecasts[0].get(q) for q in QUANTILES.values()],
                                audit.get("persistence")))
    result["missing_or_incompatible_origins"] = missing
    metrics = {}
    for h in horizons:
        expected = len(origins) - late[h]
        score = paired_forecast_metrics(*zip(*rows[h])) if rows[h] else {"n_scored": 0}
        complete = expected > 0 and score["n_scored"] == expected and not missing
        metrics[h] = {"expected_forecasts": expected, "issued_after_forecast_cutoff": late[h],
                      "n_scored": score["n_scored"], "metrics": score if complete else None}
    result["by_horizon"] = metrics
    result["status"] = "complete" if all(r["metrics"] for r in metrics.values()) else "incomplete"
    return result


def verify_replay(entries, inputs, models_root=None):
    from models.inference import EpisenseInference

    checked = 0
    skipped = 0
    for entry in entries:
        audit = entry.get("audit") or {}
        if entry.get("source") != "issued" or not audit.get("model_sha256"):
            skipped += 1
            continue
        root = (models_root or ROOT / "models") / "releases"
        release = audit.get("release")
        directory = (root / release).resolve() if release else root.parent
        if (release and directory.parent != root.resolve()) or not directory.is_dir():
            raise ValueError(f"Versão de modelos ausente ou inválida para {entry['origin_se']}")
        if model_fingerprint(directory) != audit["model_sha256"]:
            raise ValueError(f"Artefatos de modelo divergentes para {entry['origin_se']}")
        frame = load_inputs(inputs, audit["input_sha256"])
        origin = entry["origin_se"]
        if frame["SE"].iloc[-1] != origin:
            raise ValueError(f"Snapshot diverge da origem {origin}")
        engine = EpisenseInference(str(directory), audit["config"])
        replay = forecast_rows(origin, engine.predict(frame))
        published = [{key: row[key] for key in ("h", "target_se", *QUANTILES.values())}
                     for row in entry["forecast"]]
        if replay != published:
            raise ValueError(f"Replay diverge da previsão publicada em {origin}")
        if entry.get("h8_shadow"):
            from models.h8_shadow import shadow_forecast
            if shadow_forecast(engine, frame, origin) != entry["h8_shadow"]:
                raise ValueError(f"Replay diverge da candidata H8 em {origin}")
        checked += 1
    return {"checked": checked, "skipped_without_model_hash": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from config.config import load_config
    from config.municipalities import SERVICES
    cfg = load_config()
    local_code = str(cfg["project"]["geocode"])
    parser.add_argument("--geocode", choices=["5002704", *SERVICES], default=local_code)
    parser.add_argument("--as-of", type=date.fromisoformat,
                        default=pd.Timestamp.now(tz="America/Campo_Grande").date())
    parser.add_argument("--maturity-weeks", type=int, default=16)
    parser.add_argument("--monitor-from-origin",
                        help="Primeira origem do acompanhamento prospectivo H1/H2")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--vintage-csv", type=Path)
    args = parser.parse_args()
    if args.maturity_weeks < 0:
        parser.error("--maturity-weeks deve ser não negativo")
    if args.geocode == local_code:
        processed = ROOT / cfg["paths"]["data_processed"]
        raw = ROOT / cfg["paths"]["data_raw"]
        models_root = ROOT / cfg["paths"]["models"]
    else:
        city_dir = ROOT / "cities" / args.geocode
        processed, raw, models_root = city_dir / "processed", city_dir / "raw", city_dir / "pilot"
    archive = processed / "forecast_archive.json"
    if not (processed / "episense_base.csv").exists():
        parser.error(f"Base municipal ausente em {processed}")
    entries = json.loads(archive.read_text()) if archive.exists() else []
    vintage = reconstruct(raw / "infodengue", raw / "openmeteo", args.as_of)
    if args.vintage_csv:
        vintage.to_csv(args.vintage_csv, index=False)
    # Um valor preliminar arquivado não se torna maduro pela passagem do tempo.
    mature_vintage = vintage.loc[[
        pd.Timestamp(row.collected_on) >= week_date(row.SE) + pd.Timedelta(weeks=1 + args.maturity_weeks)
        for row in vintage.itertuples()
    ]]
    report = evaluate(entries, mature_vintage.set_index("SE")["cases"].to_dict(),
                      today=args.as_of, maturity_weeks=args.maturity_weeks)
    first_origin = args.monitor_from_origin
    if first_origin is None:
        first_origin = "202645" if args.geocode == "5002704" else min(
            (row["origin_se"] for row in entries if row.get("source") == "issued"),
            default=date_to_epiweek(args.as_of))
    report["operational_monitor"] = operational_monitor(
        entries, vintage, today=args.as_of, maturity_weeks=args.maturity_weeks, first_origin=first_origin)
    report["dated_case_observations"] = len(vintage)
    report["with_complete_dated_weather"] = (
        int(vintage["weather_collected_on"].notna().sum()) if "weather_collected_on" in vintage else 0
    )
    report["geocode"] = args.geocode
    report["final_holdout"] = (final_holdout(entries, raw / "infodengue", args.as_of)
                               if args.geocode == "5002704" else {"status": "sem_protocolo_municipal"})
    report["as_of"] = str(args.as_of)
    report["outcomes"] = "última contagem arquivada até a data de corte"
    report["maturity_weeks"] = args.maturity_weeks
    if args.replay:
        as_of_entries = [e for e in entries if e.get("source") != "issued" or
                         pd.Timestamp(e["generated_at"]).tz_convert("America/Campo_Grande").date() <= args.as_of]
        report["replay"] = verify_replay(as_of_entries, processed / "forecast_inputs", models_root)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
