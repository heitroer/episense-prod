"""Replica a cadeia temporal EpiSense em três séries isoladas de arboviroses."""
import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.collect_infodengue import process_infodengue_data
from data.process_data import EpisenseDataProcessor
from scripts.benchmark_model_variants import CASE_FEATURES, HORIZONS, run_variant
from scripts.evaluate_expansion import series_in

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cg-weather", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    processor = EpisenseDataProcessor()
    features = CASE_FEATURES + ["temp_lag2", "temp_lag4", "precip_lag2", "precip_lag4",
                                "humidity_lag2", "humidity_lag4"]
    for name in series_in(directory):
        city, disease = name.split("_")
        weather_path = args.cg_weather if city == "5002704" else directory / f"{city}_weather.csv"
        weather = pd.read_csv(weather_path, dtype={"SE":str})
        source = pd.read_csv(directory/f"{name}.csv",dtype={"SE":str})
        reports = process_infodengue_data(source)
        base = processor.create_base_features(processor.merge_data(reports, weather))
        if len(base) != len(source) or base["is_imputed"].any():
            raise ValueError(f"Semanas ausentes na série {name}")
        cfg = yaml.safe_load((ROOT/"config/config.yaml").read_text())
        cfg["project"].update(geocode=int(city), disease=disease)
        output = directory/"canonical"/name
        output.mkdir(parents=True)
        snapshot = output/"training_base.csv"
        base.to_csv(snapshot,index=False)
        os.environ["EPISENSE_TRAINING_BASE"] = str(snapshot)
        logging.getLogger("models.inference").setLevel(logging.ERROR)
        run_variant("local",base,cfg,output,HORIZONS,variant=(1,features),workers=2)
