"""Preserve the first InfoDengue response observed for each latest Campo Grande week."""
import csv
from datetime import date, datetime, timedelta, timezone
import io
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile
from urllib.parse import urlencode


def save_counts(rows, folder):
    records = sorted((str(row["SE"]), str(row["casos"])) for row in rows)
    weeks = [week for week, _ in records]
    if not weeks or len(weeks) != len(set(weeks)):
        raise ValueError("Janela InfoDengue vazia ou duplicada")
    if any(len(week) != 6 or not week.isdigit() or not 1 <= int(week[4:]) <= 53 for week in weeks):
        raise ValueError("Semana epidemiológica inválida")
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{weeks[-1]}.csv"
    if destination.exists():
        return destination
    collected = datetime.now(timezone.utc).isoformat()
    with NamedTemporaryFile(mode="w", newline="", dir=folder, delete=False) as tmp:
        try:
            writer = csv.writer(tmp)
            writer.writerow(["collected_at_utc", "SE", "casos"])
            writer.writerows((collected, week, value) for week, value in records)
            tmp.flush()
            os.fsync(tmp.fileno())
            try:
                os.link(tmp.name, destination)
            except FileExistsError:
                pass
        finally:
            os.unlink(tmp.name)
    return destination


def epi_year_week(day):
    sunday = day - timedelta(days=(day.weekday() + 1) % 7)
    year = (sunday + timedelta(days=3)).year
    jan4 = date(year, 1, 4)
    first = jan4 - timedelta(days=(jan4.weekday() + 1) % 7)
    return year, 1 + (sunday - first).days // 7


def fetch_window():
    today = date.today()
    start_year, start_week = epi_year_week(today - timedelta(days=126))
    end_year, end_week = epi_year_week(today)
    query = urlencode(dict(geocode=5002704, disease="dengue", format="csv", ew_format="SE",
                           ew_start=start_week, ey_start=start_year, ew_end=end_week, ey_end=end_year))
    result = subprocess.run(["/usr/bin/curl", "--fail", "--silent", "--show-error", "--location",
                             "--max-time", "60", "--retry", "2",
                             "https://info.dengue.mat.br/api/alertcity?" + query],
                            check=True, capture_output=True, text=True)
    return csv.DictReader(io.StringIO(result.stdout))


if __name__ == "__main__":
    print(save_counts(fetch_window(), Path(sys.argv[1])))
