import concurrent.futures
import csv
import hashlib
from pathlib import Path
import urllib.request

work = Path(__file__).parent
labels = work / 'labels'
labels.mkdir(exist_ok=True)
with (work / 'age_records.csv').open(newline='') as f:
    rows = [r for r in csv.DictReader(f) if int(r['subject']) < 35 and r['record type'] == 'Hypnogram']

def download(row):
    name = row['fname']
    path = labels / name
    url = 'https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette/' + name
    for attempt in range(3):
        try:
            if path.exists() and hashlib.sha1(path.read_bytes()).hexdigest() == row['sha']:
                return name
            with urllib.request.urlopen(url, timeout=15) as response:
                data = response.read()
            if hashlib.sha1(data).hexdigest() != row['sha']:
                raise ValueError('SHA1 mismatch')
            path.write_bytes(data)
            return name
        except Exception:
            if attempt == 2:
                raise

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    print('Verified labels:', len(list(pool.map(download, rows))), flush=True)
