import argparse
import json
from glob import glob
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("files", type=str, nargs="+", help="JSON files")
parser.add_argument("--output", type=str, default="benchmark.csv", help="CSV output file")
args = parser.parse_args()
rows = []
for path in args.files:
    data = json.load(open(path))
    row = {}
    row.update(data["metrics"])
    row.update(data)
    del row["metrics"]
    rows.append(row)
df = pd.DataFrame(rows)
df["model_fullname"] = df.apply(lambda s:s["model"]+" "+s["pretrained"], axis=1)
df.to_csv(args.output, index=False)
