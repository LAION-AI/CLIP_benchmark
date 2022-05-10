import json
from glob import glob
import pandas as pd
rows = []
for path in glob("*.json"):
    data = json.load(open(path))
    row = {}
    row.update(data["metrics"])
    row.update(data)
    del row["metrics"]
    rows.append(row)
df = pd.DataFrame(rows)
df["model_fullname"] = df.apply(lambda s:s["model"]+" "+s["pretrained"], axis=1)
df.to_csv("benchmark.csv", index=False)