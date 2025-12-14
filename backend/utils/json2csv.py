import json
import pandas as pd

input_json = "../arxiv_data/arxiv-metadata-oai-snapshot.json"   # Kaggle 解壓後的檔名
output_csv = "../arxiv_data/arxiv_cornell_title_abstract.csv"

rows = []
with open(input_json, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        rows.append({
            "id": obj.get("id", ""),
            "title": obj.get("title", ""),
            "abstract": obj.get("abstract", ""),
            "categories": obj.get("categories", "")
        })

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print("Saved:", output_csv, "rows:", len(df))
