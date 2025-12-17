# utils/slice_arxiv.py

import pandas as pd


df = pd.read_csv("./arxiv_data/arxiv_cornell_title_abstract.csv")

# 2. 過濾你想要的主題
#   這個例子：cs.CL 類別 + title 或 abstract 內含 "transformer"
mask_cat = df["categories"].fillna("").str.contains("cs.CL")
mask_kw = (df["title"].fillna("").str.contains("transformer", case=False) |
           df["abstract"].fillna("").str.contains("transformer", case=False))

df_demo = df[mask_cat & mask_kw].copy()

print("符合條件的筆數:", len(df_demo))

# 3. 取前 N 筆當 demo，例如 300 篇
N = 500
df_demo = df_demo.head(N)

# 4. 存成獨立 demo CSV
demo_path = "./arxiv_data/arxiv_demo_transformer.csv"
df_demo.to_csv(demo_path, index=False)
print("Demo CSV saved to:", demo_path)
