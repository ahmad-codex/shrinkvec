"""Build step for the demo image: a float32 slice of the DBpedia OpenAI embeddings as .npy.

Kept in float32 on purpose. Rounding the data to float16 here would make halfvec look perfect.
"""

import sys
import urllib.request

import numpy as np
import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main/data/"
FILE = "train-00000-of-00026-3c7b99d1c7eda36e.parquet"

rows, out = int(sys.argv[1]), sys.argv[2]
request = urllib.request.Request(URL + FILE, headers={"User-Agent": "shrinkvec-demo-build"})
with urllib.request.urlopen(request) as r, open("/tmp/part.parquet", "wb") as f:
    while chunk := r.read(1 << 20):
        f.write(chunk)
column = pq.read_table("/tmp/part.parquet", columns=["openai"]).column("openai").combine_chunks()
vectors = column.values.to_numpy(zero_copy_only=False).reshape(len(column), -1).astype(np.float32)
np.save(out, np.ascontiguousarray(vectors[:rows]))
print(f"saved {rows} x {vectors.shape[1]} float32 vectors to {out}")
