# Roadmap

shrinkvec answers one question: how small can these embeddings get before search gets worse, and
how do I get there in Postgres? Items move from Next to Now when someone is working on them.

## Now

- **Dimension truncation.** Matryoshka models (OpenAI text-embedding-3, Nomic, others) can be cut to
  fewer dimensions. Compare "fewer dimensions" with "fewer bits", and both together.
- **Filtered search.** Recall under a `WHERE` clause, with and without pgvector's iterative index
  scans, since filters are where approximate indexes lose the most.
- **IVFFlat** on `vector` and `halfvec` in the Postgres runner.
- **More extensions when installed:** pgvectorscale (DiskANN with statistical binary quantization)
  and pg_turbovec.
- **A text-embedding-3 dataset** with a clear license, next to ada-002, GloVe and SIFT.

## Next

- **Throughput under load:** N concurrent clients, not only single-query latency.
- **Product quantization (PQ)** in the lab, as the classic baseline the papers compare against.
- **RaBitQ with many centroids** and a 4-bit query, to match VectorChord's exact setup in the lab.
- **Cache the ground truth** between runs on the same vectors.
- **A Python API page**, for using the lab from a notebook.

## Later

- **Faster lab scans** with integer kernels instead of decoding to floats.
- **Memory view:** the size of each option at 1M and 100M vectors.
- **Compare reports** from several runs (models, datasets) in one page.

## Not planned

- A vector database or a Postgres extension of its own. shrinkvec measures the options that exist.
- Writing to your tables. The only schema it touches is its own.
