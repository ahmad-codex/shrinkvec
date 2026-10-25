# Roadmap

shrinkvec answers one question: how small can these embeddings get before search gets worse, and
how do I get there in Postgres? Items move from Next to Now when someone is working on them.

## Now

- **Dimension truncation.** Matryoshka models (OpenAI text-embedding-3, Nomic, others) can be cut to
  fewer dimensions. Compare "fewer dimensions" with "fewer bits", and both together. ([#3](https://github.com/ahmad-codex/shrinkvec/issues/3))
- **Filtered search.** Recall under a `WHERE` clause, with and without pgvector's iterative index
  scans, since filters are where approximate indexes lose the most. ([#4](https://github.com/ahmad-codex/shrinkvec/issues/4))
- **IVFFlat** on `vector` and `halfvec` in the Postgres runner. ([#5](https://github.com/ahmad-codex/shrinkvec/issues/5))
- **More extensions when installed:** pgvectorscale (DiskANN with statistical binary quantization)
  and pg_turbovec. ([#6](https://github.com/ahmad-codex/shrinkvec/issues/6), [#7](https://github.com/ahmad-codex/shrinkvec/issues/7))
- **A text-embedding-3 dataset** with a clear license, next to ada-002, GloVe and SIFT. ([#8](https://github.com/ahmad-codex/shrinkvec/issues/8))

## Next

- **Throughput under load:** N concurrent clients, not only single-query latency. ([#9](https://github.com/ahmad-codex/shrinkvec/issues/9))
- **Product quantization (PQ)** in the lab, as the classic baseline the papers compare against. ([#10](https://github.com/ahmad-codex/shrinkvec/issues/10))
- **RaBitQ with many centroids** and a 4-bit query, to match VectorChord's exact setup in the lab. ([#11](https://github.com/ahmad-codex/shrinkvec/issues/11))
- **Cache the ground truth** between runs on the same vectors. ([#12](https://github.com/ahmad-codex/shrinkvec/issues/12))
- **A Python API page**, for using the lab from a notebook. ([#13](https://github.com/ahmad-codex/shrinkvec/issues/13))

## Later

- **Faster lab scans** with integer kernels instead of decoding to floats. ([#14](https://github.com/ahmad-codex/shrinkvec/issues/14))
- **Memory view:** the size of each option at 1M and 100M vectors. ([#15](https://github.com/ahmad-codex/shrinkvec/issues/15))
- **Compare reports** from several runs (models, datasets) in one page. ([#16](https://github.com/ahmad-codex/shrinkvec/issues/16))

## Not planned

- A vector database or a Postgres extension of its own. shrinkvec measures the options that exist.
- Writing to your tables. The only schema it touches is its own.
