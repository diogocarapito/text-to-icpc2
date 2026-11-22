# Memoire: how text-to-ICPC2 was built

Detailed notes for reconstructing how the tool was made, to support the article and paper. Each note covers one topic, records the numbers with their source files and commits, and says how to regenerate them.

How this folder relates to the other docs:

- [`../WRITEUP.md`](../WRITEUP.md): condensed material in article order (summary, methods, results, threats).
- [`../JOURNAL.md`](../JOURNAL.md): dated lab notebook, in the order things happened.
- **`memoire/`**: the in-depth analyses behind a WRITEUP claim. Numbers here are reproducible from a script in `scripts/`; the raw output sits in `memoire/data/`.

## Notes

| # | note | topic | script |
|---|---|---|---|
| 01 | [Dataset lineage: v1 → v2](01-dataset-lineage.md) | how much of the 2024–25 dataset is in the data that trained the published v2 model | `scripts/dataset_lineage.py` |

## Adding a note

Name it `NN-topic.md`. Start with the question and the short answer, then the numbers, then how they were obtained (script, input files, commit), then caveats. Add it to the table above.
