# Cleanup receipt

`removed_files.jsonl` is the explicit file inventory checked before deletion:
2,021 files totaling 35,932,425,213 bytes. `receipt.json` records completion,
removed roots, retired code, protected inputs and verification. No run databases
were bulk archived. Essential Skin atoms and four pinned retrieval ranking tables
were extracted into `../assets/` before deleting their original intermediates.

`selected_files.json` records the SHA256 and size of every file in all six selected
release directories. All 155 files matched before and after deletion. The standalone
DILI scorer source is also unchanged. Required older BBB/Oral v1 and Skin v5
published inputs remain for explicitly pinned selection contracts. Current maps,
weights, central ledgers, pending UID worklists and provenance were not rewritten.

Validation: 309 focused selection, semantic, DILI and prediction-reader tests;
2 repository-layout checks; 12 CLI help checks; Python 3.11 syntax; shell syntax.
Seven additional retrieval tests require Torch/Transformers absent from the test
environment and were excluded after confirming dependency-only failures.
No real model calls, inference, training or publication were performed.
