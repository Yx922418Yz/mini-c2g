# Data directory

This directory is populated by scripts, not committed to git (see `.gitignore`).

## Real FineWeb micro data (local validation)

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts/download_micro_data.ps1
```
```bash
# Linux / macOS
bash scripts/download_micro_data.sh
```

This downloads, via `hf-mirror.com`, byte-range prefixes of the official
challenge shards hosted at `willdepueoai/parameter-golf`:

| file | contents |
|---|---|
| `train_head.bin` | header + 8,000,000 real FineWeb tokens (uint16) |
| `val_head.bin` | header + 2,000,000 real FineWeb tokens |
| `fineweb_1024_bpe.model` | official SP1024 SentencePiece tokenizer |
| `manifest.json` | official data manifest |

`prepare_local_data.py` rewrites the prefixes into correctly-headered shards:

- `data/micro/fineweb_train_000000.bin`
- `data/micro/fineweb_val_000000.bin`
- `data/tokenizers/fineweb_1024_bpe.model`

## Synthetic data (CI only)

`scripts/make_synthetic_shards.py` produces tiny random-token shards and a
256-piece tokenizer with no network access. These exercise code paths only;
they are not language data and must not be used for reported results.

## Full H100 data

The full FineWeb SP1024 shards are prepared with the official challenge
preprocessing (https://github.com/openai/parameter-golf, `data/prepare.py`)
and referenced through `DATA_PATH`.
