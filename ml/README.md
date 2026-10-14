# CardioVision AI — ML Pipeline

## Dataset

**Official Source**: https://archive.ics.uci.edu/dataset/411/extention+of+z
**UCI Dataset ID**: 411
**Name**: Extension of Z-Alizadeh Sani Dataset

### Download

```bash
# From project root:
pip install -r backend/requirements.txt
python ml/src/download_data.py
```

This downloads the dataset via `ucimlrepo`, saves to `ml/data/raw/`,
and generates a provenance JSON file. It will FAIL (not silently substitute)
if the download cannot be completed.

## Data Leakage Policy

The following columns are FORBIDDEN as model inputs:

| Column | Reason |
|--------|--------|
| `LAD` | Target — LAD stenosis label |
| `LCX` | Target — LCX stenosis label |
| `RCA` | Target — RCA stenosis label |
| `Cath` | Leakage proxy — catheterization result |

Run the formal audit:
```bash
python ml/src/leakage_audit.py
```

## Pipeline (Day 2+)

```
download_data.py  →  preprocess.py  →  leakage_audit.py  →  train.py  →  evaluate.py
```

All metrics are computed from real cross-validated evaluation on held-out test sets.
No metrics are hardcoded.
