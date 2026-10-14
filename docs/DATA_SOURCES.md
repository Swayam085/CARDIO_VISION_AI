# CardioVision AI — Dataset Provenance

## Official Dataset

| Field | Value |
|-------|-------|
| **Name** | Extension of Z-Alizadeh Sani Dataset |
| **Source** | UCI Machine Learning Repository |
| **Dataset ID** | 411 |
| **URL** | https://archive.ics.uci.edu/dataset/411/extention+of+z |
| **DOI** | https://doi.org/10.24432/C5QS3G |
| **Records** | 303 patients |
| **Features** | 59 (demographic, symptom, ECG, laboratory, echo, vessel labels) |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Download Method** | `ucimlrepo` Python library |

### Citation

```
Z-Alizadeh Sani, M., et al. (2022). Extension of Z-Alizadeh Sani Dataset.
UCI Machine Learning Repository. https://doi.org/10.24432/C5QS3G
```

### Feature Categories

- **Demographics**: Age, Sex, BMI
- **Risk factors**: HTN, DM, Smoking, Family History, Obesity
- **Vitals**: BP, PR
- **Symptoms**: Typical Chest Pain, Atypical, Dyspnea, Function Class
- **ECG**: ST elevation, T inversion, LVH, BBB, Q wave
- **Laboratory**: Cholesterol, TG, LDL, HDL, BUN, Creatinine, FBS, ESR, HB
- **Echo**: EF, Regional Wall Motion Abnormality (RWMA), Wall Motion Index
- **Cardiac enzymes**: CK-MB, TnI

### Leakage / Target Columns (EXCLUDED from inputs)

| Column | Type |
|--------|------|
| `CAD` | Primary target (overall CAD) |
| `LAD` | Vessel target (LAD stenosis) |
| `LCX` | Vessel target (LCX stenosis) |
| `RCA` | Vessel target (RCA stenosis) |
| `Cath` | Leakage proxy (catheterization result) |

---

## 3D Anatomical Reference

| Field | Value |
|-------|-------|
| **Source** | BodyParts3D / Anatomography |
| **Provider** | Database Center for Life Science (DBCLS), Japan |
| **URL** | https://lifesciencedb.jp/bp3d |
| **Preferred Format** | GLB / GLTF |
| **License** | Creative Commons Attribution-ShareAlike 2.1 Japan (CC BY-SA 2.1 JP) |
| **Usage** | Anatomical reference for 3D coronary vessel visualization |
| **Integration Status** | Fully Integrated — Authentic BodyParts3D binary glTF (`public/models/heart_anatomy_bp3d.glb`) |

### Attribution

> BodyParts3D, © Database Center for Life Science, licensed under CC BY-SA 2.1 Japan
> Source: https://lifesciencedb.jp/bp3d

### 3D Visualization Disclaimer

The 3D visualization represents **model-predicted vessel-level risk scores** mapped to
LAD/LCX/RCA coronary anatomy. It does NOT represent:
- Exact lesion coordinates
- CT scan or MRI image data
- Pixel-level segmentation results
- Confirmed pathological anatomy
