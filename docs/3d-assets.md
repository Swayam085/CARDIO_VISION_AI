# CardioVision AI — 3D Anatomical Assets Provenance & Architecture

## 1. Primary 3D Anatomical Data Source

| Metadata Field | Value / Provenance Details |
|---|---|
| **Repository / Portal** | BodyParts3D / Anatomography |
| **Provider** | Database Center for Life Science (DBCLS), Research Organization of Information and Systems (ROIS), Japan |
| **Official Portal URL** | [https://lifesciencedb.jp/bp3d/](https://lifesciencedb.jp/bp3d/) |
| **LSDB Archive** | [https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html) |
| **Open Source Reference Repository** | [Interactive-3D-Human-Heart-Visualization](https://github.com/simonreisinger/Interactive-3D-Human-Heart-Visualization) by Simon Reisinger (curated Postnatal Anatomical Structures from DBCLS BodyParts3D) |
| **Standard Ontology** | Foundational Model of Anatomy (FMA) & BodyParts3D Concept Identifiers (BP IDs) |
| **Asset Format** | Standard OBJ mesh primitives converted to optimized binary glTF 2.0 (`.glb`) |
| **License Terms** | **Creative Commons Attribution-ShareAlike 2.1 Japan (CC BY-SA 2.1 Japan / CC BY-SA 2.1 JP)** |

### Official Attribution Notice
> **BodyParts3D, © The Database Center for Life Science (DBCLS) licensed under CC Attribution-Share Alike 2.1 Japan.**  
> Source: https://lifesciencedb.jp/bp3d  
> DOI / LSDB: https://doi.org/10.7875/lead.author.2011.002

---

## 2. Anatomical Mesh Registry & Foundational Model of Anatomy (FMA) Mapping

The anatomical heart and coronary model used in CardioVision AI uses authentic polygon meshes directly derived from BodyParts3D with strictly verified anatomical correspondence. No arbitrary geometry has been invented.

### A. Left Anterior Descending Artery (LAD)
- **Clinical Target**: Anterior myocardial perfusion, interventricular groove
- **FMA & BP3D Components**:
  - `FMA74912` / `BP51969`: Trunk of anterior interventricular branch of left coronary artery
  - `FMA3860` / `BP51972`: Diagonal branch of anterior descending branch of left coronary artery
  - `FMA3888` / `BP51965`: Second left anterior branch of anterior interventricular branch of left coronary artery
  - `FMA3893` / `BP51614`: Anterior septal branches of anterior interventricular artery
  - `FMA3890` / `BP51962`: Third left anterior branch of anterior interventricular branch
- **Mesh Identifier in GLB Scene**: `Vessel_LAD`

### B. Left Circumflex Artery (LCX)
- **Clinical Target**: Lateral and posterior myocardial perfusion, atrioventricular groove
- **FMA & BP3D Components**:
  - `FMA74923` / `BP51973`: Trunk of circumflex branch of left coronary artery
  - `FMA3914` / `BP51610`: First posterior ventricular branch of circumflex coronary artery
  - `FMA3902` / `BP51926`: Left marginal artery
- **Mesh Identifier in GLB Scene**: `Vessel_LCX`

### C. Right Coronary Artery (RCA)
- **Clinical Target**: Right atrium, right ventricle, inferior wall and SA/AV nodal branches
- **FMA & BP3D Components**:
  - `FMA3802` / `BP51977`: Trunk of right coronary artery
  - `FMA3815` / `BP51980`: First anterior ventricular branch of right coronary artery
  - `FMA3818` / `BP51950`: Marginal branch of right coronary artery
  - `FMA3837` / `BP51956`: First posterior ventricular branch of right coronary artery
  - `FMA3829` / `BP51957`: Anterior atrial branch of right coronary artery
- **Mesh Identifier in GLB Scene**: `Vessel_RCA`

### D. Cardiovascular Landmarks & Cavity Context
- **Ascending Aorta & Left Main Stem (`Aorta_LM`)**:
  - `FMA23733` / `BP51985`: Ascending aorta proper
  - `FMA15098` / `BP51975`: Wall of bulb of aorta
  - `FMA4685` / `BP58405`: Stem of left coronary artery (Left Main / LMCA)
- **Myocardium & Cardiac Chambers (`Myocardium`)**:
  - `FMA84850` / `BP51876`: Free wall of left ventricle
  - `FMA49207` / `BP51861`: Wall of inflow part of right ventricle
  - `FMA49285` / `BP51843`: Anterior wall of right atrium
  - `FMA9306` / `BP51862`: Lateral wall of left atrium
  - `FMA9560` / `BP52011`: Anterior wall of left ventricle
- **Thoracic Silhouette (`Torso_Cage`)**:
  - Semi-transparent thoracic cage contour placing the cardiac organ in correct anatomical orientation within the chest cavity.

---

## 3. Critical Safety & Clinical Disclaimer

> [!CRITICAL]
> **NOT AN IMAGING SEGMENTATION OR LESION LOCALIZER**
>
> 1. **No Lesion Coordinates**: The underlying dataset (UCI ID 411: Extension of Z-Alizadeh Sani CAD Dataset) provides tabular clinical, laboratory, ECG, and echocardiographic variables. It does **NOT** contain DICOM CT angiography or invasive coronary angiography (ICA) lesion coordinates, pixel-level segmentations, or 3D stenosis geometries.
> 2. **No Arbitrary Blockage Markers**: CardioVision AI **does not** plot fake blockage dots, red stenosis spheres, or arbitrary geometric obstructions along the vessel mesh. Doing so would be clinically deceptive.
> 3. **True Meaning of Visualization**: The color mapping on `Vessel_LAD`, `Vessel_LCX`, and `Vessel_RCA` represents **"Model-predicted vessel-level risk"** derived from the trained machine learning classifiers consuming patient clinical features.
> 4. **Scale Definition**: The 0–100 scale is a **Visualization scale — not a clinical severity classification** or luminal percentage blockage measurement.

---

## 4. 3D Technology Stack & Architecture

- **Rendering Engine**: Three.js (v0.177+)
- **React Bindings**: React Three Fiber (`@react-three/fiber` v9+)
- **Helper Utilities**: `@react-three/drei` (v10+)
- **Model Distribution**: Optimized binary glTF 2.0 (`.glb`) stored in `public/models/heart_anatomy_bp3d.glb`
- **Asset Size & Performance Budget**:
  - Asset polygon budget: Low-to-moderate polygon count (< 40,000 triangles total)
  - Zero dedicated GPU requirement; fully fluid 60fps on integrated graphics and mobile web browsers.
  - Lazy loading with `React.Suspense` and Drei `useGLTF.preload`.
  - Normal smoothing and vertex buffer interleaving for optimal memory footprint.

---

## 5. Vessel Registry Architecture & Maintenance

The vessel registry in `src/services/vesselRegistry.ts` maps each coronary artery to:
1. `meshId`: Exact scene graph node in the GLB asset.
2. `fmaId`: Foundational Model of Anatomy standard ontology term.
3. `bpId`: BodyParts3D concept identifier.
4. `taskKey`: Target name in API payload (`'LAD'`, `'LCX'`, `'RCA'`).
5. `focusCamera`: Preset camera coordinates (position, target, FOV) for animated zoom and clinical inspection.
6. `anatomicalTerritory`: Specific myocardial territory supplied by the vessel.
