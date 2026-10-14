# CardioVision AI — Frontend Application

Interactive clinical cardiovascular AI decision-support platform built for **Multimodal AI Hackathon 2026 — Cardiovascular Track**.

## Key Features & 3D Visualization

- **Interactive 3D Anatomical Heart & Coronary Viewer**:
  - Authentic 3D polygon geometry from **BodyParts3D / Anatomography by DBCLS** (CC BY-SA 2.1 Japan).
  - Explicit mesh nodes: `Vessel_LAD`, `Vessel_LCX`, `Vessel_RCA`, `Aorta_LM`, `Myocardium`, and contextual `Thoracic_Cage`.
  - OrbitControls (rotate, zoom, pan), reset camera, hover highlight, click-to-select, and animated vessel focus.
  - Transparent Myocardium slider and Torso cage visibility toggle for internal coronary inspection.
  - Performance-optimized binary glTF 2.0 (`.glb`) with <40,000 polygon count and 60fps rendering without requiring dedicated GPUs.
- **Maintainable Vessel Registry**:
  - Maps LAD, LCX, RCA to Foundational Model of Anatomy (FMA) identifiers and BodyParts3D concept IDs.
- **Real ML Predictions Integration**:
  - Dynamically consumes real model probabilities from `POST /analyze`.
  - Displays tuned decision thresholds, model classification, and per-vessel TreeSHAP feature attributions.
- **Safety Boundaries**:
  - Clearly labeled non-clinical visualization scale (0–30, 30–60, 60–80, 80–100).
  - Explicit disclaimer: Values represent model-predicted risk probabilities, **NOT anatomical percentage blockage or lesion coordinates**.

## Technology Stack

- **Framework**: React 19 + TypeScript + Vite 8
- **3D Graphics**: Three.js + React Three Fiber (`@react-three/fiber`) + Drei (`@react-three/drei`)
- **Styling**: Tailwind CSS + Custom Dark Theme
- **Data Visualization**: Recharts + Custom 3D Shaders

## Running Locally

```bash
# Install dependencies
npm install

# Start Vite dev server
npm run dev

# Check types
npm run type-check

# Production build
npm run build
```
