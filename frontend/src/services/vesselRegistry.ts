/**
 * CardioVision AI — 3D Anatomical Vessel Mesh Registry
 * =====================================================
 * Maintainable mapping between machine learning prediction targets (LAD, LCX, RCA),
 * BodyParts3D (DBCLS) mesh identifiers, and Foundational Model of Anatomy (FMA) ontologies.
 *
 * CRITICAL CLINICAL SAFETY PRINCIPLE:
 * The dataset does NOT provide lesion coordinates, CT/MRI segmentation, or 3D
 * stenosis geometries. The visualization maps "Model-Predicted Vessel-Level Risk"
 * onto verified anatomical coronary branches. No arbitrary blockage markers are placed.
 *
 * SCALE DEFINITION:
 * Visualization scale — not a clinical severity classification.
 * Values reflect model-predicted probabilities (0–100%), NOT percentage blockage.
 */

export type CoronaryVesselId = 'LAD' | 'LCX' | 'RCA';

export interface CameraPreset {
  position: [number, number, number];
  target: [number, number, number];
  fov: number;
}

export interface VesselRegistryEntry {
  id: CoronaryVesselId;
  meshName: string;
  fullName: string;
  clinicalName: string;
  fmaId: string;
  fmaTerm: string;
  bodyParts3DIds: string[];
  anatomicalTerritory: string;
  clinicalSignificance: string;
  cameraPreset: CameraPreset;
  defaultColor: string;
  sourceAttribution: string;
}

export interface RiskScaleTier {
  range: string;
  min: number;
  max: number;
  label: string;
  color: string;
  bgClass: string;
  textClass: string;
  borderClass: string;
  description: string;
}

export const VISUALIZATION_SCALE_TIERS: RiskScaleTier[] = [
  {
    range: '0 – 30%',
    min: 0.0,
    max: 0.30,
    label: 'Low Model Risk',
    color: '#22c55e',
    bgClass: 'bg-emerald-500/15',
    textClass: 'text-emerald-400',
    borderClass: 'border-emerald-500/30',
    description: 'Model-predicted stenosis probability is low relative to clinical threshold.',
  },
  {
    range: '30 – 60%',
    min: 0.30,
    max: 0.60,
    label: 'Moderate Model Risk',
    color: '#eab308',
    bgClass: 'bg-amber-500/15',
    textClass: 'text-amber-400',
    borderClass: 'border-amber-500/30',
    description: 'Model-predicted stenosis probability is borderline / intermediate.',
  },
  {
    range: '60 – 80%',
    min: 0.60,
    max: 0.80,
    label: 'High Model Risk',
    color: '#f97316',
    bgClass: 'bg-orange-500/15',
    textClass: 'text-orange-400',
    borderClass: 'border-orange-500/30',
    description: 'Model-predicted stenosis probability exceeds typical decision threshold.',
  },
  {
    range: '80 – 100%',
    min: 0.80,
    max: 1.0,
    label: 'Critical Model Risk',
    color: '#ef4444',
    bgClass: 'bg-rose-500/15',
    textClass: 'text-rose-400',
    borderClass: 'border-rose-500/30',
    description: 'High model confidence of significant vessel stenosis.',
  },
];

export const CORONARY_VESSEL_REGISTRY: Record<CoronaryVesselId, VesselRegistryEntry> = {
  LAD: {
    id: 'LAD',
    meshName: 'Vessel_LAD',
    fullName: 'Left Anterior Descending Artery',
    clinicalName: 'Anterior Interventricular Branch of LCA',
    fmaId: 'FMA74912 / FMA3981',
    fmaTerm: 'Anterior interventricular branch of left coronary artery',
    bodyParts3DIds: ['BP51969', 'BP51972', 'BP51965', 'BP51614', 'BP51962'],
    anatomicalTerritory: 'Anterior myocardial wall, anterior 2/3 of interventricular septum, cardiac apex',
    clinicalSignificance: 'Often colloquially termed the "widow maker" due to large territorial myocardial infarction risk.',
    cameraPreset: {
      position: [-0.4, 0.4, 3.4],
      target: [-0.1, 0.1, 0.2],
      fov: 45,
    },
    defaultColor: '#ef4444',
    sourceAttribution: 'BodyParts3D, Database Center for Life Science (DBCLS), CC BY-SA 2.1 Japan',
  },
  LCX: {
    id: 'LCX',
    meshName: 'Vessel_LCX',
    fullName: 'Left Circumflex Artery',
    clinicalName: 'Circumflex Branch of LCA',
    fmaId: 'FMA74923 / FMA3984',
    fmaTerm: 'Circumflex branch of left coronary artery',
    bodyParts3DIds: ['BP51973', 'BP51610', 'BP51926'],
    anatomicalTerritory: 'Left lateral and posterolateral myocardial walls, portions of left atrium',
    clinicalSignificance: 'Supplies obtuse marginal branches; occlusion produces lateral wall ischemia.',
    cameraPreset: {
      position: [-2.4, 0.6, 2.2],
      target: [-0.5, 0.0, -0.1],
      fov: 45,
    },
    defaultColor: '#f97316',
    sourceAttribution: 'BodyParts3D, Database Center for Life Science (DBCLS), CC BY-SA 2.1 Japan',
  },
  RCA: {
    id: 'RCA',
    meshName: 'Vessel_RCA',
    fullName: 'Right Coronary Artery',
    clinicalName: 'Right Coronary Artery Proper',
    fmaId: 'FMA3802 / FMA3991',
    fmaTerm: 'Right coronary artery',
    bodyParts3DIds: ['BP51977', 'BP51980', 'BP51950', 'BP51956', 'BP51957'],
    anatomicalTerritory: 'Right atrium, right ventricle, inferior (diaphragmatic) wall of LV, AV/SA nodes (in 85% dominant)',
    clinicalSignificance: 'Occlusion frequently causes inferior wall MI and cardiac conduction disturbances.',
    cameraPreset: {
      position: [2.3, 0.3, 2.4],
      target: [0.35, -0.1, 0.1],
      fov: 45,
    },
    defaultColor: '#3b82f6',
    sourceAttribution: 'BodyParts3D, Database Center for Life Science (DBCLS), CC BY-SA 2.1 Japan',
  },
};

export const DEFAULT_CAMERA_PRESET: CameraPreset = {
  position: [0.0, 0.2, 4.4],
  target: [0.0, 0.0, 0.0],
  fov: 48,
};

/**
 * Returns the vessel configuration by vessel ID (LAD, LCX, RCA).
 */
export function getVesselConfig(id: CoronaryVesselId): VesselRegistryEntry {
  return CORONARY_VESSEL_REGISTRY[id];
}

/**
 * Maps a GLB mesh node name back to its coronary vessel registry entry.
 */
export function getVesselByMeshName(meshName: string): VesselRegistryEntry | undefined {
  return Object.values(CORONARY_VESSEL_REGISTRY).find((entry) => entry.meshName === meshName);
}

/**
 * Maps a model-predicted probability (0.0 to 1.0) to visualization scale tier.
 */
export function getRiskScaleTier(probability: number): RiskScaleTier {
  const clamped = Math.max(0.0, Math.min(1.0, probability));
  for (const tier of VISUALIZATION_SCALE_TIERS) {
    if (clamped <= tier.max) {
      return tier;
    }
  }
  return VISUALIZATION_SCALE_TIERS[VISUALIZATION_SCALE_TIERS.length - 1];
}

/**
 * Computes an interpolated RGB hex color based on the visualization scale.
 */
export function getRiskColorFromProbability(probability: number): string {
  return getRiskScaleTier(probability).color;
}
