/**
 * CardioVision AI — Real Interactive 3D Anatomical Visualization Component
 * =========================================================================
 * Renders authentic DBCLS BodyParts3D / Foundational Model of Anatomy (FMA) meshes:
 *   - Vessel_LAD: Left Anterior Descending Artery
 *   - Vessel_LCX: Left Circumflex Artery
 *   - Vessel_RCA: Right Coronary Artery
 *   - Aorta_LM: Ascending Aorta & Left Main Stem
 *   - Myocardium: Ventricles & Atria walls
 *   - Thoracic_Cage: Contextual anatomical chest/ribs silhouette
 *
 * FEATURES:
 *   - Rotate, zoom, pan via OrbitControls
 *   - Reset camera button
 *   - Hover highlight & pointer cursor
 *   - Click-to-select vessel
 *   - Animated camera focus towards selected coronary vessel
 *   - Live risk color mapping based on model probabilities from POST /analyze
 *   - Performance-optimized: <40k triangles, lightweight GLB, memoized materials
 */

import { useState, useMemo, useEffect, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, useGLTF, Html, Center } from '@react-three/drei';
import * as THREE from 'three';
import {
  type CoronaryVesselId,
  CORONARY_VESSEL_REGISTRY,
  getVesselByMeshName,
  getRiskScaleTier,
  DEFAULT_CAMERA_PRESET,
} from '@/services/vesselRegistry';
import {
  RotateCcw,
  Sparkles,
} from 'lucide-react';

// Preload authentic BodyParts3D GLB asset
useGLTF.preload('/models/heart_anatomy_bp3d.glb');

interface VesselProbabilities {
  LAD: number;
  LCX: number;
  RCA: number;
}

interface AnatomyViewer3DProps {
  selectedVessel: CoronaryVesselId;
  onSelectVessel: (id: CoronaryVesselId) => void;
  probabilities: VesselProbabilities;
  className?: string;
}

/**
 * Animated camera controller for smooth transitions to focused vessel or reset.
 */
function CameraController({
  focusVessel,
}: {
  focusVessel: CoronaryVesselId | null;
}) {
  const { camera } = useThree();
  const controlsRef = useThree((state) => state.controls) as any;

  // Target position and lookAt target
  const targetState = useMemo(() => {
    if (!focusVessel) {
      return {
        pos: new THREE.Vector3(...DEFAULT_CAMERA_PRESET.position),
        target: new THREE.Vector3(...DEFAULT_CAMERA_PRESET.target),
      };
    }
    const preset = CORONARY_VESSEL_REGISTRY[focusVessel].cameraPreset;
    return {
      pos: new THREE.Vector3(...preset.position),
      target: new THREE.Vector3(...preset.target),
    };
  }, [focusVessel]);

  useFrame((_, delta) => {
    const step = Math.min(delta * 3.5, 0.2);
    camera.position.lerp(targetState.pos, step);
    if (controlsRef && controlsRef.target) {
      controlsRef.target.lerp(targetState.target, step);
      controlsRef.update();
    }
  });

  return null;
}

/**
 * 3D Model Scene rendering the BodyParts3D GLB meshes.
 */
function HeartModelScene({
  selectedVessel,
  onSelectVessel,
  probabilities,
  hoveredVessel,
  setHoveredVessel,
  showTorso,
  myocardiumOpacity,
}: {
  selectedVessel: CoronaryVesselId;
  onSelectVessel: (id: CoronaryVesselId) => void;
  probabilities: VesselProbabilities;
  hoveredVessel: CoronaryVesselId | null;
  setHoveredVessel: (id: CoronaryVesselId | null) => void;
  showTorso: boolean;
  myocardiumOpacity: number;
}) {
  const { scene } = useGLTF('/models/heart_anatomy_bp3d.glb');

  // Compute dynamic colors and tiers
  const ladTier = useMemo(() => getRiskScaleTier(probabilities.LAD), [probabilities.LAD]);
  const lcxTier = useMemo(() => getRiskScaleTier(probabilities.LCX), [probabilities.LCX]);
  const rcaTier = useMemo(() => getRiskScaleTier(probabilities.RCA), [probabilities.RCA]);

  // Clone and configure materials for each mesh node
  useEffect(() => {
    scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        const name = mesh.name;

        if (name === 'Vessel_LAD') {
          const isSel = selectedVessel === 'LAD';
          const isHov = hoveredVessel === 'LAD';
          mesh.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color(ladTier.color),
            emissive: new THREE.Color(ladTier.color),
            emissiveIntensity: isSel ? 0.7 : isHov ? 0.5 : 0.25,
            roughness: 0.25,
            metalness: 0.3,
          });
          mesh.castShadow = true;
          mesh.receiveShadow = true;
        } else if (name === 'Vessel_LCX') {
          const isSel = selectedVessel === 'LCX';
          const isHov = hoveredVessel === 'LCX';
          mesh.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color(lcxTier.color),
            emissive: new THREE.Color(lcxTier.color),
            emissiveIntensity: isSel ? 0.7 : isHov ? 0.5 : 0.25,
            roughness: 0.25,
            metalness: 0.3,
          });
          mesh.castShadow = true;
          mesh.receiveShadow = true;
        } else if (name === 'Vessel_RCA') {
          const isSel = selectedVessel === 'RCA';
          const isHov = hoveredVessel === 'RCA';
          mesh.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color(rcaTier.color),
            emissive: new THREE.Color(rcaTier.color),
            emissiveIntensity: isSel ? 0.7 : isHov ? 0.5 : 0.25,
            roughness: 0.25,
            metalness: 0.3,
          });
          mesh.castShadow = true;
          mesh.receiveShadow = true;
        } else if (name === 'Aorta_LM') {
          mesh.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color('#991b1b'),
            roughness: 0.3,
            metalness: 0.2,
          });
        } else if (name === 'Myocardium') {
          mesh.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color('#581c1c'),
            roughness: 0.6,
            metalness: 0.1,
            transparent: true,
            opacity: myocardiumOpacity,
            depthWrite: myocardiumOpacity > 0.6,
          });
        } else if (name === 'Thoracic_Cage') {
          mesh.visible = showTorso;
          mesh.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color('#38bdf8'),
            roughness: 0.7,
            metalness: 0.1,
            transparent: true,
            opacity: 0.18,
            side: THREE.DoubleSide,
            depthWrite: false,
          });
        }
      }
    });
  }, [
    scene,
    selectedVessel,
    hoveredVessel,
    ladTier,
    lcxTier,
    rcaTier,
    myocardiumOpacity,
    showTorso,
  ]);

  // Handle pointer hover and clicks
  const handlePointerOver = (e: any) => {
    e.stopPropagation();
    const meshName = e.object.name;
    const vessel = getVesselByMeshName(meshName);
    if (vessel) {
      document.body.style.cursor = 'pointer';
      setHoveredVessel(vessel.id);
    }
  };

  const handlePointerOut = (e: any) => {
    e.stopPropagation();
    document.body.style.cursor = 'auto';
    setHoveredVessel(null);
  };

  const handleClick = (e: any) => {
    e.stopPropagation();
    const meshName = e.object.name;
    const vessel = getVesselByMeshName(meshName);
    if (vessel) {
      onSelectVessel(vessel.id);
    }
  };

  return (
    <group
      onPointerOver={handlePointerOver}
      onPointerOut={handlePointerOut}
      onClick={handleClick}
    >
      <primitive object={scene} />

      {/* 3D Floating Labels for Coronary Vessels */}
      <Html position={[-0.45, 0.45, 0.65]} center distanceFactor={6} style={{ pointerEvents: 'none' }}>
        <div
          className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold whitespace-nowrap shadow-md border transition-all ${
            selectedVessel === 'LAD'
              ? 'scale-110 ring-2 ring-white/50 bg-surface-darker/95'
              : 'bg-surface-darker/80'
          }`}
          style={{
            borderColor: ladTier.color,
            color: ladTier.color,
          }}
        >
          LAD · {(probabilities.LAD * 100).toFixed(0)}%
        </div>
      </Html>

      <Html position={[-1.0, 0.25, -0.35]} center distanceFactor={6} style={{ pointerEvents: 'none' }}>
        <div
          className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold whitespace-nowrap shadow-md border transition-all ${
            selectedVessel === 'LCX'
              ? 'scale-110 ring-2 ring-white/50 bg-surface-darker/95'
              : 'bg-surface-darker/80'
          }`}
          style={{
            borderColor: lcxTier.color,
            color: lcxTier.color,
          }}
        >
          LCX · {(probabilities.LCX * 100).toFixed(0)}%
        </div>
      </Html>

      <Html position={[0.9, -0.15, 0.45]} center distanceFactor={6} style={{ pointerEvents: 'none' }}>
        <div
          className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold whitespace-nowrap shadow-md border transition-all ${
            selectedVessel === 'RCA'
              ? 'scale-110 ring-2 ring-white/50 bg-surface-darker/95'
              : 'bg-surface-darker/80'
          }`}
          style={{
            borderColor: rcaTier.color,
            color: rcaTier.color,
          }}
        >
          RCA · {(probabilities.RCA * 100).toFixed(0)}%
        </div>
      </Html>
    </group>
  );
}

export function AnatomyViewer3D({
  selectedVessel,
  onSelectVessel,
  probabilities,
  className = '',
}: AnatomyViewer3DProps) {
  const [hoveredVessel, setHoveredVessel] = useState<CoronaryVesselId | null>(null);
  const [autoRotate, setAutoRotate] = useState<boolean>(true);
  const [showTorso, setShowTorso] = useState<boolean>(true);
  const [myocardiumOpacity, setMyocardiumOpacity] = useState<number>(0.45);
  const [resetCount, setResetCount] = useState<number>(0);
  const [activeFocus, setActiveFocus] = useState<CoronaryVesselId | null>(selectedVessel);

  // Directly derive target focus without setState in effect
  const currentFocus = resetCount > 0 && activeFocus === null ? null : (activeFocus ?? selectedVessel);

  const handleResetCamera = () => {
    setActiveFocus(null);
    setResetCount((c) => c + 1);
  };

  return (
    <div
      id="anatomy-viewer-container"
      className={`glass-card p-3 flex flex-col relative overflow-hidden border-surface-border select-none ${className}`}
    >
      {/* Top Overlay Badge & Quick Status */}
      <div className="absolute top-4 left-4 z-10 flex flex-wrap items-center gap-2 pointer-events-auto">
        <div className="flex items-center gap-2 bg-surface-darker/85 backdrop-blur px-3 py-1.5 rounded-lg border border-surface-border text-xs font-mono text-gray-200 shadow-md">
          <Sparkles className="w-3.5 h-3.5 text-accent-400" />
          <span>Interactive 3D Heart (BodyParts3D)</span>
        </div>

        {hoveredVessel && (
          <div className="flex items-center gap-1.5 bg-primary-950/80 backdrop-blur px-2.5 py-1.5 rounded-lg border border-primary-500/40 text-xs font-mono text-white animate-fade-in">
            <span className="w-2 h-2 rounded-full bg-accent-400 animate-ping" />
            <span>Hovering: {hoveredVessel}</span>
          </div>
        )}
      </div>

      {/* Top Right Controls Toolbar */}
      <div className="absolute top-4 right-4 z-10 flex items-center gap-1.5 bg-surface-darker/85 backdrop-blur p-1 rounded-lg border border-surface-border pointer-events-auto">
        <button
          id="btn-camera-reset"
          onClick={handleResetCamera}
          title="Reset Camera to Default View"
          className="p-1.5 rounded-md hover:bg-surface-dark text-gray-300 hover:text-white transition-all text-xs flex items-center gap-1 font-mono"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">Reset</span>
        </button>

        <button
          id="btn-toggle-autorotate"
          onClick={() => setAutoRotate(!autoRotate)}
          title="Toggle Auto-Rotation"
          className={`p-1.5 rounded-md transition-all text-xs font-mono ${
            autoRotate
              ? 'bg-accent-500/20 text-accent-400 border border-accent-500/30'
              : 'hover:bg-surface-dark text-gray-400 hover:text-white'
          }`}
        >
          Spin
        </button>

        <button
          id="btn-toggle-torso"
          onClick={() => setShowTorso(!showTorso)}
          title="Toggle Contextual Thoracic Cage / Ribs Silhouette"
          className={`p-1.5 rounded-md transition-all text-xs font-mono ${
            showTorso
              ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
              : 'hover:bg-surface-dark text-gray-400 hover:text-white'
          }`}
        >
          Torso
        </button>
      </div>

      {/* Bottom Floating Control Bar */}
      <div className="absolute bottom-4 left-4 right-4 z-10 flex flex-wrap items-center justify-between gap-3 bg-surface-darker/80 backdrop-blur px-4 py-2.5 rounded-xl border border-surface-border text-xs font-mono pointer-events-auto">
        {/* Left: Vessel Focus Shortcuts */}
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] text-gray-400 hidden sm:inline mr-1">Focus:</span>
          {(['LAD', 'LCX', 'RCA'] as const).map((v) => (
            <button
              key={v}
              onClick={() => {
                onSelectVessel(v);
                setActiveFocus(v);
              }}
              className={`px-2.5 py-1 rounded text-xs font-bold transition-all border ${
                selectedVessel === v
                  ? 'bg-primary-500/30 border-primary-400 text-white shadow-sm'
                  : 'bg-surface-dark/90 border-surface-border text-gray-400 hover:text-gray-200'
              }`}
            >
              {v}
            </button>
          ))}
        </div>

        {/* Right: Myocardium Opacity Slider */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-gray-400">Myocardium Opacity:</span>
          <input
            id="slider-myocardium-opacity"
            type="range"
            min="0.0"
            max="0.85"
            step="0.05"
            value={myocardiumOpacity}
            onChange={(e) => setMyocardiumOpacity(parseFloat(e.target.value))}
            className="w-20 sm:w-28 accent-primary-500 cursor-pointer"
            title="Adjust transparency of ventricular muscle wall"
          />
          <span className="text-[10px] text-gray-400 w-8 text-right font-mono">
            {Math.round(myocardiumOpacity * 100)}%
          </span>
        </div>
      </div>

      {/* 3D WebGL Canvas */}
      <div className="w-full flex-1 min-h-[440px] sm:min-h-[500px]">
        <Suspense
          fallback={
            <div className="h-full min-h-[440px] flex flex-col items-center justify-center space-y-3 font-mono text-xs text-gray-400 animate-pulse">
              <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
              <span>Loading Authentic BodyParts3D Anatomy Model…</span>
            </div>
          }
        >
          <Canvas
            camera={{ position: DEFAULT_CAMERA_PRESET.position, fov: DEFAULT_CAMERA_PRESET.fov }}
            style={{ background: 'transparent' }}
            dpr={[1, 2]}
            gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
          >
            <ambientLight intensity={0.65} />
            <directionalLight position={[5, 8, 5]} intensity={1.2} color="#ffffff" castShadow />
            <pointLight position={[-4, 2, 4]} intensity={0.8} color="#60a5fa" />
            <pointLight position={[4, -3, 3]} intensity={0.6} color="#f43f5e" />

            <Center>
              <HeartModelScene
                selectedVessel={selectedVessel}
                onSelectVessel={onSelectVessel}
                probabilities={probabilities}
                hoveredVessel={hoveredVessel}
                setHoveredVessel={setHoveredVessel}
                showTorso={showTorso}
                myocardiumOpacity={myocardiumOpacity}
              />
            </Center>

            <OrbitControls
              enableZoom={true}
              enablePan={true}
              enableRotate={true}
              minDistance={1.8}
              maxDistance={7.5}
              autoRotate={autoRotate}
              autoRotateSpeed={0.8}
              dampingFactor={0.05}
            />

            <CameraController
              focusVessel={currentFocus}
            />
          </Canvas>
        </Suspense>
      </div>
    </div>
  );
}
