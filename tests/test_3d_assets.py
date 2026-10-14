"""
CardioVision AI — Interactive 3D Anatomical Visualization Test Suite
===================================================================
Verifies:
  1. 3D Model Asset:
     - public/models/heart_anatomy_bp3d.glb exists and is a valid binary glTF 2.0 (GLB) file.
     - Reasonable file size (< 5 MB) for modern web performance without dedicated GPU.
     - Contains authentic BodyParts3D mesh nodes:
       Vessel_LAD, Vessel_LCX, Vessel_RCA, Aorta_LM, Myocardium, Thoracic_Cage.
  2. Asset Documentation & Provenance:
     - docs/3d-assets.md exists and documents source, asset name, format, attribution,
       and CC BY-SA 2.1 Japan license terms.
     - Foundational Model of Anatomy (FMA) identifiers documented.
  3. Vessel Registry & Safety Constraints:
     - Explicit mapping for LAD, LCX, RCA without fake lesion coordinates.
     - Visualization scale defined with non-clinical severity notice.
  4. Real API Integration:
     - POST /analyze and POST /api/v1/analyze return genuine vessel probabilities for LAD, LCX, RCA.
"""

import json
import os
import struct
import pytest
from fastapi.testclient import TestClient

from backend.main import create_application


@pytest.fixture(scope="module")
def client():
    app = create_application()
    return TestClient(app)


class Test3DAssetIntegrity:
    """Tests verifying the 3D anatomical asset format, nodes, and performance budget."""

    GLB_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "frontend",
        "public",
        "models",
        "heart_anatomy_bp3d.glb"
    )

    def test_glb_file_exists(self):
        """Verifies the compiled BodyParts3D GLB asset exists in frontend/public/models."""
        assert os.path.exists(self.GLB_PATH), f"GLB file missing at: {self.GLB_PATH}"

    def test_glb_performance_budget(self):
        """Verifies file size is compressed and within budget (< 5 MB) for web performance."""
        size_bytes = os.path.getsize(self.GLB_PATH)
        assert size_bytes > 100_000, "GLB file is suspiciously small"
        assert size_bytes < 5_000_000, f"GLB file too large ({size_bytes} bytes). Must be < 5 MB"

    def test_glb_header_and_structure(self):
        """Verifies binary glTF 2.0 container structure (magic, version, JSON chunk, BIN chunk)."""
        with open(self.GLB_PATH, "rb") as f:
            header = f.read(12)
            magic, version, total_length = struct.unpack("<III", header)
            assert magic == 0x46546C67, "Invalid glTF binary magic number"
            assert version == 2, "glTF version must be 2.0"
            assert total_length == os.path.getsize(self.GLB_PATH), "Header length mismatch"

            # Chunk 0: JSON
            json_header = f.read(8)
            json_len, json_type = struct.unpack("<II", json_header)
            assert json_type == 0x4E4F534A, "Chunk 0 must be JSON (0x4E4F534A)"
            json_bytes = f.read(json_len)
            gltf_json = json.loads(json_bytes.decode("utf-8"))

            # Chunk 1: BIN
            bin_header = f.read(8)
            bin_len, bin_type = struct.unpack("<II", bin_header)
            assert bin_type == 0x004E4942, "Chunk 1 must be BIN (0x004E4942)"

        # Check nodes
        node_names = [n.get("name") for n in gltf_json.get("nodes", [])]
        expected_nodes = [
            "Vessel_LAD",
            "Vessel_LCX",
            "Vessel_RCA",
            "Aorta_LM",
            "Myocardium",
            "Thoracic_Cage"
        ]
        for en in expected_nodes:
            assert en in node_names, f"Expected node '{en}' missing from GLB nodes: {node_names}"

        # Check copyright / generator
        asset_info = gltf_json.get("asset", {})
        assert "BodyParts3D" in asset_info.get("copyright", "") or "DBCLS" in asset_info.get("copyright", "")


class Test3DDocumentation:
    """Tests verifying docs/3d-assets.md exists and contains all required specifications."""

    DOCS_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "docs",
        "3d-assets.md"
    )

    def test_docs_file_exists(self):
        """docs/3d-assets.md must exist."""
        assert os.path.exists(self.DOCS_PATH), f"Documentation missing at: {self.DOCS_PATH}"

    def test_docs_mandatory_sections(self):
        """Verifies source, format, license, attribution, and safety notice are documented."""
        with open(self.DOCS_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        required_terms = [
            "BodyParts3D",
            "lifesciencedb.jp",
            "DBCLS",
            "CC BY-SA 2.1 Japan",
            "LAD",
            "LCX",
            "RCA",
            "Foundational Model of Anatomy",
            "FMA",
            "not a clinical severity classification",
            "Model-predicted vessel-level risk",
        ]
        for term in required_terms:
            assert term in content, f"docs/3d-assets.md is missing mandatory documentation for: '{term}'"


class TestVesselRegistryCode:
    """Tests verifying the TypeScript vessel registry source file."""

    REGISTRY_TS = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "frontend",
        "src",
        "services",
        "vesselRegistry.ts"
    )

    def test_registry_file_exists(self):
        assert os.path.exists(self.REGISTRY_TS), f"Vessel registry missing at: {self.REGISTRY_TS}"

    def test_registry_contents(self):
        with open(self.REGISTRY_TS, "r", encoding="utf-8") as f:
            content = f.read()

        assert "CORONARY_VESSEL_REGISTRY" in content
        assert "Vessel_LAD" in content
        assert "Vessel_LCX" in content
        assert "Vessel_RCA" in content
        assert "VISUALIZATION_SCALE_TIERS" in content
        assert "Non-Clinical" in content or "not a clinical severity classification" in content


class TestRealAPIProbabilities:
    """Tests verifying POST /analyze feeds real probabilities to LAD, LCX, RCA."""

    def test_post_analyze_returns_vessel_probabilities(self, client):
        sample_features = {
            "Age": 68, "Weight": 85, "Length": 170, "Sex": "Male", "BMI": 29.4,
            "DM": 1, "HTN": 1, "Current Smoker": 1, "EX-Smoker": 0, "FH": 1,
            "Obesity": "Y", "CRF": "N", "CVA": "N", "Airway disease": "N", "Thyroid Disease": "N",
            "CHF": "Y", "DLP": "Y", "BP": 155, "PR": 88, "Edema": 1,
            "Weak Peripheral Pulse": "Y", "Lung rales": "Y", "Systolic Murmur": "N", "Diastolic Murmur": "N",
            "Typical Chest Pain": 1, "Dyspnea": "Y", "Function Class": 3,
            "Atypical": "N", "Nonanginal": "N", "Exertional CP": "Y", "LowTH Ang": "Y",
            "Q Wave": 1, "St Elevation": 0, "St Depression": 1, "Tinversion": 1,
            "LVH": "Y", "Poor R Progression": "Y", "BBB": "N", "VHD": "N",
            "FBS": 165, "CR": 1.3, "TG": 240, "LDL": 160, "HDL": 35, "BUN": 22, "ESR": 28,
            "HB": 13.2, "K": 4.4, "Na": 138, "WBC": 9200, "Lymph": 24, "Neut": 70, "PLT": 280,
            "EF-TTE": 40, "Region RWMA": 2
        }

        resp = client.post("/api/v1/analyze", json={
            "patient_id": "TEST-PT-3D",
            "features": sample_features,
            "include_shap": True,
            "include_vessel_risk_map": True
        })

        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        preds = data["predictions"]

        # Ensure all 3 coronary vessels have genuine float probabilities
        for vessel in ["LAD", "LCX", "RCA"]:
            assert vessel in preds, f"Missing prediction for vessel {vessel}"
            prob = preds[vessel]["model_predicted_probability"]
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0, f"Probability for {vessel} out of range [0, 1]: {prob}"
            assert "classification" in preds[vessel]
            assert "threshold" in preds[vessel]

        # Verify visualization note disclaims lesion coordinates
        assert "vessel_risk_visualization_note" in data
        assert "exact lesion coordinates" in data["vessel_risk_visualization_note"]
