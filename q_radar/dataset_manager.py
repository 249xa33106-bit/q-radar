import os
import time
import pandas as pd
import cv2
import numpy as np

class DatasetManager:
    def __init__(self):
        self.assets_dir = "assets"
        self._ensure_sample_assets()

    def _ensure_sample_assets(self):
        """Ensures sample images exist in assets/ directory."""
        if not os.path.exists(os.path.join(self.assets_dir, "xr_pneumothorax.png")):
            from generate_sample_assets import create_preset_assets
            create_preset_assets()

    def get_preset_samples(self):
        """Returns pre-configured sample medical image presets for instant 1-click testing."""
        return {
            "Pneumothorax Apex (Critical 91%)": {
                "id": "XR-104",
                "patient_id": "ANON-2841",
                "image_type": "Chest X-Ray (AP)",
                "department": "Emergency Dept",
                "symptoms": "Acute dyspnea, right pleuritic chest pain",
                "image_path": os.path.join(self.assets_dir, "xr_pneumothorax.png"),
                "expected_pathology": "Pneumothorax"
            },
            "Cardiomegaly & Effusion (Critical 84%)": {
                "id": "XR-116",
                "patient_id": "ANON-9032",
                "image_type": "Chest X-Ray (PA)",
                "department": "ICU",
                "symptoms": "Orthopnea, bilateral lower extremity edema",
                "image_path": os.path.join(self.assets_dir, "xr_cardiomegaly.png"),
                "expected_pathology": "Cardiomegaly"
            },
            "Pulmonary Consolidation (High 71%)": {
                "id": "XR-087",
                "patient_id": "ANON-4419",
                "image_type": "Chest X-Ray (PA)",
                "department": "Pulmonology",
                "symptoms": "Fever 38.8°C, productive cough, leukocytosis",
                "image_path": os.path.join(self.assets_dir, "xr_consolidation.png"),
                "expected_pathology": "Consolidation"
            },
            "Pleural Effusion (Review 56%)": {
                "id": "XR-121",
                "patient_id": "ANON-1550",
                "image_type": "Chest X-Ray (Lateral)",
                "department": "General Internal Med",
                "symptoms": "Dullness to percussion left lung base",
                "image_path": os.path.join(self.assets_dir, "xr_effusion.png"),
                "expected_pathology": "Pleural Effusion"
            },
            "Clear Normal Thorax (Low 13%)": {
                "id": "XR-099",
                "patient_id": "ANON-6701",
                "image_type": "Chest X-Ray (PA)",
                "department": "Outpatient Triage",
                "symptoms": "Routine employment screening",
                "image_path": os.path.join(self.assets_dir, "xr_normal.png"),
                "expected_pathology": "Normal"
            }
        }

    def get_initial_radiology_queue(self):
        """Initializes the Smart Radiology Queue with pre-analyzed benchmark cases."""
        return [
            {
                "scan_id": "XR-104",
                "patient_id": "ANON-2841",
                "image_type": "Chest X-Ray (AP)",
                "department": "Emergency Dept",
                "symptoms": "Acute dyspnea, right pleuritic chest pain",
                "anomaly_score": 91.4,
                "urgency": "CRITICAL",
                "badge": "🔴",
                "status": "Review Now",
                "reviewed": False,
                "timestamp": "19:04:12",
                "image_path": os.path.join(self.assets_dir, "xr_pneumothorax.png"),
                "expected_pathology": "Pneumothorax"
            },
            {
                "scan_id": "XR-116",
                "patient_id": "ANON-9032",
                "image_type": "Chest X-Ray (PA)",
                "department": "ICU",
                "symptoms": "Orthopnea, bilateral lower extremity edema",
                "anomaly_score": 84.2,
                "urgency": "CRITICAL",
                "badge": "🔴",
                "status": "Review Now",
                "reviewed": False,
                "timestamp": "18:52:45",
                "image_path": os.path.join(self.assets_dir, "xr_cardiomegaly.png"),
                "expected_pathology": "Cardiomegaly"
            },
            {
                "scan_id": "XR-087",
                "patient_id": "ANON-4419",
                "image_type": "Chest X-Ray (PA)",
                "department": "Pulmonology",
                "symptoms": "Fever 38.8°C, productive cough, leukocytosis",
                "anomaly_score": 71.0,
                "urgency": "HIGH",
                "badge": "🟠",
                "status": "Priority Review",
                "reviewed": False,
                "timestamp": "18:35:10",
                "image_path": os.path.join(self.assets_dir, "xr_consolidation.png"),
                "expected_pathology": "Consolidation"
            },
            {
                "scan_id": "XR-121",
                "patient_id": "ANON-1550",
                "image_type": "Chest X-Ray (Lateral)",
                "department": "Internal Med",
                "symptoms": "Dullness to percussion left lung base",
                "anomaly_score": 56.3,
                "urgency": "REVIEW",
                "badge": "🟡",
                "status": "Routine Review",
                "reviewed": False,
                "timestamp": "18:10:02",
                "image_path": os.path.join(self.assets_dir, "xr_effusion.png"),
                "expected_pathology": "Pleural Effusion"
            },
            {
                "scan_id": "XR-099",
                "patient_id": "ANON-6701",
                "image_type": "Chest X-Ray (PA)",
                "department": "Outpatient",
                "symptoms": "Routine employment screening",
                "anomaly_score": 13.5,
                "urgency": "LOW",
                "badge": "🟢",
                "status": "Completed",
                "reviewed": True,
                "timestamp": "17:45:30",
                "image_path": os.path.join(self.assets_dir, "xr_normal.png"),
                "expected_pathology": "Normal"
            }
        ]

    def sort_queue(self, queue):
        """Sorts radiology queue by Urgency priority (CRITICAL > HIGH > REVIEW > LOW) and anomaly score."""
        urgency_rank = {"CRITICAL": 1, "HIGH": 2, "REVIEW": 3, "LOW": 4}
        sorted_queue = sorted(
            queue,
            key=lambda item: (
                1 if item.get("reviewed", False) else 0,  # Unreviewed cases first
                urgency_rank.get(item["urgency"], 5),
                -item["anomaly_score"]
            )
        )
        return sorted_queue

    def export_queue_df(self, queue):
        """Converts queue list into Pandas DataFrame for display or CSV export."""
        records = []
        for rank, item in enumerate(queue, start=1):
            records.append({
                "Rank": f"#{rank}",
                "Scan ID": item["scan_id"],
                "Patient ID": item["patient_id"],
                "Modality": item["image_type"],
                "Department": item.get("department", "Radiology"),
                "Anomaly Score": f"{item['anomaly_score']:.1f}%",
                "Urgency": f"{item['badge']} {item['urgency']}",
                "AI Status": item["status"],
                "Time": item["timestamp"],
                "Reviewed": "✓ Yes" if item.get("reviewed") else "Pending"
            })
        return pd.DataFrame(records)
