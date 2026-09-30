import os
import io
import time
import json
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

# Import Q-RADAR core modules
from q_radar.image_processor import MedicalImageProcessor
from q_radar.quantum_engine import QuantumFeatureEngine
from q_radar.urgency_engine import UrgencyEngine
from q_radar.dataset_manager import DatasetManager
from q_radar.limited_data_lab import LimitedDataLab

app = FastAPI(
    title="Q-RADAR REST API Backend",
    description="Quantum-Assisted Radiology Triage & Anomaly Routing API (QAIC UC-003)",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
dataset_mgr = DatasetManager()
image_proc = MedicalImageProcessor()
quantum_eng = QuantumFeatureEngine(num_qubits=4)
urgency_eng = UrgencyEngine()
lab_suite = LimitedDataLab()

# Global memory queue store
API_QUEUE = dataset_mgr.get_initial_radiology_queue()
API_QUEUE = dataset_mgr.sort_queue(API_QUEUE)

@app.get("/")
def root():
    return {
        "system": "Q-RADAR — Quantum-Assisted Radiology Triage REST API",
        "use_case": "QAIC UC-003 Medical Imaging",
        "status": "ONLINE",
        "version": "1.0.0",
        "disclaimer": "AI-assisted research prototype. Not for clinical diagnosis."
    }

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "HEALTHY",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "qiskit_backend": quantum_eng.backend_label,
        "qubit_count": quantum_eng.num_qubits,
        "active_queue_count": len(API_QUEUE)
    }

@app.get("/api/v1/queue")
def get_queue(urgency: str = None, search: str = None):
    global API_QUEUE
    queue = API_QUEUE.copy()
    
    if search:
        sq = search.lower()
        queue = [item for item in queue if sq in item["scan_id"].lower() or sq in item["patient_id"].lower()]
        
    if urgency and urgency != "All":
        queue = [item for item in queue if item["urgency"].upper() == urgency.upper()]
        
    sorted_q = dataset_mgr.sort_queue(queue)
    
    # Format light payload for REST response
    response_list = []
    for rank, item in enumerate(sorted_q, start=1):
        response_list.append({
            "rank": rank,
            "scan_id": item["scan_id"],
            "patient_id": item["patient_id"],
            "image_type": item["image_type"],
            "department": item.get("department", "Radiology"),
            "anomaly_score": item["anomaly_score"],
            "urgency": item["urgency"],
            "badge": item["badge"],
            "status": item["status"],
            "timestamp": item["timestamp"],
            "reasons": item.get("reasons", [])
        })
    return {"queue": response_list, "total_count": len(response_list)}

@app.post("/api/v1/analyze")
async def analyze_scan(
    file: UploadFile = File(None),
    patient_id: str = Form("ANON-2841"),
    image_type: str = Form("Chest X-Ray (AP)"),
    department: str = Form("Emergency Dept"),
    symptoms: str = Form("Acute shortness of breath"),
    preset_name: str = Form(None)
):
    global API_QUEUE
    
    img_bytes = None
    filename = "upload.png"
    
    if preset_name and preset_name in dataset_mgr.get_preset_samples():
        preset_info = dataset_mgr.get_preset_samples()[preset_name]
        filename = preset_info["image_path"]
        with open(filename, "rb") as f:
            img_bytes = f.read()
        patient_id = preset_info["patient_id"]
        image_type = preset_info["image_type"]
        department = preset_info["department"]
        symptoms = preset_info["symptoms"]
    elif file is not None:
        img_bytes = await file.read()
        filename = file.filename
    else:
        # Fallback to default preset pneumothorax
        filename = os.path.join("assets", "xr_pneumothorax.png")
        with open(filename, "rb") as f:
            img_bytes = f.read()

    # Process image
    img_gray, dicom_meta = image_proc.load_image_bytes(img_bytes, filename)
    enhanced_dict = image_proc.enhance_image(img_gray)
    features_16d = image_proc.extract_classical_features(enhanced_dict["clahe"])
    
    # Quantum Feature Engine Execution
    q_features = quantum_eng.reduce_features_to_qubits(features_16d, num_qubits=4)
    q_sim_results = quantum_eng.run_quantum_simulation(q_features, topology="Ring CNOT")
    
    q_anomaly = q_sim_results["raw_anomaly_score"]
    heatmap_data = image_proc.generate_explainability_heatmap(enhanced_dict["clahe"], anomaly_score_pct=q_anomaly)
    top_roi_score = heatmap_data["rois"][0]["roi_anomaly_score"] if len(heatmap_data["rois"]) > 0 else q_anomaly
    
    urgency_res = urgency_eng.calculate_urgency(
        quantum_anomaly_score=q_anomaly,
        roi_anomaly_score=top_roi_score,
        uncertainty_pct=q_sim_results["uncertainty_pct"],
        clinical_symptoms=symptoms
    )
    
    scan_id = f"XR-{np.random.randint(100, 999)}"
    
    record = {
        "scan_id": scan_id,
        "patient_id": patient_id,
        "image_type": image_type,
        "department": department,
        "symptoms": symptoms,
        "anomaly_score": urgency_res["anomaly_score_pct"],
        "urgency": urgency_res["urgency_level"],
        "badge": urgency_res["badge_color"],
        "status": urgency_res["action_label"],
        "reviewed": False,
        "timestamp": time.strftime("%H:%M:%S"),
        "dicom_meta": dicom_meta,
        "reasons": urgency_res["reasons"]
    }
    
    API_QUEUE.append(record)
    API_QUEUE = dataset_mgr.sort_queue(API_QUEUE)
    
    return {
        "status": "SUCCESS",
        "scan_id": scan_id,
        "patient_id": patient_id,
        "anomaly_score_pct": urgency_res["anomaly_score_pct"],
        "urgency_level": urgency_res["urgency_level"],
        "priority_rank": API_QUEUE.index(record) + 1,
        "recommendation": urgency_res["recommendation"],
        "reasons": urgency_res["reasons"],
        "quantum_metrics": {
            "fidelity": q_sim_results["fidelity"],
            "uncertainty_pct": q_sim_results["uncertainty_pct"],
            "z_expectations": q_sim_results["z_expectations"],
            "top_basis_states": q_sim_results["top_basis_states"],
            "circuit_ascii": q_sim_results["circuit_ascii"]
        },
        "rois": heatmap_data["rois"]
    }

@app.get("/api/v1/case/{scan_id}")
def get_case(scan_id: str):
    for item in API_QUEUE:
        if item["scan_id"].lower() == scan_id.lower():
            return {"case": item}
    raise HTTPException(status_code=404, detail="Scan ID not found in radiology queue")

@app.post("/api/v1/benchmark")
def run_benchmark(dataset_size: int = 50):
    df = lab_suite.get_comparison_table(dataset_size=dataset_size)
    return {
        "dataset_size": dataset_size,
        "benchmark_results": df.to_dict(orient="records")
    }

@app.get("/api/v1/export/csv")
def export_csv():
    df = dataset_mgr.export_queue_df(API_QUEUE)
    csv_str = df.to_csv(index=False)
    return Response(content=csv_str, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=q_radar_triage.csv"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
