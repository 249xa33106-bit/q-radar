import subprocess
import sys
import time
import os
import urllib.request
import json
import threading

def run_backend():
    print("🚀 Starting Q-RADAR FastAPI REST Backend on http://localhost:8000 ...")
    cmd = [sys.executable, "-m", "uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "8000"]
    subprocess.run(cmd)

def run_frontend():
    print("🚀 Starting Q-RADAR Streamlit Frontend UI on http://localhost:8501 ...")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
    subprocess.run(cmd)

def monitor_health():
    time.sleep(4)
    print("\n" + "="*65)
    print("⚛️  Q-RADAR PRODUCTION DEPLOYMENT HEALTH CHECK")
    print("="*65)
    
    # Check Backend
    try:
        resp = urllib.request.urlopen("http://localhost:8000/api/v1/health", timeout=2)
        if resp.status == 200:
            data = json.loads(resp.read().decode())
            print(f"✅ FastAPI REST Backend: ONLINE (http://localhost:8000)")
            print(f"   • Qiskit Backend: {data.get('qiskit_backend')}")
            print(f"   • Qubits: {data.get('qubit_count')}")
        else:
            print("⚠️ FastAPI REST Backend: Starting...")
    except Exception as e:
        print(f"⚠️ FastAPI REST Backend check note: {e}")

    # Check Frontend
    try:
        resp_f = urllib.request.urlopen("http://localhost:8501", timeout=2)
        if resp_f.status == 200:
            print("✅ Streamlit Web Frontend: ONLINE (http://localhost:8501)")
        else:
            print("⚠️ Streamlit Web Frontend: Starting...")
    except Exception as e:
        print(f"⚠️ Streamlit Web Frontend check note: {e}")

    print("="*65)
    print("🌐 ACCESS LINKS:")
    print("   • Web Application UI:  http://localhost:8501")
    print("   • REST API Swagger:    http://localhost:8000/docs")
    print("   • REST API Health:     http://localhost:8000/api/v1/health")
    print("="*65 + "\n")

if __name__ == "__main__":
    t_backend = threading.Thread(target=run_backend, daemon=True)
    t_frontend = threading.Thread(target=run_frontend, daemon=True)
    
    t_backend.start()
    t_frontend.start()
    
    monitor_health()
    
    # Keep main process alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Q-RADAR production servers.")
