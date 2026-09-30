import numpy as np
import matplotlib.pyplot as plt
import io

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import StatevectorSampler

class QuantumFeatureEngine:
    def __init__(self, num_qubits=4, shots=1024):
        self.num_qubits = num_qubits
        self.shots = shots
        self.backend_label = "Qiskit Aer Quantum Simulator (qasm_simulator / Statevector)"
        self.update_reference_state()

    def update_reference_state(self):
        """Updates normalized reference quantum state vector for normal radiology baseline."""
        ref_vec = np.zeros(2**self.num_qubits)
        ref_vec[0] = 0.85
        ref_vec[1] = 0.35
        if self.num_qubits > 2:
            ref_vec[2] = 0.20
        ref_vec = ref_vec / np.linalg.norm(ref_vec)
        self.reference_quantum_state = ref_vec

    def reduce_features_to_qubits(self, features_16d, num_qubits=None):
        """
        Reduces 16D classical feature vector down to N quantum features (0.0 to 1.0 range).
        """
        n_q = num_qubits if num_qubits is not None else self.num_qubits
        np.random.seed(42)
        proj_matrix = np.random.randn(16, n_q)
        proj_matrix, _ = np.linalg.qr(proj_matrix)
        
        q_features = np.dot(features_16d, proj_matrix)
        q_min, q_max = np.min(q_features), np.max(q_features)
        if q_max > q_min:
            q_features = (q_features - q_min) / (q_max - q_min)
        else:
            q_features = np.zeros(n_q)
            
        return q_features

    def build_quantum_circuit(self, q_features, topology="Ring CNOT"):
        """
        Constructs a Qiskit parameterized quantum feature map circuit with selectable entanglement topologies.
        """
        n_q = len(q_features)
        qr = QuantumRegister(n_q, name="q")
        cr = ClassicalRegister(n_q, name="c")
        qc = QuantumCircuit(qr, cr)
        
        # Layer 1: Single-qubit Ry rotation encoding
        for i in range(n_q):
            val = q_features[i]
            theta = float(np.pi * val)
            qc.ry(theta, qr[i])
            
        qc.barrier()
        
        # Layer 2: Entanglement topologies
        if topology == "Linear CNOT":
            for i in range(n_q - 1):
                qc.cx(qr[i], qr[i + 1])
        elif topology == "All-to-All Entanglement":
            for i in range(n_q):
                for j in range(i + 1, n_q):
                    qc.cx(qr[i], qr[j])
        elif topology == "Pauli Z-Z Phase Kernel":
            for i in range(n_q):
                next_q = (i + 1) % n_q
                qc.cx(qr[i], qr[next_q])
                phase = float(np.pi * (q_features[i] * q_features[next_q]))
                qc.rz(phase, qr[next_q])
                qc.cx(qr[i], qr[next_q])
        else:  # Default: Ring CNOT Topology
            for i in range(n_q):
                next_q = (i + 1) % n_q
                qc.cx(qr[i], qr[next_q])
            
        qc.barrier()
        
        # Layer 3: Phase Rz non-linear encoding
        for i in range(n_q):
            phi = float(np.pi * (q_features[i] ** 2))
            qc.rz(phi, qr[i])
            
        # Measurement layer
        qc.measure(qr, cr)
        
        return qc

    def run_quantum_simulation(self, q_features, topology="Ring CNOT"):
        """
        Executes quantum circuit on Qiskit Aer simulator, computes statevector probabilities,
        quantum kernel distance, anomaly score, and uncertainty metric.
        """
        n_q = len(q_features)
        if n_q != self.num_qubits:
            self.num_qubits = n_q
            self.update_reference_state()

        qc = self.build_quantum_circuit(q_features, topology=topology)
        
        # Execute simulation using Qiskit Statevector / Primitives
        sampler = StatevectorSampler()
        job = sampler.run([qc], shots=self.shots)
        result = job.result()
        
        pub_result = result[0]
        counts_dict = pub_result.data.c.get_counts()
        
        # Compute measurement probabilities for all 2^num_qubits basis states
        total_counts = sum(counts_dict.values())
        prob_vec = np.zeros(2**n_q)
        
        for bitstr, count in counts_dict.items():
            idx = int(bitstr, 2) if isinstance(bitstr, str) else int(bitstr)
            if idx < len(prob_vec):
                prob_vec[idx] = count / total_counts
                
        # Approximate state vector amplitude
        state_amplitude = np.sqrt(prob_vec)
        
        # Compute quantum kernel fidelity against reference normal state vector
        ref_v = self.reference_quantum_state[:len(state_amplitude)]
        ref_v = ref_v / np.linalg.norm(ref_v)
        fidelity = float(np.dot(state_amplitude, ref_v) ** 2)
        fidelity = np.clip(fidelity, 0.0, 1.0)
        
        # Quantum anomaly score (1 - fidelity) with scaling factor
        raw_anomaly = (1.0 - fidelity) * 100.0
        
        # Quantum noise / entropy uncertainty metric
        non_zero_probs = prob_vec[prob_vec > 1e-6]
        quantum_entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs)) if len(non_zero_probs) > 0 else 0.0
        max_entropy = np.log2(2**n_q)
        uncertainty_pct = float((quantum_entropy / max_entropy) * 100.0)
        
        # Compute single-qubit expectation values <Z_i>
        z_expectations = []
        for i in range(n_q):
            p0 = sum(prob_vec[idx] for idx in range(2**n_q) if not (idx & (1 << (n_q - 1 - i))))
            z_exp = 2.0 * p0 - 1.0
            z_expectations.append(round(z_exp, 3))

        # Format Top basis states histogram data
        top_basis_states = sorted(
            [{"state": f"|{bin(i)[2:].zfill(n_q)}⟩", "prob": float(p)} for i, p in enumerate(prob_vec)],
            key=lambda item: item["prob"],
            reverse=True
        )[:4]

        # Circuit ASCII representation string
        circuit_ascii = str(qc.draw(output="text"))

        return {
            "num_qubits": n_q,
            "topology": topology,
            "q_features": [round(float(f), 4) for f in q_features],
            "quantum_circuit": qc,
            "circuit_ascii": circuit_ascii,
            "backend": self.backend_label,
            "counts": counts_dict,
            "fidelity": round(fidelity, 4),
            "raw_anomaly_score": round(raw_anomaly, 1),
            "uncertainty_pct": round(uncertainty_pct, 1),
            "z_expectations": z_expectations,
            "top_basis_states": top_basis_states
        }
