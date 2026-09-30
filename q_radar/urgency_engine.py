import numpy as np

class UrgencyEngine:
    def __init__(self):
        pass

    def calculate_urgency(self, quantum_anomaly_score, roi_anomaly_score, uncertainty_pct, clinical_symptoms="None"):
        """
        Calculates multi-factorial Urgency Score and assigns risk tier (CRITICAL, HIGH, REVIEW, LOW).
        """
        # Weights for multi-factor formula
        w_quantum = 0.50
        w_roi = 0.35
        w_uncertainty = 0.15
        
        # Clinical symptom modifier (+5% to +15% if acute symptoms reported)
        symptom_bonus = 0.0
        sym_lower = clinical_symptoms.lower()
        if any(term in sym_lower for term in ["dyspnea", "shortness of breath", "acute pain", "chest pain", "hypoxia", "trauma"]):
            symptom_bonus = 10.0
        elif any(term in sym_lower for term in ["fever", "cough", "hemoptysis"]):
            symptom_bonus = 5.0

        # Multi-factor raw urgency score
        raw_score = (w_quantum * quantum_anomaly_score + 
                     w_roi * roi_anomaly_score + 
                     w_uncertainty * (100.0 - uncertainty_pct) + 
                     symptom_bonus)
                     
        final_anomaly_score = float(np.clip(raw_score, 5.0, 99.0))
        
        # Determine Urgency Level
        if final_anomaly_score >= 85.0:
            urgency_level = "CRITICAL"
            badge_color = "🔴"
            recommendation = "PRIORITISE FOR IMMEDIATE RADIOLOGIST REVIEW"
            action_label = "Review Now"
            time_estimate = "< 15 Mins"
        elif final_anomaly_score >= 70.0:
            urgency_level = "HIGH"
            badge_color = "🟠"
            recommendation = "PRIORITISE FOR SHIFT REVIEW"
            action_label = "Priority Review"
            time_estimate = "< 1 Hour"
        elif final_anomaly_score >= 40.0:
            urgency_level = "REVIEW"
            badge_color = "🟡"
            recommendation = "ROUTINE RADIOLOGY REVIEW"
            action_label = "Routine Review"
            time_estimate = "< 4 Hours"
        else:
            urgency_level = "LOW"
            badge_color = "🟢"
            recommendation = "STANDARD ARCHIVE / ROUTINE QUEUE"
            action_label = "Routine Review"
            time_estimate = "24 Hours"

        # Generate Explainable Prioritisation Reasons
        reasons = self._generate_prioritisation_reasons(
            final_anomaly_score, quantum_anomaly_score, roi_anomaly_score, uncertainty_pct, clinical_symptoms
        )

        return {
            "anomaly_score_pct": round(final_anomaly_score, 1),
            "urgency_level": urgency_level,
            "badge_color": badge_color,
            "recommendation": recommendation,
            "action_label": action_label,
            "time_estimate": time_estimate,
            "reasons": reasons,
            "clinical_disclaimer": "AI-assisted triage priority calculation. Does not replace radiologist clinical diagnosis.",
            "wording_compliant_statement": "AI detected an image pattern requiring priority review."
        }

    def _generate_prioritisation_reasons(self, final_score, q_score, roi_score, uncertainty, symptoms):
        reasons = []
        
        if q_score > 75.0:
            reasons.append(f"Elevated quantum kernel Hilbert space deviation (Quantum Anomaly Score: {q_score:.1f}%)")
        else:
            reasons.append(f"Quantum feature map pattern matches learned baseline bounds ({q_score:.1f}%)")
            
        if roi_score > 70.0:
            reasons.append(f"High spatial intensity deviation detected in localized lung ROI (ROI Score: {roi_score:.1f}%)")
        elif roi_score > 45.0:
            reasons.append(f"Moderate spatial residual pattern variance in chest parenchyma ({roi_score:.1f}%)")
            
        if uncertainty < 30.0:
            reasons.append(f"High model statevector confidence (Uncertainty: {uncertainty:.1f}%)")
        else:
            reasons.append(f"Moderate quantum state variance noted (Uncertainty: {uncertainty:.1f}%)")
            
        if symptoms != "None" and len(symptoms.strip()) > 0:
            reasons.append(f"Acute clinical symptoms noted in patient metadata: '{symptoms}'")
            
        reasons.append("Multi-factor AI triage index places case in top-tier radiology review queue")
        return reasons
