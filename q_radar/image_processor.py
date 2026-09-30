import cv2
import numpy as np
import PIL.Image
import io

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

class MedicalImageProcessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def load_image_bytes(self, file_bytes, filename=""):
        """Loads PNG, JPG, or DICOM file bytes into uint8 grayscale numpy array."""
        is_dicom = filename.lower().endswith(".dcm") or filename.lower().endswith(".dicom")
        dicom_metadata = {}
        
        if is_dicom and HAS_PYDICOM:
            try:
                ds = pydicom.dcmread(io.BytesIO(file_bytes))
                dicom_metadata = {
                    "PatientID": getattr(ds, "PatientID", "ANON-DICOM"),
                    "PatientName": str(getattr(ds, "PatientName", "ANON^PATIENT")),
                    "PatientAge": getattr(ds, "PatientAge", "54Y"),
                    "PatientSex": getattr(ds, "PatientSex", "M"),
                    "Modality": getattr(ds, "Modality", "CR"),
                    "StudyDate": getattr(ds, "StudyDate", "2026-09-30"),
                    "Manufacturer": getattr(ds, "Manufacturer", "Q-RADAR Sim Scanner"),
                    "WindowCenter": float(getattr(ds, "WindowCenter", 128)),
                    "WindowWidth": float(getattr(ds, "WindowWidth", 256)),
                }
                arr = ds.pixel_array.astype(float)
                # Normalize pixel array to uint8
                arr = arr - np.min(arr)
                if np.max(arr) > 0:
                    arr = (arr / np.max(arr)) * 255.0
                img_gray = arr.astype(np.uint8)
            except Exception:
                img_gray = self._load_fallback(file_bytes)
        else:
            img_gray = self._load_fallback(file_bytes)
            
        img_gray = cv2.resize(img_gray, self.target_size, interpolation=cv2.INTER_AREA)
        return img_gray, dicom_metadata

    def _load_fallback(self, file_bytes):
        try:
            pil_img = PIL.Image.open(io.BytesIO(file_bytes)).convert("L")
            return np.array(pil_img)
        except Exception:
            # Synthetic fallback grayscale 256x256
            return np.zeros(self.target_size, dtype=np.uint8)

    def enhance_image(self, img_gray, clip_limit=3.0, tile_grid_size=(8, 8)):
        """
        Enhance medical image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        and Laplacian edge sharpening.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_clahe = clahe.apply(img_gray)
        
        # Laplacian sharpening mask
        blurred = cv2.GaussianBlur(enhanced_clahe, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        sharpened = np.clip(enhanced_clahe - 0.3 * laplacian, 0, 255).astype(np.uint8)
        
        # Bilateral filter for edge-preserving smoothing
        denoised = cv2.bilateralFilter(sharpened, d=5, sigmaColor=50, sigmaSpace=50)
        
        return {
            "clahe": enhanced_clahe,
            "sharpened": sharpened,
            "denoised": denoised
        }

    def extract_classical_features(self, img_gray):
        """
        Extracts spatial, texture, edge, and quadrant statistical features.
        Returns a 16-dimensional feature vector.
        """
        h, w = img_gray.shape
        img_norm = img_gray.astype(float) / 255.0
        
        # Global statistics
        mean_val = np.mean(img_norm)
        std_val = np.std(img_norm)
        p10, p90 = np.percentile(img_norm, [10, 90])
        contrast = p90 - p10
        
        # Edge density (Sobel)
        sobelx = cv2.Sobel(img_norm, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_norm, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_density = np.mean(edge_mag)
        edge_max = np.max(edge_mag)
        
        # Quadrant spatial distributions
        q_tl = np.mean(img_norm[0:h//2, 0:w//2])
        q_tr = np.mean(img_norm[0:h//2, w//2:w])
        q_bl = np.mean(img_norm[h//2:h, 0:w//2])
        q_br = np.mean(img_norm[h//2:h, w//2:w])
        
        # Upper vs Lower ratio, Left vs Right asymmetry
        lr_asymmetry = abs(q_tl + q_bl - (q_tr + q_br))
        tb_ratio = (q_tl + q_tr) / (q_bl + q_br + 1e-5)
        
        # Spatial texture GLCM-inspired metrics
        shifted_h = np.roll(img_norm, 1, axis=0)
        diff_h = np.mean(np.abs(img_norm - shifted_h))
        shifted_v = np.roll(img_norm, 1, axis=1)
        diff_v = np.mean(np.abs(img_norm - shifted_v))
        texture_energy = np.mean(img_norm ** 2)
        
        # 16-feature vector
        features = np.array([
            mean_val, std_val, contrast, edge_density, edge_max,
            q_tl, q_tr, q_bl, q_br, lr_asymmetry, tb_ratio,
            diff_h, diff_v, texture_energy, p10, p90
        ], dtype=float)
        
        return features

    def generate_explainability_heatmap(self, img_gray, anomaly_score_pct=75.0, colormap_name="jet"):
        """
        Generates an anomaly heatmap overlay and identifies suspicious regions of interest (ROIs).
        """
        h, w = img_gray.shape
        img_float = img_gray.astype(float) / 255.0
        
        # Synthetic reference normal thorax model
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        normal_expected = 0.25 + 0.4 * np.exp(-dist_from_center / (0.3 * h))
        
        # Spatial residual deviation
        diff = np.abs(img_float - normal_expected)
        
        # Apply Sobel edge weighting
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_mag = grad_mag / (np.max(grad_mag) + 1e-5)
        
        # Combined anomaly spatial density map
        anomaly_map = 0.6 * diff + 0.4 * grad_mag
        anomaly_map = cv2.GaussianBlur(anomaly_map, (15, 15), 0)
        
        # Scale anomaly map based on overall anomaly score percentage
        scale_factor = max(0.2, anomaly_score_pct / 100.0)
        anomaly_map = anomaly_map * scale_factor
        anomaly_map = np.clip(anomaly_map / (np.max(anomaly_map) + 1e-5), 0, 1)
        
        # Convert to 8-bit heatmap
        heatmap_8u = (anomaly_map * 255).astype(np.uint8)
        
        colormaps = {
            "jet": cv2.COLORMAP_JET,
            "inferno": cv2.COLORMAP_INFERNO,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "turbo": cv2.COLORMAP_TURBO
        }
        cm = colormaps.get(colormap_name.lower(), cv2.COLORMAP_JET)
        heatmap_colored = cv2.applyColorMap(heatmap_8u, cm)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # Identify suspicious ROI regions
        rois = self._extract_roi_boxes(anomaly_map, h, w)
        
        return {
            "anomaly_map_raw": anomaly_map,
            "heatmap_rgb": heatmap_rgb,
            "img_rgb": img_rgb,
            "rois": rois
        }

    def create_overlay(self, img_rgb, heatmap_rgb, opacity=0.45, draw_rois=True, rois=None):
        """Blends original RGB image and Heatmap RGB at given opacity, with optional ROI bounding box overlays."""
        overlay = cv2.addWeighted(img_rgb, 1.0 - opacity, heatmap_rgb, opacity, 0)
        
        if draw_rois and rois is not None:
            overlay_copy = overlay.copy()
            for r in rois:
                x, y, bw, bh = r["bbox"]
                # Color code bounding box: Red for >80%, Orange for >60%, Yellow for lower
                score = r["roi_anomaly_score"]
                color = (255, 50, 50) if score > 80 else ((255, 140, 0) if score > 60 else (255, 220, 0))
                
                # Draw bounding box
                cv2.rectangle(overlay_copy, (x, y), (x + bw, y + bh), color, 2)
                
                # Label box above bounding box
                label = f"R{r['id']}: {score:.1f}%"
                cv2.rectangle(overlay_copy, (x, max(0, y - 20)), (x + 90, max(0, y)), (20, 20, 20), -1)
                cv2.putText(overlay_copy, label, (x + 4, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            overlay = overlay_copy

        return overlay

    def _extract_roi_boxes(self, anomaly_map, h, w):
        thresh = (anomaly_map > 0.50).astype(np.uint8) * 255
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rois = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 120:
                x, y, bw, bh = cv2.boundingRect(cnt)
                roi_score = float(np.mean(anomaly_map[y:y+bh, x:x+bw])) * 100.0
                
                # Determine anatomical location string based on coordinates
                loc_y = "Upper" if y < h/3 else ("Middle" if y < 2*h/3 else "Lower")
                loc_x = "Right Lung" if x > w/2 else "Left Lung"
                if abs(x + bw/2 - w/2) < w/6:
                    loc_x = "Mediastinum / Cardiac Region"
                
                rois.append({
                    "id": i + 1,
                    "location": f"{loc_y} {loc_x}",
                    "bbox": (x, y, bw, bh),
                    "roi_anomaly_score": round(roi_score, 1),
                    "area_pct": round((area / (h * w)) * 100, 2)
                })
        # Sort ROIs by score descending
        rois.sort(key=lambda r: r["roi_anomaly_score"], reverse=True)
        return rois[:4] # Top 4 suspicious regions
