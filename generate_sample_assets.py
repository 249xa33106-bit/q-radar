import os
import numpy as np
import cv2

def generate_chest_xray(pathology="normal", width=512, height=512, seed=42):
    np.random.seed(seed)
    
    # Background thorax grid (dark radiolucent outside, grey soft tissue)
    img = np.zeros((height, width), dtype=float)
    
    # Thorax contour (ellipse)
    center = (width // 2, height // 2)
    cv2.ellipse(img, center, (int(width * 0.42), int(height * 0.46)), 0, 0, 360, 0.25, -1)
    
    # Lungs (dark radiolucent areas - left & right lung fields)
    lung_y = int(height * 0.45)
    lung_height = int(height * 0.35)
    lung_width = int(width * 0.16)
    
    left_lung_x = int(width * 0.30)
    right_lung_x = int(width * 0.70)
    
    # Draw dark lungs
    cv2.ellipse(img, (left_lung_x, lung_y), (lung_width, lung_height), -8, 0, 360, 0.05, -1)
    cv2.ellipse(img, (right_lung_x, lung_y), (lung_width, lung_height), 8, 0, 360, 0.05, -1)
    
    # Mediastinum & Cardiac silhouette (bright radiopaque center & heart left side)
    cv2.ellipse(img, (width // 2, int(height * 0.45)), (int(width * 0.07), int(height * 0.38)), 0, 0, 360, 0.65, -1)
    
    # Cardiac shadow
    cardiac_r = int(width * (0.22 if pathology == "cardiomegaly" else 0.14))
    cv2.circle(img, (int(width * 0.43), int(height * 0.58)), cardiac_r, 0.75, -1)
    
    # Ribs (radiopaque horizontal arches)
    for i in range(7):
        ry = int(height * (0.20 + i * 0.08))
        cv2.ellipse(img, (width // 2, ry), (int(width * 0.38), int(height * 0.04)), 0, 180, 360, 0.55, 3)
        cv2.ellipse(img, (width // 2, ry + 15), (int(width * 0.38), int(height * 0.04)), 0, 0, 180, 0.45, 2)
        
    # Clavicles (bright horizontal bands at top)
    cv2.line(img, (int(width * 0.15), int(height * 0.18)), (int(width * 0.45), int(height * 0.22)), 0.85, 6)
    cv2.line(img, (int(width * 0.85), int(height * 0.18)), (int(width * 0.55), int(height * 0.22)), 0.85, 6)
    
    # Spine (vertical central line of vertebrae)
    cv2.line(img, (width // 2, int(height * 0.10)), (width // 2, int(height * 0.90)), 0.70, 8)
    for i in range(12):
        sy = int(height * (0.12 + i * 0.065))
        cv2.rectangle(img, (width // 2 - 8, sy), (width // 2 + 8, sy + 10), 0.80, -1)
        
    # Diaphragm domes
    cv2.ellipse(img, (left_lung_x, int(height * 0.82)), (int(width * 0.18), int(height * 0.08)), 0, 180, 360, 0.70, -1)
    cv2.ellipse(img, (right_lung_x, int(height * 0.84)), (int(width * 0.18), int(height * 0.08)), 0, 180, 360, 0.70, -1)

    # Vascular markings (bronchovascular tree noise)
    noise = np.random.normal(0, 0.03, (height, width))
    img = np.clip(img + noise, 0, 1)
    
    # Add pathology features
    if pathology == "pneumothorax":
        # Right apex hyperlucency & collapsed lung visceral pleural line
        cv2.ellipse(img, (int(width * 0.72), int(height * 0.28)), (int(width * 0.12), int(height * 0.14)), 5, 0, 360, 0.01, -1)
        cv2.ellipse(img, (int(width * 0.72), int(height * 0.28)), (int(width * 0.12), int(height * 0.14)), 5, 0, 360, 0.95, 2)
    elif pathology == "consolidation":
        # Dense cloud-like opacity in right lower lobe
        cv2.circle(img, (int(width * 0.70), int(height * 0.62)), int(width * 0.11), 0.78, -1)
        # Blur the cloud to simulate airspace consolidation
        patch = cv2.GaussianBlur(img[int(height*0.50):int(height*0.74), int(width*0.58):int(width*0.82)], (31, 31), 0)
        img[int(height*0.50):int(height*0.74), int(width*0.58):int(width*0.82)] = patch
    elif pathology == "effusion":
        # Blunting of left costophrenic angle / fluid level
        cv2.rectangle(img, (int(width * 0.12), int(height * 0.68)), (int(width * 0.38), int(height * 0.84)), 0.75, -1)
    elif pathology == "nodule":
        # Solitary circumscribed focal nodule
        cv2.circle(img, (int(width * 0.32), int(height * 0.40)), 16, 0.88, -1)

    # Smooth & scale to 8-bit image
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img_8u = (img * 255).astype(np.uint8)
    return img_8u

def create_preset_assets():
    os.makedirs("assets", exist_ok=True)
    samples = {
        "pneumothorax": "assets/xr_pneumothorax.png",
        "cardiomegaly": "assets/xr_cardiomegaly.png",
        "consolidation": "assets/xr_consolidation.png",
        "effusion": "assets/xr_effusion.png",
        "normal": "assets/xr_normal.png"
    }
    
    for name, path in samples.items():
        img = generate_chest_xray(pathology=name, seed=hash(name) % 1000)
        cv2.imwrite(path, img)
        print(f"Generated sample asset: {path}")

if __name__ == "__main__":
    create_preset_assets()
