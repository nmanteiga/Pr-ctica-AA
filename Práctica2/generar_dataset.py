import cv2
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv

MODEL_PATH = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)
DATASET_PATH = "dataset"
OUTPUT_CSV = "dataset/landmarks_lse.csv"

clases = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = [f"{axis}{i}" for i in range(21) for axis in ['x','y','z']] + ['letra']
    writer.writerow(header)

    for letra in clases:
        ruta_letra = os.path.join(DATASET_PATH, letra)
        print(f"Procesando letra {letra}...")
        for img_name in os.listdir(ruta_letra):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            image = cv2.imread(os.path.join(ruta_letra, img_name))
            if image is None: continue
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            results = detector.detect(mp_image)
            
            if results.hand_landmarks:
                landmarks = results.hand_landmarks[0]
                row = []
                # coge la muñeca (landmark 0) como origen de coordenadas
                base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
                
                # factor de escala: distancia de la muñeca al nudillo del dedo medio (punto 9)
                scale = ((landmarks[9].x - base_x)**2 + (landmarks[9].y - base_y)**2 + (landmarks[9].z - base_z)**2)**0.5
                if scale == 0: scale = 1e-6
                
                for lm in landmarks:
                    # guarda la DISTANCIA RELATIVA y ESCALADA
                    row.extend([(lm.x - base_x)/scale, (lm.y - base_y)/scale, (lm.z - base_z)/scale])
                row.append(letra)
                writer.writerow(row)

print(f"CSV guardado en {OUTPUT_CSV}")