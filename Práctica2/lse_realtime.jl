# ==============================================================================
# LSE REAL-TIME - VERSIÓN FINAL ESTABLE (MAC M1)
# ==============================================================================
using PyCall, DecisionTree, DataFrames, CSV, Printf

# 1. entrenar el modelo
println(">> 1. Entrenando modelo...")
df = CSV.read("dataset/landmarks_lse.csv", DataFrame)
inputs = Matrix(df[:, 1:63])
targets = Vector(df[:, :letra])
model = DecisionTree.DecisionTreeClassifier(max_depth=12)
DecisionTree.fit!(model, inputs, targets)

# 2. configurar MediaPipe
println(">> 2. Iniciando MediaPipe...")
cv2 = pyimport("cv2")
mp = pyimport("mediapipe")
tasks_python = pyimport("mediapipe.tasks.python")
vision = pyimport("mediapipe.tasks.python.vision")

# helper en Python puro para extraer coordenadas de forma segura
py"""
def extract_first_hand(results):
    if not results.hand_landmarks:
        return []
    # results.hand_landmarks[0] es la primera mano encontrada
    return [(lm.x, lm.y, lm.z) for lm in results.hand_landmarks[0]]
"""

# definir opciones
base_options = tasks_python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# conexiones LSE (índices 0-based para Python landmarks)
CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20)
]

# 3. abrir Cámara
println(">> 3. Abriendo cámara...")
cap = cv2.VideoCapture(0)

try
    while true
        ret, frame = pycall(cap.read, Tuple{Bool, PyObject})
        if !ret break end
        h = frame.shape[1] 
        w = frame.shape[2]
        
        rgb_frame = pycall(cv2.cvtColor, PyObject, frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # DETECTAR
        results = detector.detect(mp_image)
        # extraer usando Python puro
        landmarks = py"extract_first_hand"(results)
        
        if length(landmarks) == 21
            puntos_vector = Float64[]
            
            # toma la muñeca (nodo 0 en Python, nodo 1 en Julia) como punto base
            base_x, base_y, base_z = landmarks[1][1], landmarks[1][2], landmarks[1][3]
            
            # factor de escala (Muñeca al nodo 9 es el índice 10 en Julia)
            scale = sqrt((landmarks[10][1] - base_x)^2 + (landmarks[10][2] - base_y)^2 + (landmarks[10][3] - base_z)^2)
            if scale == 0 scale = 1e-6 end

            # 1. extraer y dibujar puntos (nodos)
            for lm in landmarks
                try
                    x, y, z = lm[1], lm[2], lm[3]
                    # DISTANCIA RELATIVA y NORMALIZADA POR ESCALA (Invarianza a posición y tamaño)
                    push!(puntos_vector, Float64((x - base_x)/scale), Float64((y - base_y)/scale), Float64((z - base_z)/scale))
                    
                    px, py = Int(round(x * w)), Int(round(y * h))
                    pycall(cv2.circle, PyObject, frame, (px, py), 5, (0, 0, 255), -1)
                catch e end
            end

            # 2. dibujar esqueleto (aristas)
            for (idx1, idx2) in CONNECTIONS
                try
                    # en julia landmarks ya está indizado en 1
                    p1 = landmarks[idx1+1] 
                    p2 = landmarks[idx2+1]
                    pycall(cv2.line, PyObject, frame, 
                             (Int(round(p1[1] * w)), Int(round(p1[2] * h))), 
                             (Int(round(p2[1] * w)), Int(round(p2[2] * h))), 
                             (0, 255, 0), 2)
                catch e end
            end

            # 3. predicción
            if length(puntos_vector) == 63
                entrada = reshape(puntos_vector, 1, :)
                prediccion = DecisionTree.predict(model, entrada)[1]
                
                # interfaz visual
                pycall(cv2.rectangle, PyObject, frame, (10, 10), (450, 100), (0, 0, 0), -1)
                pycall(cv2.putText, PyObject, frame, "LETRA: $prediccion", (30, 80), 
                            cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 0), 3)
            end
        else
            pycall(cv2.putText, PyObject, frame, "BUSCANDO MANO...", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        end
        
        cv2.imshow("RECONOCIMIENTO LSE - FIC", frame)
        if cv2.waitKey(1) & 0xFF == 27 break end
    end
finally
    cap.release()
    cv2.destroyAllWindows()
    try detector.close() catch e end
    println(">> Programa finalizado.")
end
