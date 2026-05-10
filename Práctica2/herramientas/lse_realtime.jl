# ==============================================================================
# LSE REAL-TIME 
# ==============================================================================

# GESTIÓN DE DEPENDENCIAS Y PAQUETES ---
println(">> 1. Cargando paquetes y dependencias...")
using Pkg

# función para asegurar que un paquete de Julia esté instalado
function ensure_julia_package(pkg_name::String)
    try
        @eval using $(Symbol(pkg_name))
    catch
        println("Paquete de Julia '$pkg_name' no encontrado. Instalando...")
        Pkg.add(pkg_name)
        @eval using $(Symbol(pkg_name))
    end
end

# asegurar paquetes de Julia
ensure_julia_package("PyCall")
ensure_julia_package("Conda")
ensure_julia_package("DecisionTree")
ensure_julia_package("DataFrames")
ensure_julia_package("CSV")
ensure_julia_package("Printf")

using PyCall, Conda, DecisionTree, DataFrames, CSV, Printf

# instalar dependencias de Python si no existen
try
    pyimport("cv2")
catch e
    println("Paquete 'opencv' de Python no encontrado. Instalando con Conda...")
    Conda.add("opencv")
end

try
    pyimport("mediapipe")
catch e
    println("Paquete 'mediapipe' de Python no encontrado. Instalando con Conda...")
    Conda.add("mediapipe")
end


# ENTRENAMIENTO DEL MODELO ---
println(">> 2. Entrenando modelo...")
# Usar @__DIR__ para construir una ruta robusta al archivo del dataset
script_dir = @__DIR__
dataset_path = joinpath(script_dir, "..", "dataset", "landmarks_lse.csv")
model_path = joinpath(script_dir, "hand_landmarker.task")

if !isfile(dataset_path)
    println("Error: No se encuentra el archivo 'dataset/landmarks_lse.csv'.")
    println("Asegúrate de que la estructura de carpetas es correcta: 'herramientas/' y 'dataset/' deben estar al mismo nivel.")
    exit()
end
if !isfile(model_path)
    println("Error: No se encuentra el archivo 'hand_landmarker.task'.")
    println("Asegúrate de que está en el directorio principal del proyecto.")
    exit()
end

df = CSV.read(dataset_path, DataFrame)
inputs = Matrix(df[:, 1:63])
targets = Vector(df[:, :letra])
model = DecisionTree.DecisionTreeClassifier(max_depth=12)
DecisionTree.fit!(model, inputs, targets)

# CONFIGURACIÓN DE MEDIAPIPE ---
println(">> 3. Iniciando MediaPipe...")
cv2 = pyimport("cv2")
mp = pyimport("mediapipe")
tasks_python = pyimport("mediapipe.tasks.python")
vision = pyimport("mediapipe.tasks.python.vision")

# Helper en Python puro para extraer coordenadas de forma segura
py"""
def extract_first_hand(results):
    if not results.hand_landmarks:
        return []
    # results.hand_landmarks[0] es la primera mano encontrada
    return [(lm.x, lm.y, lm.z) for lm in results.hand_landmarks[0]]
"""

# def opciones usando la ruta robusta
base_options = tasks_python.BaseOptions(model_asset_path=model_path)
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

# PROCESAMIENTO DE VÍDEO EN TIEMPO REAL ---
println(">> 4. Abriendo cámara... (Presiona 'ESC' para salir)")
cap = cv2.VideoCapture(0)

if !pycall(cap.isOpened, Bool)
    println("Error: No se pudo abrir la cámara.")
    exit()
end

try
    while true
        ret, frame = pycall(cap.read, Tuple{Bool, PyObject})
        if !ret break end
        
        # obtener dimensiones del frame de forma segura
        h, w, _ = frame.shape
        
        rgb_frame = pycall(cv2.cvtColor, PyObject, frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # DETECTAR
        results = detector.detect(mp_image)
        # extraer usando Python puro
        landmarks = py"extract_first_hand"(results)
        
        if length(landmarks) == 21
            puntos_vector = Float64[]
            
            # toma la muñeca (nodo 0 en Python) como punto base
            base_x, base_y, base_z = landmarks[1]
            
            # factor de escala (Muñeca al nodo 9)
            scale = sqrt((landmarks[10][1] - base_x)^2 + (landmarks[10][2] - base_y)^2 + (landmarks[10][3] - base_z)^2)
            if scale < 1e-6 scale = 1.0 end # Evitar división por cero

            # 1. extraer y dibujar puntos (nodos)
            for lm in landmarks
                x, y, z = lm
                push!(puntos_vector, (x - base_x)/scale, (y - base_y)/scale, (z - base_z)/scale)
                
                px, py = Int(round(x * w)), Int(round(y * h))
                pycall(cv2.circle, PyObject, frame, (px, py), 5, (0, 0, 255), -1)
            end

            # 2. dibujar esqueleto (aristas)
            for (idx1, idx2) in CONNECTIONS
                # en Julia landmarks ya está indizado en 1
                p1 = landmarks[idx1+1] 
                p2 = landmarks[idx2+1]
                pycall(cv2.line, PyObject, frame, 
                         (Int(round(p1[1] * w)), Int(round(p1[2] * h))), 
                         (Int(round(p2[1] * w)), Int(round(p2[2] * h))), 
                         (0, 255, 0), 2)
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
        if cv2.waitKey(1) & 0xFF == 27 break end # 27 es el código ASCII de la tecla ESC
    end
finally
    cap.release()
    cv2.destroyAllWindows()
    try detector.close() catch e end
    println(">> Programa finalizado.")
end