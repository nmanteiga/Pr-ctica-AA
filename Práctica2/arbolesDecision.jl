# ==============================================================================
# APRENDIZAJE AUTOMÁTICO - UNIVERSIDADE DA CORUÑA
# SECCIÓN 4.4: ÁRBOLES DE DECISIÓN - LSE
# ==============================================================================
using Random, Statistics, DataFrames, CSV, Printf
include("pr1/soluciones2.jl") 

# semilla aleatoria obligatoria para garantizar repetibilidad 
Random.seed!(42)

println("Cargando características geométricas (Landmarks LSE)...")
ruta_csv = "dataset/landmarks_lse.csv"

if !isfile(ruta_csv)
    error("ERROR: No se encuentra el CSV. Ejecuta primero generar_dataset.py en Python.")
end

# lee el csv generado con MediaPipe
df = CSV.read(ruta_csv, DataFrame)
inputs = Matrix(df[:, 1:63])
targets = Vector(df[:, :letra])
clases_presentes = sort(unique(targets))

# al menos 6 profundidades
profundidades = [2, 4, 8, 12, 16, 20] 
indices_cv = crossvalidation(targets, 5) 
# variables para guardar los mejores rtos de la matriz
mejor_acc = -1.0
mejor_matriz = nothing

println("\n" * "="^65)
println("RESULTADOS EXPERIMENTALES: ÁRBOLES DE DECISIÓN")
println("="^65)
@printf("%-15s %-20s %-20s\n", "Profundidad", "Accuracy (Media)", "Desv. Típica")
println("-"^65)

for d in profundidades
    # hiperparámetros
    params = Dict("max_depth" => d)
    
    # modelCrossValidation devuelve una tupla con métricas y la matriz
    resultados = modelCrossValidation(:DecisionTreeClassifier, params, (inputs, targets), indices_cv)
    
    acc_media, acc_std = resultados[1]
    matriz_confusion = resultados[8] # la matriz está en la posición 8
    
    @printf("%-15d %-20.4f %-20.4f\n", d, acc_media, acc_std)
    
    # guarda la matriz del modelo con mejor Accuracy
    if acc_media > mejor_acc
        global mejor_acc = acc_media
        global mejor_matriz = matriz_confusion
    end
end
println("-"^65)

# imprime la matriz de composición
println("\nMATRIZ DE CONFUSIÓN (Mejor configuración):")
print("      ")
for c in clases_presentes print(" $(c) ") end
println()

for i in 1:size(mejor_matriz, 1)
    @printf("%-5s", clases_presentes[i])
    for j in 1:size(mejor_matriz, 2)
        @printf("[%2d]", mejor_matriz[i, j])
    end
    println()
end
println("="^65)