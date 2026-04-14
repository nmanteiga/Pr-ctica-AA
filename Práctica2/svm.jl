# ==============================================================================
# APRENDIZAJE AUTOMÁTICO - UNIVERSIDADE DA CORUÑA
# SECCIÓN 4.3: SVM - LSE
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

# pidense polo menos 8 configuracions de hiperparámetros (kernel e C)
kernels = ["linear", "rbf", "poly"]
valores_C = [0.1, 1.0, 10.0, 100.0]
configuraciones = [
    ("linear",1.0),
    ("linear",10.0),
    ("linear",100.0),
    ("rbf",1.0),
    ("rbf",10.0),
    ("rbf",100.0),
    ("sigmoid",1.0),
    ("sigmoid",10.0),
    ("sigmoid",100.0),
]

indices_cv = crossvalidation(targets, 5)
# variables para os mellores resultados
mejor_acc = -1.0
mejor_matriz = nothing
mellor_config = ("", 0.0)

println("\n" * "="^65)
println("RESULTADOS EXPERIMENTALES: SVM")
println("="^65)
@printf("%-12s %-10s %-20s %-20s\n", "Kernel", "C", "Accuracy (Media)", "Desv. Típica")
println("-"^65)

for (kernel, C) in configuraciones
    # todos os hiperparámetros para os kernels
    params = Dict{String, Any}(
        "kernel" => kernel, 
        "C" => C,
        "gamma" => 1/63, #landmarks (21 puntos * 3 eixos)
        "coef0" => 0   
    )

    resultados = modelCrossValidation(:SVC, params, (inputs, targets), indices_cv)

    acc_media, acc_std = resultados[1]
    matriz_confusion = resultados[8] 

    @printf("%-12s %-10.1f %-20.4f %-20.4f\n", kernel, C, acc_media, acc_std)

    if acc_media > mejor_acc
        global mejor_acc = acc_media
        global mejor_matriz = matriz_confusion
        global mellor_config = (kernel, C)
    end
end
println("-"^65)
@printf("Mejor configuración: kernel=%-8s C=%.1f   Accuracy=%.4f\n",
        mellor_config[1], mellor_config[2], mejor_acc)

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