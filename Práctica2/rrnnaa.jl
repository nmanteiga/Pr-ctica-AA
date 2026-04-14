# ==============================================================================
# APRENDIZAJE AUTOMÁTICO - UNIVERSIDADE DA CORUÑA
# SECCIÓN 4.2: REDES NEURONALES ARTIFICIALES (RR.NN.AA.) - LSE
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
inputs = Float32.(inputs)
targets = Vector(df[:, :letra])
clases_presentes = sort(unique(targets))

# arquitecturas: 8 configuraciones distintas (1-2 capas ocultas)
# formato: topology es un array con el número de neuronas en cada capa oculta
arquitecturas = [
    (topology = [32], learningRate = 0.01, maxEpochs = 100, numExecutions = 10),           # 1 capa: 32
    (topology = [64], learningRate = 0.01, maxEpochs = 100, numExecutions = 10),           # 1 capa: 64
    (topology = [128], learningRate = 0.01, maxEpochs = 100, numExecutions = 10),          # 1 capa: 128
    (topology = [256], learningRate = 0.005, maxEpochs = 100, numExecutions = 10),         # 1 capa: 256
    (topology = [64, 32], learningRate = 0.01, maxEpochs = 100, numExecutions = 10),       # 2 capas: 64-32
    (topology = [128, 64], learningRate = 0.005, maxEpochs = 100, numExecutions = 10),     # 2 capas: 128-64
    (topology = [256, 128], learningRate = 0.005, maxEpochs = 100, numExecutions = 10),    # 2 capas: 256-128
    (topology = [96, 48], learningRate = 0.01, maxEpochs = 100, numExecutions = 10),       # 2 capas: 96-48
]

indices_cv = crossvalidation(targets, 5) 
# variables para guardar los mejores resultados de la matriz
mejor_acc = -1.0
mejor_matriz = nothing
mejor_arquitectura = nothing

println("\n" * "="^80)
println("RESULTADOS EXPERIMENTALES: REDES NEURONALES ARTIFICIALES")
println("="^80)
@printf("%-30s %-20s %-20s\n", "Arquitectura", "Accuracy (Media)", "Desv. Típica")
println("-"^80)

for (idx, arch) in enumerate(arquitecturas)
    # crear descripción de arquitectura
    capas_str = join(arch.topology, "-")
    arch_desc = "$capas_str | LR: $(arch.learningRate) | Epochs: $(arch.maxEpochs)"
    
    try
        # hiperparámetros para el modelo
        params = Dict(
            "topology" => arch.topology,
            "learningRate" => arch.learningRate,
            "maxEpochs" => arch.maxEpochs,
            "numExecutions" => arch.numExecutions,
            "validationRatio" => 0.2
        )
        
        # modelCrossValidation devuelve métricas y la matriz de confusión
        resultados = modelCrossValidation(:ANN, params, (inputs, targets), indices_cv)
        
        acc_media, acc_std = resultados[1]
        matriz_confusion = resultados[8] # la matriz está en la posición 8
        
        @printf("%-30s %-20.4f %-20.4f\n", arch_desc, acc_media, acc_std)
        
        # guarda la matriz del modelo con mejor Accuracy
        if acc_media > mejor_acc
            global mejor_acc = acc_media
            global mejor_matriz = matriz_confusion
            global mejor_arquitectura = arch_desc
        end
    catch e
        println("ERROR en arquitectura $arch_desc: $(e)")
        @printf("%-30s %-20s %-20s\n", arch_desc, "ERROR", "N/A")
    end
end
println("-"^80)

# imprime la mejor configuración encontrada
println("\n" * "="^80)
println("MEJOR CONFIGURACIÓN: $mejor_arquitectura")
println("ACCURACY MÁXIMO: $(mejor_acc)")
println("="^80)

# imprime la matriz de confusión
if !isnothing(mejor_matriz)
    println("\nMATRIZ DE CONFUSIÓN (Mejor configuración):")
    print("      ")
    for c in clases_presentes print(" $(c) ") end
    println()

    for i in 1:size(mejor_matriz, 1)
        @printf("%-5s", clases_presentes[i])
        for j in 1:size(mejor_matriz, 2)
            @printf("[%2d]", round(Int, mejor_matriz[i, j]))
        end
        println()
    end
end
println("="^80)
