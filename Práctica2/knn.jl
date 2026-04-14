# ==============================================================================
# APRENDIZAJE AUTOMÁTICO - UNIVERSIDADE DA CORUÑA
# SECCIÓN 4.5: k-NEAREST NEIGHBORS (kNN) - LSE
# ==============================================================================
using Random, Statistics, DataFrames, CSV, Printf
include("pr1/soluciones2.jl") 

# Semilla aleatoria obligatoria para garantizar repetibilidad 
Random.seed!(42)

println("Cargando características geométricas (Landmarks LSE)...")
ruta_csv = "dataset/landmarks_lse.csv"

if !isfile(ruta_csv)
    error("ERROR: No se encuentra el CSV. Comprueba que tienes la carpeta 'dataset' extraída en tu carpeta.")
end

# Lee el csv generado con MediaPipe
df = CSV.read(ruta_csv, DataFrame)
inputs = Matrix(df[:, 1:63])
# Nos aseguramos de que sean Float64 para los cálculos de distancia
inputs = Float64.(inputs) 
targets = Vector(df[:, :letra])
clases_presentes = sort(unique(targets))

# Valores de k a evaluar (se exigen al menos 6 configuraciones, usamos impares)
valores_k = [1, 3, 5, 7, 9, 11, 15] 
indices_cv = crossvalidation(targets, 5) 

# Variables para guardar los mejores resultados
mejor_acc = -1.0
mejor_matriz = nothing
mejor_k = -1

println("\n" * "="^65)
println("RESULTADOS EXPERIMENTALES: k-NEAREST NEIGHBORS (kNN)")
println("="^65)
@printf("%-15s %-20s %-20s\n", "Valor de k", "Accuracy (Media)", "Desv. Típica")
println("-"^65)

for k in valores_k
    # Hiperparámetros ajustados a tu soluciones2.jl
    params = Dict("n_neighbors" => k)
    
    try
        # modelCrossValidation devuelve métricas y la matriz
        resultados = modelCrossValidation(:KNeighborsClassifier, params, (inputs, targets), indices_cv)
        
        acc_media, acc_std = resultados[1]
        matriz_confusion = resultados[8] # la matriz está en la posición 8
        
        @printf("%-15d %-20.4f %-20.4f\n", k, acc_media, acc_std)
        
        # Guarda la matriz del modelo con mejor Accuracy
        if acc_media > mejor_acc
            global mejor_acc = acc_media
            global mejor_matriz = matriz_confusion
            global mejor_k = k
        end
    catch e
        println("ERROR con k=$k: $(e)")
    end
end
println("-"^65)

# Imprime la matriz de confusión del mejor modelo
if !isnothing(mejor_matriz)
    println("\nMATRIZ DE CONFUSIÓN (Mejor configuración: k=$(mejor_k)):")
    
    # Ajuste de cabecera para que cuadre exactamente con los 4 espacios de [%2d]
    print("     ") 
    for c in clases_presentes
        @printf("  %s ", c)
    end
    println()

    for i in 1:size(mejor_matriz, 1)
        @printf("%-4s ", clases_presentes[i])
        for j in 1:size(mejor_matriz, 2)
            @printf("[%2d]", round(Int, mejor_matriz[i, j]))
        end
        println()
    end
end
println("="^65)
