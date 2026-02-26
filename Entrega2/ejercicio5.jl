# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    # Codigo a desarrollar
    vector= collect(1:k)
    repeats= ceil(Int64, N/k)
    clonados= repeat(vector, repeats)
    recortados = clonados[1:N]
    shuffle!(recortados)
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    # Codigo a desarrollar
    indices = zeros(Int64, length(targets))
    numpositivos= sum(targets)
    indices[targets] = crossvalidation(numpositivos, k)
    numnegativos = length(indices) - numpositivos
    indices[.!targets] = crossvalidation(numnegativos, k)
    return indices
    
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    # Codigo a desarrollar
    vector = zeros(Int64, size(targets,1))
    for i in 1:size(targets,2)
        columna = targets[:, i]
        numpositivos= sum(columna)
        vector[columna] = crossvalidation(numpositivos, k)
    end
    return vector
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    # Codigo a desarrollar
    matriz_booleana = oneHotEncoding(targets)
    resultado =crossvalidation(matriz_booleana, k)
    return resultado
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;
