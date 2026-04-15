# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VN = sum((outputs .== 0) .& (targets .== 0))
    FN = sum((outputs .== 0) .& (targets .== 1))
    FP = sum((outputs .== 1) .& (targets .== 0))
    VP = sum((outputs .== 1) .& (targets .== 1))
    precision = (VN + VP) / (VN + VP + FN + FP)
    tasa_error = (FP + FN) / (VN + VP + FN + FP)
    sensibilidad = VP + FN == 0 ? 1 : VP / (VP + FN)
    especificidad = VN + FP == 0 ? 1 : VN / (VN + FP)
    vpp = VP + FP == 0 ? 1 : VP / (VP + FP)
    vpn = VN + FN == 0 ? 1 : VN / (VN + FN)
    F1 = sensibilidad + vpp == 0 ? 0 : 2 * vpp * sensibilidad / (vpp + sensibilidad)
    matriz = [VN FP; FN VP]
    return precision, tasa_error, sensibilidad, especificidad, vpp, vpn, F1, matriz
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs_bool = outputs .>= threshold
    return confusionMatrix(outputs_bool, targets)
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    if size(outputs, 2) != size(targets, 2) || size(outputs, 2) == 2
        throw(ArgumentError("El número de columnas debe ser igual y distinto de 2"))
    elseif size(outputs, 2) == 1
        vec_outputs = vec(outputs)
        vec_targets = vec(targets)
        return confusionMatrix(vec_outputs, vec_targets)
    else
        sensibilidad = zeros(size(outputs, 2))
        especificidad = zeros(size(outputs, 2))
        vpp = zeros(size(outputs, 2))
        vpn = zeros(size(outputs, 2))
        F1 = zeros(size(outputs, 2))
        for i in 1:size(outputs, 2)
            _, _, sensibilidad[i], especificidad[i], vpp[i], vpn[i], F1[i], _ = confusionMatrix(vec(outputs[:, i]), vec(targets[:, i]))
        end
        matriz = targets' * outputs
        counts = vec(sum(targets, dims=1))
        if weighted
            total = sum(counts)
            sensibilidadM = sum(counts .* sensibilidad) / total
            especificidadM = sum(counts .* especificidad) / total
            vppM = sum(counts .* vpp) / total
            vpnM = sum(counts .* vpn) / total
            F1M = sum(counts .* F1) / total
        else
            sensibilidadM = mean(sensibilidad)
            especificidadM = mean(especificidad)
            vppM = mean(vpp)
            vpnM = mean(vpn)
            F1M = mean(F1)
        end
        precision = accuracy(outputs, targets)
        tasa_error = 1 - precision
        return precision, tasa_error, sensibilidadM, especificidadM, vppM, vpnM, F1M, matriz
    end
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    conversion = classifyOutputs(outputs, threshold=threshold)
    return confusionMatrix(conversion, targets; weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    if length(outputs) != length(targets)
        throw(ArgumentError("outputs y targets deben tener la misma longitud"))
    end
    if !(all(in.(outputs, Ref(classes))) && all(in.(targets, Ref(classes))))
        throw(ArgumentError("Todos los valores de outputs y targets deben estar en classes"))
    end
    outputs_enc = oneHotEncoding(outputs, classes)
    targets_enc = oneHotEncoding(targets, classes)
    return confusionMatrix(outputs_enc, targets_enc; weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))
    return confusionMatrix(outputs, targets, classes; weighted=weighted)
end;

using SymDoME
using GeneticProgramming


# 1. Versión para clasificación BINARIA
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    inputs, targets = trainingDataset
    targetsFloat = Float64.(targets)
    
    # Añadimos un valor de reducción MSE muy pequeño para que explore más
    d = SymDoME.DoME(Float64.(inputs), targetsFloat; 
        dataInRows=true, 
        maximumNodes=maximumNodes, 
        minimumReductionMSE=1e-12) 
        
    PerformSearches!(d)
    
    testOutputsFloat = evaluateTree(d.tree, Float64.(testInputs); dataInRows=true)
    
    if isa(testOutputsFloat, Number)
        testOutputsFloat = fill(testOutputsFloat, size(testInputs,1))
    end
    return testOutputsFloat .>= 0.5
end 
# 2. Versión para clasificación MULTICLASE con Matriz (La que usa modelCrossValidation)
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    inputs, targets = trainingDataset
    numClasses = size(targets, 2)
    numTestSamples = size(testInputs, 1)
    scores = zeros(Float64, numTestSamples, numClasses)

    for c in 1:numClasses
        colTargets = Float64.(targets[:, c])
        d = SymDoME.DoME(Float64.(inputs), colTargets; dataInRows=true, maximumNodes=maximumNodes)
        PerformSearches!(d)
        
        # Obtenemos las predicciones del árbol para esta clase
        vals = evaluateTree(d.tree, Float64.(testInputs); dataInRows=true)
        
        # El operador .= es vital para que no falle la asignación
        if isa(vals, Number)
            scores[:, c] .= vals
        else
            scores[:, c] .= vals
        end
    end

    # Elegimos la clase con mayor puntuación (Winner Takes All)
    winners = argmax.(eachrow(scores))
    outputsBool = falses(numTestSamples, numClasses)
    for i in 1:numTestSamples
        outputsBool[i, winners[i]] = true
    end
    return outputsBool
end

# 3. Versión para clasificación con etiquetas (ej: "setosa", "virginica")
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    inputs, targets = trainingDataset
    classes = unique(targets)
    targetsBool = oneHotEncoding(targets, classes)
    
    # Llama a la versión 2 definida arriba
    outputsBool = trainClassDoME((inputs, targetsBool), testInputs, maximumNodes)

    numTestSamples = size(testInputs, 1)
    predictions = Vector{String}(undef, numTestSamples)
    for i in 1:numTestSamples
        idx = findfirst(outputsBool[i, :])
        predictions[i] = string(idx === nothing ? classes[1] : classes[idx])
    end
    return predictions
end