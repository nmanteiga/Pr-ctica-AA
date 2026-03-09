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
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

using SymDoME
using GeneticProgramming


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;