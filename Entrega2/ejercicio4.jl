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
    
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
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