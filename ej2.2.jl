function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    #Codigo a desarrollar
    resultado = outputs .>= threshold
    return resultado 
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    #Codigo a desarrollar
    if size(outputs, 2)== 1
        vector= outputs[:]
        results = classifyOutputs(vector; threshold=threshold)
        return reshape(results,:,1)
    else 
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        outputsbool= falses(size(outputs))
        outputsbool[indicesMaxEachInstance] .= true
        return outputsbool
    end
end;


function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #
    # Codigo a desarrollar
    #
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    #
    # Codigo a desarrollar
    #
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

