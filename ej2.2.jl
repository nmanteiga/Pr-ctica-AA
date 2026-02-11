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
        outputs= falses(size(outputs))
        outputs[indicesMaxEachInstance] .= true
        return outputs
    end
end;

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # Codigo a desarrollar
    return mean(outputs.== targets)
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    # Codigo a desarrollar
    if size(outputs, 2)== 1
        vector1= outputs[:,1]
        vector2 =targets[:,1]
        results = accuracy(vector1,vector2)
        return results
    else
        return mean(eachrow(outputs) .== eachrow(targets))
    end
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



