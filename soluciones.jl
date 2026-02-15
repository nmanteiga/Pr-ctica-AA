# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses


function oneHotEncoding(feature::AbstractArray{<:Any,1},  
classes::AbstractArray{<:Any,1}) 
    if length(classes)<=2
        esClase1 = feature .== classes[1];
        return reshape(esClase1, :, 1);
    else
        oneHot = Array{Bool,2}(undef, length(feature), length(classes));
        for numClass in 1:length(classes)
            esEstaClase = feature .== classes[numClass];
            oneHot[:,numClass] .= esEstaClase;
        end;
        return oneHot;
    end;
end;

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    min = minimum(dataset,dims=1);
    max = maximum(dataset,dims=1);
    return (min,max);
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    media = mean(dataset,dims=1);
    desviacion = std(dataset,dims=1);
    return (media,desviacion);
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minVal = normalizationParameters[1];
    maxVal = normalizationParameters[2];
    dataset .-= minVal;
    dataset ./= (maxVal.-minVal);
    dataset[:,vec(minVal.==maxVal)] .= 0;
    return dataset;
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset);
    return normalizeMinMax!(dataset,normalizationParameters);
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    return normalizeMinMax!(copy(dataset),normalizationParameters);
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    return normalizeMinMax!(copy(dataset));
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    media = normalizationParameters[1];
    desviacion = normalizationParameters[2];
    dataset .-= media;
    dataset ./= desviacion;
    dataset[:,vec(desviacion.==0)] .= 0;
    return dataset;
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset);
    return normalizeZeroMean!(dataset,normalizationParameters);
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    return normalizeZeroMean!(copy(dataset),normalizationParameters);
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    return normalizeZeroMean!(copy(dataset));
end;


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
    # Codigo a desarrollar
    realtobool=classifyOutputs(outputs, threshold=threshold)
    accuracy(realtobool, targets)
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    # Codigo a desarrollar
    conversion= classifyOutputs(outputs, threshold=threshold)
    
    accuracy(conversion, targets)
end;

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain()
    numInputsLayer = numInputs
    
    for (i, numOutputsLayer) in enumerate(topology)
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]))
        numInputsLayer = numOutputsLayer
    end
    
    if numOutputs == 1
        #Para a clasificación binaria
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        #Para a clasificación multiclase
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs), softmax)
    end
    
    return ann
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    inputs, targets = dataset
    
    inputs = Float32.(inputs)
    targets = Float32.(targets)
    
    numInputs = size(inputs, 2)
    numOutputs = size(targets, 2)
    
    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)
    
    opt_state = Flux.setup(Adam(learningRate), ann)
    
    x = inputs'
    y = targets'
    
    #binarycrossentropy para clasificación binaria e crossentropy para clasificación multiclase
    loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)
    
    losses = Float32[]
    
    push!(losses, Float32(loss(ann, x, y)))
    
    epoch = 0
    while epoch < maxEpochs && losses[end] > minLoss
        Flux.train!(loss, ann, [(x, y)], opt_state)
        push!(losses, Float32(loss(ann, x, y)))
        epoch += 1
    end
    
    return (ann, losses)
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    targetsMatrix = reshape(targets, :, 1)

    return trainClassANN(topology, (inputs, targetsMatrix); transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)
end;

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    #
    # Codigo a desarrollar
    #
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;


