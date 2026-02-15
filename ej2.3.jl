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

