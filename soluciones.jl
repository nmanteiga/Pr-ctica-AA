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

function holdOut(N, P)
    perm = randperm(N)                  # 1..N
    n_test = round(Int, P * N)          # patrones para test
    test_idx = perm[1:n_test]           # primeros para test
    train_idx = perm[n_test+1:end]      # resto para entrenamiento
    return (train_idx, test_idx)
end

# Nueva función con train/val/test
function holdOut(N, Pval, Ptest)
    # 1) Separar test
    train1_idx, test_idx = holdOut(N, Ptest)

    # 2) Ajustar la tasa de validación al tamaño del nuevo subconjunto
    Pval_adjusted = (Pval * N) / length(train1_idx)

    # 3) Separar validación
    val_train, val_idx = holdOut(length(train1_idx), Pval_adjusted)

    # 4) Mapear índices locales a globales
    val_idx = train1_idx[val_idx]
    train_idx = train1_idx[val_train]

    return (train_idx, val_idx, test_idx)
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    # se contruye la rna
    n_inputs = size(trainingDataset[1], 2)
    n_outputs = size(trainingDataset[2], 2)
    ann = buildClassANN(n_inputs, topology, n_outputs; transferFunctions=transferFunctions)

    # configuración del entrenamiento
    loss(m, x, y) = (size(y, 1) == 1) ? Flux.Losses.binarycrossentropy(m(x), y) : Flux.Losses.crossentropy(m(x), y)
    
    # optimizador ADAM
    opt_state = Flux.setup(Adam(learningRate), ann)

    x_train = Float32.(trainingDataset[1]')
    y_train = trainingDataset[2]'
    x_val = Float32.(validationDataset[1]')
    y_val = validationDataset[2]'
    x_test = Float32.(testDataset[1]')
    y_test = testDataset[2]'

    # vectores históricos de loss
    train_losses = Float32[]
    val_losses = Float32[]
    test_losses = Float32[]

    # variables para early-stop 
    bestVal_loss = Inf32
    best_ann = deepcopy(ann)
    epochsSince_best = 0

    # cálculo del loss incial (ciclo 0)
    train_loss_0 = loss(ann, x_train, y_train)
    push!(train_losses, train_loss_0)

    if !isempty(validationDataset[1])
        val_loss0 = loss(ann, x_val, y_val)
        push!(val_losses, val_loss0)
        bestVal_loss = val_loss0
    end

    if !isempty(testDataset[1])
        push!(test_losses, loss(ann, x_test, y_test))
    end

    # ciclo de entrenamiento
    for epoch in 1:maxEpochs
        # entrenamiento ciclo 1
        Flux.train!(loss, ann, [(x_train, y_train)], opt_state)

        # cálculo y guardado de loss
        current_loss = loss(ann, x_train, y_train)
        push!(train_losses, current_loss)

        if !isempty(testDataset[1])
            push!(test_losses, loss(ann, x_test, y_test))
        end

        # lógica de validación y early-stop
        if !isempty(validationDataset[1])
            currentVal_loss = loss(ann, x_val, y_val)
            push!(val_losses, currentVal_loss)

            if currentVal_loss < bestVal_loss
                bestVal_loss = currentVal_loss
                best_ann = deepcopy(ann)
                epochsSince_best = 0
            else
                epochsSince_best += 1
            end

            # early-stop
            if epochsSince_best >= maxEpochsVal
                return (best_ann, train_losses, val_losses, test_losses)
            end
        end

        # parada por minloss
        if current_loss <= minLoss
            break
        end
    end
    
    # devuelve la mejor validadción si no se para(enunciado)
    final_ann = isempty(validationDataset[1]) ? ann : best_ann
    return (final_ann, train_losses, val_losses, test_losses)
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    # conversión vectores a matrices
    train_X = trainingDataset[1]
    train_Y = reshape(trainingDataset[2], :, 1)
    
    # validación
    val_X = validationDataset[1]
    val_Y = reshape(validationDataset[2], :, 1)

    # testing
    test_X = testDataset[1]
    test_Y = reshape(testDataset[2], :, 1)

    # llamada a la función anterior
    return trainClassANN(topology, (train_X, train_Y);
        validationDataset = (val_X, val_Y),
        testDataset       = (test_X, test_Y),
        transferFunctions = transferFunctions,
        maxEpochs         = maxEpochs,
        minLoss           = minLoss,
        learningRate      = learningRate,
        maxEpochsVal      = maxEpochsVal)
end