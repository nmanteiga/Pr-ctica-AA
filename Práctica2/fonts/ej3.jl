using Random

function holdOut(N::Int, P::Real)
    perm = randperm(N)                  # 1..N
    n_test = round(Int, P * N)          # patrones para test
    test_idx = perm[1:n_test]           # primeros para test
    train_idx = perm[n_test+1:end]      # resto para entrenamiento
    return (train_idx, test_idx)
end

# Nueva función con train/val/test
function holdOut(N::Int, Pval::Real, Ptest::Real)
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