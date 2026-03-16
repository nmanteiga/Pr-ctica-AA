# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
    inputs, targets = dataset

    #ler hiperparametros do dicionario para string e symbol
    function getHP(d::Dict, key::String)
        if haskey(d, key)
            return d[key]
        elseif haskey(d, Symbol(key))
            return d[Symbol(key)]
        else
            return nothing
        end
    end

    #para RNA
    if modelType == :ANN
        topology = getHP(modelHyperparameters, "topology")
        learningRate = getHP(modelHyperparameters, "learningRate")
        validationRatio = getHP(modelHyperparameters, "validationRatio")
        numExecutions = getHP(modelHyperparameters, "numExecutions")
        maxEpochs = getHP(modelHyperparameters, "maxEpochs")
        maxEpochsVal = getHP(modelHyperparameters, "maxEpochsVal")

        kwargs = Dict{Symbol,Any}()
        learningRate !== nothing && (kwargs[:learningRate] = learningRate)
        validationRatio !== nothing && (kwargs[:validationRatio] = validationRatio)
        numExecutions !== nothing && (kwargs[:numExecutions] = numExecutions)
        maxEpochs !== nothing && (kwargs[:maxEpochs] = maxEpochs)
        maxEpochsVal !== nothing && (kwargs[:maxEpochsVal] = maxEpochsVal)

        #funcion exercicio anterior
        return ANNCrossValidation(topology, (inputs, targets), crossValidationIndices; kwargs...)
    end

    #resto
    targets = string.(targets)
    classes = unique(targets)
    numFolds = maximum(crossValidationIndices)
    numClasses = length(classes)

    #vectores metricas
    acc_v = Vector{Float64}(undef, numFolds)
    err_v = Vector{Float64}(undef, numFolds)
    sens_v = Vector{Float64}(undef, numFolds)
    spec_v = Vector{Float64}(undef, numFolds)
    vpp_v = Vector{Float64}(undef, numFolds)
    vpn_v = Vector{Float64}(undef, numFolds)
    f1_v = Vector{Float64}(undef, numFolds)
    confMat = zeros(Int, numClasses, numClasses)

    for fold in 1:numFolds
        testIdx = crossValidationIndices .== fold
        trainIdx = .!testIdx

        trainInputs = inputs[trainIdx, :]
        testInputs = inputs[testIdx, :]
        trainTargets = targets[trainIdx]
        testTargets = targets[testIdx]

        if modelType == :DoME
            #funcion do exercicio 4
            testOutputs = trainClassDoME((trainInputs, trainTargets), testInputs,getHP(modelHyperparameters, "maximumNodes"))
        else
            if modelType == :SVC
                kernel_str = getHP(modelHyperparameters, "kernel")
                C = getHP(modelHyperparameters, "C")

                if kernel_str == "linear"
                    model = SVMClassifier(kernel = LIBSVM.Kernel.Linear, cost = Float64(C))

                elseif kernel_str == "rbf"
                    model = SVMClassifier(kernel = LIBSVM.Kernel.RadialBasis, cost = Float64(C), gamma = Float64(getHP(modelHyperparameters, "gamma")))

                elseif kernel_str == "sigmoid"
                    model = SVMClassifier(kernel = LIBSVM.Kernel.Sigmoid, cost = Float64(C), gamma = Float64(getHP(modelHyperparameters, "gamma")), coef0 = Int32(getHP(modelHyperparameters, "coef0")))

                elseif kernel_str == "poly"
                    model = SVMClassifier(kernel = LIBSVM.Kernel.Polynomial, cost = Float64(C), gamma = Float64(getHP(modelHyperparameters, "gamma")), degree = Float64(getHP(modelHyperparameters, "degree")), coef0 = Int32(getHP(modelHyperparameters, "coef0")))
                end

            elseif modelType == :DecisionTreeClassifier
                model = DTClassifier(max_depth = getHP(modelHyperparameters, "max_depth"), rng = Random.MersenneTwister(1)) #reproducibilidade

            elseif modelType == :KNeighborsClassifier
                model = kNNClassifier(K = getHP(modelHyperparameters, "n_neighbors"))

            end
            
            #entrenamento e prediccións
            mach = machine(model, MLJ.table(trainInputs), categorical(trainTargets))
            MLJ.fit!(mach, verbosity=0)
            testOutputs = MLJ.predict(mach, MLJ.table(testInputs))

            if modelType != :SVC
                testOutputs = mode.(testOutputs)
            end

            testOutputs = string.(testOutputs)
        end

        acc, err, sens, spec, vpp, vpn, f1, cm = confusionMatrix(testOutputs, testTargets, classes)
        acc_v[fold] = acc
        err_v[fold] = err
        sens_v[fold] = sens
        spec_v[fold] = spec
        vpp_v[fold] = vpp
        vpn_v[fold] = vpn
        f1_v[fold] = f1
        confMat += cm
    end

    return ((mean(acc_v), std(acc_v)), (mean(err_v), std(err_v)), (mean(sens_v), std(sens_v)), (mean(spec_v), std(spec_v)), (mean(vpp_v), std(vpp_v)), (mean(vpn_v), std(vpn_v)), (mean(f1_v), std(f1_v)), confMat)
end;