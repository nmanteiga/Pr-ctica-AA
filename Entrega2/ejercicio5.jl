# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    # Codigo a desarrollar
    vector= collect(1:k)
    repeats= ceil(Int64, N/k)
    clonados= repeat(vector, repeats)
    recortados = clonados[1:N]
    shuffle!(recortados)
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    # Codigo a desarrollar
    indices = zeros(Int64, length(targets))
    numpositivos= sum(targets)
    indices[targets] = crossvalidation(numpositivos, k)
    numnegativos = length(indices) - numpositivos
    indices[.!targets] = crossvalidation(numnegativos, k)
    return indices
    
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    # Codigo a desarrollar
    vector = zeros(Int64, size(targets,1))
    for i in 1:size(targets,2)
        columna = targets[:, i]
        numpositivos= sum(columna)
        vector[columna] = crossvalidation(numpositivos, k)
    end
    return vector
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    # Codigo a desarrollar
    matriz_booleana = oneHotEncoding(targets)
    resultado =crossvalidation(matriz_booleana, k)
    return resultado
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)
    
    inputs = dataset[1]
    targets = dataset[2]
    classes = unique(targets)
    numClasses = length(classes)
    targets_bool = oneHotEncoding(targets, classes)
    num_folds = maximum(crossValidationIndices)
    N = size(inputs, 1)

    acc_folds = zeros(Float64, num_folds)
    err_folds = zeros(Float64, num_folds)
    sens_folds = zeros(Float64, num_folds)
    spec_folds = zeros(Float64, num_folds)
    vpp_folds = zeros(Float64, num_folds)
    vpn_folds = zeros(Float64, num_folds)
    f1_folds = zeros(Float64, num_folds)
    
    matriz_confusion_global = zeros(Float64, numClasses, numClasses)

    for k in 1:num_folds
        test_idx = (crossValidationIndices .== k)
        train_idx = (crossValidationIndices .!= k)

        train_inputs = inputs[train_idx, :]
        train_targets = targets_bool[train_idx, :]
        test_inputs = inputs[test_idx, :]
        test_targets = targets_bool[test_idx, :]

        acc_exec = zeros(Float64, numExecutions)
        err_exec = zeros(Float64, numExecutions)
        sens_exec = zeros(Float64, numExecutions)
        spec_exec = zeros(Float64, numExecutions)
        vpp_exec = zeros(Float64, numExecutions)
        vpn_exec = zeros(Float64, numExecutions)
        f1_exec = zeros(Float64, numExecutions)
        cm_exec = zeros(Float64, numClasses, numClasses, numExecutions)

        N_train = sum(train_idx)
        val_ratio_fold = (validationRatio * N) / N_train

        for e in 1:numExecutions
            if validationRatio > 0
                train_val_idx, val_idx = holdOut(N_train, val_ratio_fold)
                
                t_inputs = train_inputs[train_val_idx, :]
                t_targets = train_targets[train_val_idx, :]
                v_inputs = train_inputs[val_idx, :]
                v_targets = train_targets[val_idx, :]
                
                ann, _ = trainClassANN(topology, (t_inputs, t_targets); 
                                    validationDataset=(v_inputs, v_targets),
                                    transferFunctions=transferFunctions,
                                    maxEpochs=maxEpochs, minLoss=minLoss, 
                                    learningRate=learningRate, maxEpochsVal=maxEpochsVal)
            else
                ann, _ = trainClassANN(topology, (train_inputs, train_targets); 
                                    transferFunctions=transferFunctions,
                                    maxEpochs=maxEpochs, minLoss=minLoss, 
                                    learningRate=learningRate)
            end

            outputs = ann(test_inputs')'

            acc, err, sens, spec, vpp, vpn, f1, conf = confusionMatrix(outputs, test_targets)
            
            acc_exec[e] = acc
            err_exec[e] = err
            sens_exec[e] = sens
            spec_exec[e] = spec
            vpp_exec[e] = vpp
            vpn_exec[e] = vpn
            f1_exec[e] = f1
            cm_exec[:, :, e] = conf
        end

        acc_folds[k] = mean(acc_exec)
        err_folds[k] = mean(err_exec)
        sens_folds[k] = mean(sens_exec)
        spec_folds[k] = mean(spec_exec)
        vpp_folds[k] = mean(vpp_exec)
        vpn_folds[k] = mean(vpn_exec)
        f1_folds[k] = mean(f1_exec)

        matriz_confusion_global .+= mean(cm_exec, dims=3)[:, :, 1]
    end

    return (
        (mean(acc_folds), std(acc_folds)),
        (mean(err_folds), std(err_folds)),
        (mean(sens_folds), std(sens_folds)),
        (mean(spec_folds), std(spec_folds)),
        (mean(vpp_folds), std(vpp_folds)),
        (mean(vpn_folds), std(vpn_folds)),
        (mean(f1_folds), std(f1_folds)),
        matriz_confusion_global
    )
end
