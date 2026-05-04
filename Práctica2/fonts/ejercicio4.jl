# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 -------------------------------------------
# ----------------------------------------------------------------------------------------------



function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    numInstances = length(targets);
    @assert(length(outputs)==numInstances);
    # Valores de la matriz de confusion
    TN = sum(.!outputs .& .!targets); # VerdaderOs negativos
    FN = sum(.!outputs .&   targets); # Falsos negativos
    TP = sum(  outputs .&   targets); # Verdaderos positivos
    FP = sum(  outputs .& .!targets); # Falsos negativos
    # Creamos la matriz de confusión, poniendo en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
    #  Primera fila/columna: negativos
    #  Segunda fila/columna: positivos
    confMatrix = [TN FP;
                  FN TP];
    # Metricas que se derivan de la matriz de confusion:
    acc         = (TN+TP)/(TN+FN+TP+FP);
    errorRate   = 1. - acc;
    # Para sensibilidad, especificidad, VPP y VPN controlamos que algunos casos pueden ser NaN
    #  Para el caso de sensibilidad y especificidad, en un conjunto de entrenamiento estos no pueden ser NaN, porque esto indicaria que se ha intentado entrenar con una unica clase
    #   Sin embargo, sí pueden ser NaN en el caso de aplicar un modelo en un conjunto de test, si este sólo tiene patrones de una clase
    #  Para VPP y VPN, sí pueden ser NaN en caso de que el clasificador lo haya clasificado todo como negativo o positivo respectivamente
    # En estos casos, estas metricas habria que dejarlas a NaN para indicar que no se han podido evaluar
    #  Sin embargo, como es posible que se quiera combinar estos valores al evaluar una clasificacion multiclase, es necesario asignarles un valor. El criterio que se usa aqui es que estos valores seran igual a 1
    #   Se utiliza este criterio porque, por ejemplo en el caso de recall (sensibilidad), no habría fallado en ningún positivo, porque no hay ninguno
    recall      = (TP==FN==0) ? 1. : TP/(TP+FN); # Sensibilidad
    specificity = (TN==FP==0) ? 1. : TN/(TN+FP); # Especificidad
    precision   = (TP==FP==0) ? 1. : TP/(TP+FP); # Va1or predictivo positivo
    NPV         = (TN==FN==0) ? 1. : TN/(TN+FN); # Valor predictivo negativo
    # Calculamos F1
    F1          = (recall==precision==0) ? 0. : 2*(recall*precision)/(recall+precision);
    @assert(!isnan(F1));
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;


confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = confusionMatrix(outputs.>=threshold, targets);


function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert(size(outputs)==size(targets));
    (numInstances, numClasses) = size(targets);
    # Nos aseguramos de que no hay dos columnas
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    end;

    # Nos aseguramos de que en cada fila haya uno y sólo un valor a true
    @assert(all(sum(outputs, dims=2).==1));
    # Reservamos memoria para las metricas de cada clase, inicializandolas a 0 porque algunas posiblemente no se calculen
    recall      = zeros(numClasses);
    specificity = zeros(numClasses);
    precision   = zeros(numClasses);
    NPV         = zeros(numClasses);
    F1          = zeros(numClasses);
    # Calculamos las metricas para cada clase, usando la función anterior para problemas de 2 clases
    for numClass in 1:numClasses
        # Calculamos las metricas de cada problema binario correspondiente a cada clase y las almacenamos en los vectores correspondientes
        (_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
    end;

    # Creamos la matriz de confusión
    confMatrix = targets' * outputs;

    # Aplicamos las formas de combinar las metricas macro o weighted
    if weighted
        # Calculamos los valores de ponderacion para hacer el promedio
        numInstancesFromEachClass = vec(sum(targets, dims=1));
        @assert(numInstances == sum(numInstancesFromEachClass));
        weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
        recall      = sum(weights.*recall);
        specificity = sum(weights.*specificity);
        precision   = sum(weights.*precision);
        NPV         = sum(weights.*NPV);
        F1          = sum(weights.*F1);
    else
        recall      = mean(recall);
        specificity = mean(specificity);
        precision   = mean(precision);
        NPV         = mean(NPV);
        F1          = mean(F1);
    end;
    # Precision y tasa de error las calculamos con las funciones definidas previamente
    acc = accuracy(outputs, targets);
    errorRate = 1 - acc;

    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;

confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs; threshold=threshold), targets; weighted=weighted)




function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
    @assert(all([in(label, classes) for label in vcat(targets, outputs)]));
    # Es importante pasar el mismo vector de clases como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
    return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs));
    return confusionMatrix(outputs, targets, classes; weighted=weighted);
end;



# Funciones auxiliares para visualizar por pantalla la matriz de confusion y las metricas que se derivan de ella
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;
printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) =  printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)



printConfusionMatrix(outputs::AbstractArray{Bool,1},   targets::AbstractArray{Bool,1})                      = printConfusionMatrix(reshape(outputs, :, 1), reshape(targets, :, 1));
printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = printConfusionMatrix(outputs.>=threshold,    targets);

printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true) = printConfusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs));
    printConfusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;    





using SymDoME


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)

    (trainingInputs, trainingTargets) = trainingDataset;

    trainingInputs = Float64.(trainingInputs);
    testInputs = Float64.(testInputs);

    model, _, _, _ = dome(trainingInputs, trainingTargets;
        maximumNodes = maximumNodes
    )

    # El resultado de evaluateTree puede ser un valor real, o un vector
    testOutputs = evaluateTree(model, testInputs);
    if isa(testOutputs, Real)
        testOutputs = repeat([testOutputs], size(testInputs,1));
    end;
    return testOutputs;
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)

    (trainingInputs, trainingTargets) = trainingDataset;
    numColumns = size(trainingTargets,2);

    if numColumns==1

        testOutputs = trainClassDoME((trainingInputs, vec(trainingTargets)), testInputs, maximumNodes);
        return reshape(testOutputs, :, 1)

    end;

    # Nos aseguramos de que hay mas de dos clases
    @assert(numColumns>2);

    testOutputs = Array{Float64,2}(undef, size(testInputs,1), numColumns);
    for numClass in Base.OneTo(numColumns)

        testOutputs[:,numClass] .= trainClassDoME((trainingInputs, trainingTargets[:,numClass]), testInputs, maximumNodes)

    end;
    return testOutputs
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)

    (trainingInputs, trainingTargets) = trainingDataset;
    classes = unique(trainingTargets);

    testOutputsDoME = trainClassDoME((trainingInputs, oneHotEncoding(trainingTargets, classes)), testInputs, maximumNodes);
    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0);

    testOutputs = Array{eltype(trainingTargets),1}(undef, size(testInputs,1));

    if length(classes)<=2
        @assert(isa(testOutputsBool, Vector) || size(testOutputsBool,2)==1)
        testOutputsBool = vec(testOutputsBool); # Esta línea es necesaria para versiones de Julia 1.10 o inferior. En estas, si no se pone, la línea siguiente daría error
        testOutputs[  testOutputsBool] .= classes[1];
        if length(classes)==2
            testOutputs[.!testOutputsBool] .= classes[2];
        else @assert(all(testOutputsBool))
        end;
    else
        @assert(all(sum(testOutputsBool, dims=2).==1));
        # En lugar de hacer este bucle, se podía hacer una función llamada "decodify"
        for numClass in eachindex(classes)
            testOutputs[testOutputsBool[:,numClass]] .= classes[numClass];
        end;
    end;
    return testOutputs;
end;