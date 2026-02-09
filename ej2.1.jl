using Statistics
using Flux
using Flux.Losses

using DelimitedFiles

    # Cargamos el dataset
    dataset = readdlm("iris.data",',');

    # Preparamos las entradas
    inputs = dataset[:,1:4];
    # Con cualquiera de estas 3 maneras podemos convertir la matriz de entradas de tipo Array{Any,2} en Array{Float32,2}, si los valores son numéricos:
    inputs = Float32.(inputs);
    inputs = convert(Array{Float32,2},inputs);
    inputs = [Float32(x) for x in inputs];
    println("Tamaño de la matriz de entradas: ", size(inputs,1), "x", size(inputs,2), " de tipo ", typeof(inputs));

    # Preparamos las salidas deseadas codificándolas puesto que son categóricas
    targets = dataset[:,5];
    println("Longitud del vector de salidas deseadas antes de codificar: ", length(targets), " de tipo ", typeof(targets));
    classes = unique(targets);
    numClasses = length(classes);
    if numClasses<=2
        # Si solo hay dos clases, se genera una matriz con una columna
        targets = reshape(targets.==classes[1], :, 1);
    else
        # Si hay mas de dos clases se genera una matriz con una columna por clase
        # Cualquiera de estos dos tipos (Array{Bool,2} o BitArray{2}) vale perfectamente
        # oneHot = Array{Bool,2}(undef, size(targets,1), numClasses);
        # oneHot =   BitArray{2}(undef, size(targets,1), numClasses);
        # for numClass = 1:numClasses
        #     oneHot[:,numClass] .= (targets.==classes[numClass]);
        # end;
        # Una forma de hacerlo sin bucles sería la siguiente:
        oneHot = convert(BitArray{2}, hcat([instance.==classes for instance in targets]...)');
        targets = oneHot;
    end;
    println("Tamaño de la matriz de salidas deseadas despues de codificar: ", size(targets,1), "x", size(targets,2), " de tipo ", typeof(targets));

    # Comprobamos que ambas matrices tienen el mismo número de filas
    @assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo numero de filas"


    # Realizamos la normalizacion, de un tipo u otro. Por ejemplo, mediante maximo y minimo:
    # Primero calculamos los valores de normalizacion
    normalizationParameters = (minimum(inputs, dims=1), maximum(inputs, dims=1));
    # Despues los leemos de esa tupla
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    # En realidad, no es necesario crear la tupla con los parámetros de normalización, se podrían calcular directamente los valores máximo y mínimo sin almacenarlos en una tupla
    #  Esto se hace así para que, al tenerlo como una tupla, sea más sencillo pasarla a una función, como exige el ejercicio siguiente
    # Finalmente, los aplicamos
    inputs .-= minValues;
    inputs ./= (maxValues .- minValues);
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    inputs[:, vec(minValues.==maxValues)] .= 0;



    # Dejamos aqui indicado como se haria para normalizar mediante media y desviacion tipica
    # normalizationParameters = (mean(inputs, dims=1), std(inputs, dims=1));
    # avgValues = normalizationParameters[1];
    # stdValues = normalizationParameters[2];
    # inputs .-= avgValues;
    # inputs ./= stdValues;
    # # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    # inputs[:, vec(stdValues.==0)] .= 0;




#=
function oneHotEncoding(feature::AbstractArray{<:Any,1},  
classes::AbstractArray{<:Any,1}) 
function oneHotEncoding(feature::AbstractArray{<:Any,1}) 
function oneHotEncoding(feature::AbstractArray{Bool,1}) 
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2}) 
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2}) 
function normalizeMinMax!(dataset::AbstractArray{<:Real,2},  
normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}) 
function normalizeMinMax( dataset::AbstractArray{<:Real,2},  
normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
function normalizeMinMax( dataset::AbstractArray{<:Real,2}) 
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},  
normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}) 
function normalizeZeroMean( dataset::AbstractArray{<:Real,2},  
normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
function normalizeZeroMean( dataset::AbstractArray{<:Real,2})
=#


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


# ==================== TESTS ====================
println("\n========== TESTS DE ONEHOT ENCODING ==========\n");

# Test 1: Función original con 2 clases
println("Test 1: Función original con 2 clases");
feature1 = ["A", "B", "A", "B", "A"];
classes1 = ["A", "B"];
resultado1 = oneHotEncoding(feature1, classes1);
println("Feature: $feature1");
println("Classes: $classes1");
println("Resultado:\n$resultado1");
println("Tipo: $(typeof(resultado1)), Tamaño: $(size(resultado1))\n");

# Test 2: Función original con 3 clases
println("Test 2: Función original con 3 clases");
feature2 = ["setosa", "versicolor", "virginica", "setosa", "versicolor"];
classes2 = ["setosa", "versicolor", "virginica"];
resultado2 = oneHotEncoding(feature2, classes2);
println("Feature: $feature2");
println("Classes: $classes2");
println("Resultado:\n$resultado2");
println("Tipo: $(typeof(resultado2)), Tamaño: $(size(resultado2))\n");

# Test 3: Sobrecarga 1 - Extrae categorías automáticamente
println("Test 3: Sobrecarga 1 - Extrae categorías automáticamente");
feature3 = [1, 2, 3, 1, 2, 3, 1];
resultado3 = oneHotEncoding(feature3);
println("Feature: $feature3");
println("Categorías extraídas: $(unique(feature3))");
println("Resultado:\n$resultado3");
println("Tipo: $(typeof(resultado3)), Tamaño: $(size(resultado3))\n");

# Test 4: Sobrecarga 2 - Vector booleano
println("Test 4: Sobrecarga 2 - Vector booleano");
feature4 = [true, false, true, false, true];
resultado4 = oneHotEncoding(feature4);
println("Feature: $feature4");
println("Resultado:\n$resultado4");
println("Tipo: $(typeof(resultado4)), Tamaño: $(size(resultado4))\n");

# Test 5: Verificación de exactitud - Test 2 debe coincidir con Test 3
println("Test 5: Verificación - ¿Test 2 y Test 3 dan el mismo resultado?");
feature5 = ["A", "B"];
resultado5a = oneHotEncoding(feature5, ["A", "B"]);
resultado5b = oneHotEncoding(feature5);
println("Con clases explícitas: $(resultado5a)'");
println("Con extracción automática: $(resultado5b)'");
println("¿Son iguales? $(resultado5a == resultado5b)\n");

println("========== TESTS COMPLETADOS ==========\n");

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    min = minimum(dataset, dims=1);
    max = maximum(dataset, dims=1);
    return (min,max);
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    media = mean(dataset, dims=1);
    desviacion = std(dataset, dims=1);
    return (media,desviacion);
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset);
    return normalizeMinMax!(dataset, normalizationParameters);
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    return normalizeMinMax!(copy(dataset), normalizationParameters);
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    return normalizeMinMax!(copy(dataset));
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    meanValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= meanValues;
    dataset ./= stdValues;
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset;
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset);
    return normalizeZeroMean!(dataset, normalizationParameters);
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    return normalizeZeroMean!(copy(dataset), normalizationParameters);
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    return normalizeZeroMean!(copy(dataset));
end;



