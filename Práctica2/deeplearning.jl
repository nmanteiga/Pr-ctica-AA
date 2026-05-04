# ==============================================================================================
# BLOQUE 1: LIBRERÍAS Y FUNCIONES DE LA PRÁCTICA 1
# ==============================================================================================

using Flux
using Flux.Losses
using Flux: onehotbatch, onecold, adjust!
using JLD2, FileIO
using Statistics: mean
using ImageTransformations
using Random # Para barajar los datos

# Cargar funciones desde el archivo de soluciones
include("fonts/soluciones.jl")

# --- Función para imprimir la Matriz de Confusión ---
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    numClasses = size(targets, 2)
    conf_matrix = zeros(Int, numClasses, numClasses)
    for i in 1:size(targets, 1)
        out_idx = findfirst(outputs[i, :])
        tgt_idx = findfirst(targets[i, :])
        if !isnothing(out_idx) && !isnothing(tgt_idx)
            conf_matrix[tgt_idx, out_idx] += 1
        end
    end
    println("--------------------------------------------------")
    println("Matriz de Confusión (Filas: Real, Columnas: Predicción):")
    display(conf_matrix)
    println("--------------------------------------------------")
end


# ==============================================================================================
# BLOQUE 2: FUNCIÓN PARA VALIDACIÓN CRUZADA (K-FOLD)
# ==============================================================================================
function obtener_indices_cv(num_patrones, num_folds, fold_actual, indices_aleatorios)
    tamano_fold = div(num_patrones, num_folds)
    inicio_test = (fold_actual - 1) * tamano_fold + 1
    fin_test = fold_actual == num_folds ? num_patrones : fold_actual * tamano_fold
    
    indices_test = indices_aleatorios[inicio_test:fin_test]
    indices_train = setdiff(indices_aleatorios, indices_test)
    
    return indices_train, indices_test
end


# ==============================================================================================
# BLOQUE 3: CARGA DE DATOS 
# ==============================================================================================
println("Cargando datos de MNIST.jld2...")
train_imgs_mnist   = JLD2.load("MNIST.jld2", "train_imgs");
train_labels_mnist = JLD2.load("MNIST.jld2", "train_labels");
test_imgs_mnist    = JLD2.load("MNIST.jld2", "test_imgs");
test_labels_mnist  = JLD2.load("MNIST.jld2", "test_labels");

# ¡CORTAMOS A 1000 IMÁGENES PARA QUE NO TARDE UNA ETERNIDAD!
mis_imagenes_totales  = vcat(train_imgs_mnist, test_imgs_mnist)[1:1000]
mis_etiquetas_totales = vcat(train_labels_mnist, test_labels_mnist)[1:1000]

labels = 0:9
num_patrones_totales = length(mis_imagenes_totales)

indices_aleatorios_globales = randperm(num_patrones_totales)
println("Datos cargados y listos. Total de imágenes: ", num_patrones_totales)
# ==============================================================================================
# BLOQUE 4: CÓDIGO DEL PROFESOR ADAPTADO
# ==============================================================================================

### MODIFICACIÓN 1: Cambiamos la función del profesor para que haga el 'imresize' a un tamaño fijo (28x28)
function convertirArrayImagenesWHCN(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, 28, 28, 1, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        # Redimensionamos la imagen actual a 28x28 para evitar errores de tamaño
        img_redimensionada = imresize(imagenes[i], (28, 28)) 
        nuevoArray[:,:,1,i] .= img_redimensionada[:,:]';
    end;
    return nuevoArray;
end;


### MODIFICACIÓN 2: Envolvemos la creación de la red en una función para tener 4 distintas
funcionTransferenciaCapasConvolucionales = relu;

function crear_red_neuronal(num_arquitectura)
    if num_arquitectura == 1
        # Arquitectura 1: La original del profesor
        return Chain(
            Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales), MaxPool((2,2)),
            Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales), MaxPool((2,2)),
            Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales), MaxPool((2,2)),
            x -> reshape(x, :, size(x, 4)),
            Dense(288, length(labels)),
            softmax
        )
    elseif num_arquitectura == 2
        # Arquitectura 2: Más superficial
        return Chain(
            Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales), MaxPool((2,2)),
            Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales), MaxPool((2,2)),
            x -> reshape(x, :, size(x, 4)),
            Dense(7*7*32, length(labels)),
            softmax
        )
    elseif num_arquitectura == 3
        # Arquitectura 3: Más ancha
        return Chain(
            Conv((3, 3), 1=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales), MaxPool((2,2)),
            Conv((3, 3), 32=>64, pad=(1,1), funcionTransferenciaCapasConvolucionales), MaxPool((2,2)),
            x -> reshape(x, :, size(x, 4)),
            Dense(7*7*64, length(labels)),
            softmax
        )
    elseif num_arquitectura == 4
        # Arquitectura 4: Filtros iniciales de 5x5
        return Chain(
            Conv((5, 5), 1=>16, pad=(2,2), funcionTransferenciaCapasConvolucionales), MaxPool((2,2)),
            x -> reshape(x, :, size(x, 4)),
            Dense(14*14*16, length(labels)),
            softmax
        )
    end
end


### MODIFICACIÓN 3: Bucle de Arquitecturas y Validación Cruzada
for arquitectura_actual in 1:4
    println("\n\n==================================================")
    println("EVALUANDO ARQUITECTURA ", arquitectura_actual)
    println("==================================================")
    
    for fold in 1:5
        println("\n --- FOLD ", fold, " ---")
        
        # Obtenemos los índices para este fold
        idx_train, idx_test = obtener_indices_cv(num_patrones_totales, 5, fold, indices_aleatorios_globales)
        
        # Separamos las imágenes y etiquetas de entrenamiento y test
        train_imgs   = mis_imagenes_totales[idx_train]
        train_labels = mis_etiquetas_totales[idx_train]
        test_imgs    = mis_imagenes_totales[idx_test]
        test_labels  = mis_etiquetas_totales[idx_test]

        # Usamos la función modificada con imresize
        train_imgs_whcn = convertirArrayImagenesWHCN(train_imgs);
        test_imgs_whcn  = convertirArrayImagenesWHCN(test_imgs);

        ### MODIFICACIÓN 4: Un único batch con todo (eliminamos Iterators.partition)
        # Creamos el conjunto de entrenamiento: va a ser un vector con una sola tupla
        train_set = [ (train_imgs_whcn, oneHotEncoding(train_labels, labels)') ];

        # Creamos un batch similar, pero con todas las imagenes de test
        test_set = (test_imgs_whcn, oneHotEncoding(test_labels, labels)');

        # Liberar memoria
        train_imgs_whcn = nothing;
        test_imgs_whcn = nothing;
        GC.gc(); 

        # Definimos la red llamando a nuestra función
        ann = crear_red_neuronal(arquitectura_actual)

        @assert(size(train_set[1][2],1)>2, "RNA mal construida, para 2 clases")

        # Valores de L1 y L2 para hacer regularización 
        L1 = 0;
        L2 = 0;

        # Funciones de pérdida y precisión del profesor
        #absnorm(x) = sum(abs , x)
        #sqrnorm(x) = sum(abs2, x)
        loss(ann, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y)
        accuracy_batch(batch) = accuracy(ann(batch[1])', batch[2]');

        println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100*mean(accuracy_batch.(train_set)), " %");

        # Optimizador ADAM
        eta = 0.01;
        opt_state = Flux.setup(Adam(eta), ann);

        println("Comenzando entrenamiento...")
        
        # Variables de control reiniciadas en cada fold
        mejorPrecision = -Inf;
        criterioFin = false;
        numCiclo = 0;
        numCicloUltimaMejora = 0;
        mejorModelo = nothing;

        while !criterioFin
            # Se entrena un ciclo
            Flux.train!(loss, ann, train_set, opt_state);

            numCiclo += 1;

            # Se calcula la precision
            precisionEntrenamiento = mean(accuracy_batch.(train_set));

            # Si se mejora la precision, se calcula la de test y se guarda el modelo
            if (precisionEntrenamiento > mejorPrecision)
                mejorPrecision = precisionEntrenamiento;
                mejorModelo = deepcopy(ann);
                numCicloUltimaMejora = numCiclo;
            end

            # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
            if (numCiclo - numCicloUltimaMejora >= 5) && (eta > 1e-6)
                eta /= 10.0
                println("   No se ha mejorado en 5 ciclos, baja tasa de aprendizaje a ", eta);
                adjust!(opt_state, eta)
                numCicloUltimaMejora = numCiclo;
            end

            # Criterios de parada
            if (precisionEntrenamiento >= 0.999)
                println("   Parada: Precisión de 99.9% en ciclo $numCiclo")
                criterioFin = true;
            end

            if (numCiclo - numCicloUltimaMejora >= 10)
                println("   Parada: 10 ciclos sin mejorar en ciclo $numCiclo")
                criterioFin = true;
            end
            println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");
        end

        # Evaluación en test usando el MEJOR modelo guardado
        println("Resultados finales del Fold $fold en Test:")
        printConfusionMatrix(classifyOutputs(mejorModelo(test_set[1])'), test_set[2]');
        
    end 
end
