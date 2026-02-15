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