import numpy as np

# ====================================================
#   FUNCIONES DE APRENDIZAJE Y RECUPERACIÓN (Ritter)
# ====================================================

def aprendizaje_max(X, Y):
    """
    W[i,j] = max_mu ( Y[mu,i] - X[mu,j] )
    """
    p, n = X.shape
    m = Y.shape[1]
    W = np.full((m, n), -999, dtype=np.int32)
    for mu in range(p):
        for i in range(m):
            for j in range(n):
                W[i,j] = max(W[i,j], Y[mu,i] - X[mu,j])
    return W

def aprendizaje_min(X, Y):
    """
    M[i,j] = min_mu ( Y[mu,i] - X[mu,j] )
    """
    p, n = X.shape
    m = Y.shape[1]
    M = np.full((m, n), 999, dtype=np.int32)
    for mu in range(p):
        for i in range(m):
            for j in range(n):
                M[i,j] = min(M[i,j], Y[mu,i] - X[mu,j])
    return M

def recuperacion_max(W, x_test):
    """
    y_i = min_j ( W[i,j] + x_j )
    """
    m, n = W.shape
    y = np.zeros(m, dtype=np.int32)
    for i in range(m):
        valores = []
        for j in range(n):
            valores.append(W[i,j] + x_test[j])
        y[i] = min(valores)
    return y

def recuperacion_min(M, x_test):
    """
    y_i = max_j ( M[i,j] + x_j )
    """
    m, n = M.shape
    y = np.zeros(m, dtype=np.int32)
    for i in range(m):
        valores = []
        for j in range(n):
            valores.append(M[i,j] + x_test[j])
        y[i] = max(valores)
    return y

# ====================================================
#   EJEMPLO MÍNIMO
# ====================================================

def ejemplo_minimo():
    # Patrones de entrada (X)
    X = np.array([
        [1, 0, 0],  # Clase 0
        [0, 1, 0],  # Clase 1
        [0, 0, 1]   # Clase 2
    ], dtype=np.int32)

    # Etiquetas de salida (Y)
    Y = np.array([
        [1, 0, 0],  # Clase 0
        [0, 1, 0],  # Clase 1
        [0, 0, 1]   # Clase 2
    ], dtype=np.int32)

    print("Patrones X:\n", X)
    print("Etiquetas Y:\n", Y)

    # Aprendizaje
    W = aprendizaje_max(X, Y)
    M = aprendizaje_min(X, Y)

    print("\nMatriz W (MAX):\n", W)
    print("Matriz M (MIN):\n", M)

    # Recuperación con un patrón ruidoso
    x_test = np.array([0, 0, 1])  # mezcla de clase 0 y 1
    print("\nPatrón de prueba:", x_test)

    y_out_max = recuperacion_max(W, x_test)
    y_out_min = recuperacion_min(M, x_test)

    print("Salida MAX:", y_out_max, " -> Clase predicha:", np.argmax(y_out_max))
    print("Salida MIN:", y_out_min, " -> Clase predicha:", np.argmax(y_out_min))

# ====================================================
#   MAIN
# ====================================================

if __name__ == "__main__":
    ejemplo_minimo()
