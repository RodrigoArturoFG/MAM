import numpy as np
from PIL import Image
import os

# ====================================================
#   FUNCIONES DE APRENDIZAJE Y RECUPERACIÓN
# ====================================================

def aprendizaje_max(X, Y):
    p, n = X.shape
    m = Y.shape[1]
    W = np.full((m, n), -999, dtype=np.float32)
    for mu in range(p):
        for i in range(m):
            for j in range(n):
                W[i,j] = max(W[i,j], Y[mu,i] - X[mu,j])
    return W

def aprendizaje_min(X, Y):
    p, n = X.shape
    m = Y.shape[1]
    M = np.full((m, n), 999, dtype=np.float32)
    for mu in range(p):
        for i in range(m):
            for j in range(n):
                M[i,j] = min(M[i,j], Y[mu,i] - X[mu,j])
    return M

def recuperacion_max(W, x_test):
    m, n = W.shape
    y = np.zeros(m, dtype=np.float32)
    for i in range(m):
        valores = []
        for j in range(n):
            valores.append(W[i,j] + x_test[j])
        y[i] = min(valores)
    return y

def recuperacion_min(M, x_test):
    m, n = M.shape
    y = np.zeros(m, dtype=np.float32)
    for i in range(m):
        valores = []
        for j in range(n):
            valores.append(M[i,j] + x_test[j])
        y[i] = max(valores)
    return y

# ====================================================
#   CARGA DE IMÁGENES COMO VECTORES BINARIOS
# ====================================================

def cargar_imagen(ruta, size=50, umbral=128):
    with Image.open(ruta).convert('L') as img:
        img = img.resize((size, size))
        arr = np.array(img)
        arr_bin = (arr < umbral).astype(np.float32)
        return arr_bin.flatten()

def cargar_dataset_imagenes(rutas, size=50, umbral=128):
    return np.array([cargar_imagen(r, size, umbral) for r in rutas])

# ====================================================
#   GUARDAR RESULTADO COMO BMP
# ====================================================

def guardar_resultado_imagen(vector, size, ruta_salida, nombre_archivo):
    arr = np.array(vector, dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) * 255
    arr_img = arr.reshape((size, size))
    os.makedirs(ruta_salida, exist_ok=True)
    Image.fromarray(arr_img.astype(np.uint8)).save(os.path.join(ruta_salida, nombre_archivo))
    print(f"Imagen guardada en {os.path.join(ruta_salida, nombre_archivo)}")

# ====================================================
#   PRUEBA AUTOASOCIATIVA
# ====================================================

def prueba_pokemon_autoasociativa(rutas_l, rutas_r, clases, size=50):
    # Dataset de entrenamiento (imágenes limpias)
    X = cargar_dataset_imagenes(rutas_l, size=size, umbral=128)

    # Autoasociativa: salida deseada es el mismo X
    W = aprendizaje_max(X, X)
    M = aprendizaje_min(X, X)

    print("\nMatriz W (MAX, autoasociativa):\n", W)
    print("Matriz M (MIN, autoasociativa):\n", M)

    # Dataset con ruido
    X_r = cargar_dataset_imagenes(rutas_r, size=size, umbral=128)

    for i, x in enumerate(X_r):
        y_out_max = recuperacion_max(W, x)
        y_out_min = recuperacion_min(M, x)

        # Guardar imágenes reconstruidas
        guardar_resultado_imagen(y_out_max, size, "Resultados/Autoasociativa_Max", f"Ruido_{i}_MAX.bmp")
        guardar_resultado_imagen(y_out_min, size, "Resultados/Autoasociativa_Min", f"Ruido_{i}_MIN.bmp")

        print(f"\nRuido_{i} reconstruido -> imágenes guardadas en carpetas Autoasociativa_Max y Autoasociativa_Min")

# ====================================================
#   MAIN
# ====================================================

def main():
    nombres_pokes = ["Charmander", "Gengar", "Mewtwo", "Pikachu", "Squirtle"]
    clases_pokes = {i: nombres_pokes[i] for i in range(len(nombres_pokes))}
    rutas_entrenamiento = [f"CFP/Fase de aprendizaje/{c}.bmp" for c in nombres_pokes]
    rutas_con_ruido = [f"CFP/Patrones espurios/{c}-Ruido.bmp" for c in nombres_pokes]

    prueba_pokemon_autoasociativa(rutas_entrenamiento, rutas_con_ruido, clases_pokes)

if __name__ == "__main__":
    main()