import numpy as np
from PIL import Image
from scipy.ndimage import label
import cv2
import csv

# ====================================================
#   FUNCIONES DE APRENDIZAJE Y RECUPERACIÓN (Ritter)
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
#   EXTRACCIÓN DE CARACTERÍSTICAS
# ====================================================

def extraer_caracteristicas(ruta, size=50, umbral=128):
    with Image.open(ruta).convert('L') as img:
        img = img.resize((size, size))
        arr = np.array(img)
        arr_bin = (arr < umbral).astype(np.uint8)

        # 1. Proporción de píxeles negros
        proporción = np.sum(arr_bin) / arr_bin.size

        # 2. Centroide (x,y)
        coords = np.argwhere(arr_bin == 1)
        if len(coords) > 0:
            cx, cy = coords[:,1].mean() / size, coords[:,0].mean() / size
        else:
            cx, cy = 0, 0

        # 3. Distribución horizontal y vertical (media)
        distrib_h = np.sum(arr_bin, axis=1) / size
        distrib_v = np.sum(arr_bin, axis=0) / size
        dh_mean = distrib_h.mean()
        dv_mean = distrib_v.mean()

        # 4. Número de componentes conectados
        labeled, num_comp = label(arr_bin)

        # 5. Momentos de Hu
        moments = cv2.moments(arr_bin)
        hu = cv2.HuMoments(moments).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-9)

        # 6. Cuadrantes 3x3
        h, w = arr_bin.shape
        q_features = []
        for i in range(3):
            for j in range(3):
                block = arr_bin[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
                q_features.append(np.sum(block) / block.size)

        # 7. Densidad de bordes (Sobel)
        sobelx = cv2.Sobel(arr_bin, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(arr_bin, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edge_density = np.sum(edges > 0) / edges.size

        # 8. HOG (Histogram of Oriented Gradients)
        hog = cv2.HOGDescriptor()
        hog_features = hog.compute(arr_bin)
        # Reducimos dimensionalidad tomando la media de bloques
        hog_mean = np.mean(hog_features)

        return np.concatenate(([proporción, cx, cy, dh_mean, dv_mean, num_comp, edge_density, hog_mean], q_features, hu_log))

def cargar_dataset(rutas, size=50, umbral=128):
    return np.array([extraer_caracteristicas(r, size, umbral) for r in rutas])

# ====================================================
#   ETIQUETAS BIPOLARES
# ====================================================

def construir_etiquetas_bipolares(num_patrones, num_clases):
    Y = -np.ones((num_patrones, num_clases), dtype=np.float32)
    for i in range(num_patrones):
        clase = i % num_clases
        Y[i, clase] = 1
    return Y

import os
from PIL import Image

# ====================================================
import os
import numpy as np
from PIL import Image
import cv2

# ====================================================
#   GUARDAR RESULTADO COMO CSV
# ====================================================

def guardar_resultado_csv(vector, ruta_salida, nombre_archivo):
    """
    Guarda el vector recuperado en un archivo CSV.
    """
    os.makedirs(ruta_salida, exist_ok=True)
    np.savetxt(os.path.join(ruta_salida, nombre_archivo), vector, delimiter=",")
    print(f"Vector guardado en {os.path.join(ruta_salida, nombre_archivo)}")

# ====================================================
#   GUARDAR RESULTADO COMO HEATMAP BMP
# ====================================================

def guardar_resultado_heatmap(vector, size, ruta_salida, nombre_archivo):
    """
    Convierte un vector de características en un mapa de calor BMP.
    - vector: salida recuperada (array)
    - size: tamaño final de la imagen cuadrada (ej. 50x50)
    - ruta_salida: carpeta donde guardar
    - nombre_archivo: nombre del archivo BMP
    """
    arr = np.array(vector, dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) * 255

    # Ajustar a cuadrícula cuadrada
    side = int(np.ceil(np.sqrt(len(arr))))
    arr_pad = np.pad(arr, (0, side*side - len(arr)), mode='constant')
    arr_img = arr_pad.reshape((side, side))

    # Escalar a tamaño deseado
    arr_img = cv2.resize(arr_img.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)

    os.makedirs(ruta_salida, exist_ok=True)
    Image.fromarray(arr_img).save(os.path.join(ruta_salida, nombre_archivo))
    print(f"Imagen guardada en {os.path.join(ruta_salida, nombre_archivo)}")

# ====================================================
#   EXPORTACIÓN A CSV
# ====================================================

def exportar_csv(X, nombres, archivo="caracteristicas.csv"):
    with open(archivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Proporción","Cx","Cy","DH_mean","DV_mean","Num_comp","EdgeDensity","HOG_mean"] \
                 + [f"Q{i+1}" for i in range(9)] \
                 + [f"Hu{i+1}" for i in range(7)]
        writer.writerow(["Clase"] + header)
        for i, fila in enumerate(X):
            writer.writerow([nombres[i]] + fila.tolist())
    print(f"Características exportadas a {archivo}")

# ====================================================
#   PRUEBA HETEROASOCIATIVA 
# ====================================================

def prueba_pokemon_heteroasociativa(rutas_l, rutas_r, clases):
    X = cargar_dataset(rutas_l, size=50, umbral=128)
    Y = construir_etiquetas_bipolares(len(X), len(clases))

    exportar_csv(X, list(clases.values()), "caracteristicas.csv")

    W = aprendizaje_max(X, Y)
    M = aprendizaje_min(X, Y)

    print("\nMatriz W (MAX):\n", W)
    print("Matriz M (MIN):\n", M)

    X_r = cargar_dataset(rutas_r, size=50, umbral=128)
    for i, x in enumerate(X_r):
        y_out_max = recuperacion_max(W, x)
        y_out_min = recuperacion_min(M, x)
        clase_pred_max = np.argmax(y_out_max)
        clase_pred_min = np.argmax(y_out_min)

        if clase_pred_max == clase_pred_min:
            clase_final = clase_pred_max
            fuente = "Consenso"
        else:
            clase_final = clase_pred_max
            fuente = "MAX (fallback)"

        print(f"\nRuido_{i} -> MAX: {clases[clase_pred_max]} | MIN: {clases[clase_pred_min]} | FINAL ({fuente}): {clases[clase_final]}")
        print(f"Vector salida MAX: {y_out_max}")
        print(f"Vector salida MIN: {y_out_min}")

# ====================================================
#   PRUEBA AUTOASOCIATIVA
# ====================================================

def prueba_pokemon_autoasociativa(rutas_l, rutas_r, clases, size=50):
    X = cargar_dataset(rutas_l, size=size, umbral=128)
    W = aprendizaje_max(X, X)
    M = aprendizaje_min(X, X)

    print("\nMatriz W (MAX, autoasociativa):\n", W)
    print("Matriz M (MIN, autoasociativa):\n", M)

    X_r = cargar_dataset(rutas_r, size=size, umbral=128)

    for i, x in enumerate(X_r):
        y_out_max = recuperacion_max(W, x)
        y_out_min = recuperacion_min(M, x)

        # Guardar vectores recuperados
        # guardar_resultado_csv(y_out_max, "Resultados/Autoasociativa_Max", f"restaurada_MAX_img_{i}.csv")
        # guardar_resultado_csv(y_out_min, "Resultados/Autoasociativa_Min", f"restaurada_MIN_img_{i}.csv")

        # Guardar visualización como heatmap BMP
        # guardar_resultado_heatmap(y_out_max, size, "Resultados/Autoasociativa_Max", f"restaurada_MAX_img_{i}.bmp")
        # guardar_resultado_heatmap(y_out_min, size, "Resultados/Autoasociativa_Min", f"restaurada_MIN_img_{i}.bmp")

        # Comparar con originales
        distancias_max = [np.linalg.norm(y_out_max - x_ref) for x_ref in X]
        distancias_min = [np.linalg.norm(y_out_min - x_ref) for x_ref in X]

        clase_pred_max = np.argmin(distancias_max)
        clase_pred_min = np.argmin(distancias_min)

        if clase_pred_max == clase_pred_min:
            clase_final = clase_pred_max
            fuente = "Consenso"
        else:
            clase_final = clase_pred_max
            fuente = "MAX (fallback)"

        print(f"\nRuido_{i} -> MAX: {clases[clase_pred_max]} | MIN: {clases[clase_pred_min]} | FINAL ({fuente}): {clases[clase_final]}")


# ====================================================
#   MAIN
# ====================================================

def main():
    nombres_pokes = ["Charmander", "Gengar", "Mewtwo", "Pikachu", "Squirtle"]
    clases_pokes = {i: nombres_pokes[i] for i in range(len(nombres_pokes))}
    rutas_entrenamiento = [f"CFP/Fase de aprendizaje/{c}.bmp" for c in nombres_pokes]
    rutas_con_ruido = [f"CFP/Patrones espurios/{c}-Ruido.bmp" for c in nombres_pokes]

    prueba_pokemon_heteroasociativa(rutas_entrenamiento, rutas_con_ruido, clases_pokes)
    #prueba_pokemon_autoasociativa(rutas_entrenamiento, rutas_con_ruido, clases_pokes)

if __name__ == "__main__":
    main()
