import numpy as np
from PIL import Image
import os
import sys

# ====================================================
#   FUNCIONES BASE MORFOLÓGICAS (Bipolar 1/-1)
# ====================================================

def aprendizaje_max(X, Y):
    p, n = X.shape
    m = Y.shape[1]
    W = np.full((m, n), -10, dtype=np.int32)
    for mu in range(p):
        diff = Y[mu].reshape(-1, 1) - X[mu].reshape(1, -1)
        W = np.maximum(W, diff)
    return W

def aprendizaje_min(X, Y):
    p, n = X.shape
    m = Y.shape[1]
    M = np.full((m, n), 10, dtype=np.int32)
    for mu in range(p):
        diff = Y[mu].reshape(-1, 1) - X[mu].reshape(1, -1)
        M = np.minimum(M, diff)
    return M

def recuperacion_max(W, x_test):
    return np.max(W + x_test.reshape(1, -1), axis=1)

def recuperacion_min(M, x_test):
    return np.min(M + x_test.reshape(1, -1), axis=1)

# ====================================================
#   CARGA Y GUARDADO DE IMÁGENES (Bipolar)
# ====================================================

def cargar_imagenes_a_matriz_bipolar(rutas_imagenes):
    datos = []
    shape_original = None
    for ruta in rutas_imagenes:
        with Image.open(ruta).convert('L') as img:
            img = img.resize((50, 50))
            arr = np.array(img)
            arr_bipolar = np.where(arr < 128, 1, -1).astype(np.int32)
            if shape_original is None: shape_original = arr_bipolar.shape
            datos.append(arr_bipolar.flatten())
    return np.array(datos), shape_original

def guardar_resultado_morfo(y_vector, shape_original, nombre_archivo, carpeta_destino):
    if not os.path.exists(carpeta_destino): os.makedirs(carpeta_destino)
    res = y_vector.reshape(shape_original)
    res_img = np.where(res > 0, 0, 255).astype(np.uint8)  # >0 negro, <=0 blanco
    ruta_final = os.path.join(carpeta_destino, nombre_archivo).replace("\\", "/")
    Image.fromarray(res_img).save(ruta_final)
    print(f"Imagen guardada: {ruta_final}")

# ====================================================
#   ETIQUETAS BIPOLARES POR PATRÓN
# ====================================================

def construir_etiquetas_bipolares(num_patrones, num_clases):
    Y = -np.ones((num_patrones, num_clases), dtype=np.int32)
    for i in range(num_patrones):
        clase = i % num_clases
        Y[i, clase] = 1
    return Y

# ====================================================
#   AUTOASOCIATIVA BIPOLAR
# ====================================================

def ejecutar_autoasociativa_max(rutas_l, rutas_r):
    print("\n--- AUTOASOCIATIVA MAX (Bipolar) ---")
    X, shape = cargar_imagenes_a_matriz_bipolar(rutas_l)
    X_r, _ = cargar_imagenes_a_matriz_bipolar(rutas_r)
    W_auto = aprendizaje_max(X, X)
    os.makedirs("Resultados/Autoasociativa_Max", exist_ok=True)
    for i, x_con_ruido in enumerate(X_r):
        y_r = recuperacion_max(W_auto, x_con_ruido)
        guardar_resultado_morfo(y_r, shape, f"restaurada_MAX_img_{i}.png", "Resultados/Autoasociativa_Max")

def ejecutar_autoasociativa_min(rutas_l, rutas_r):
    print("\n--- AUTOASOCIATIVA MIN (Bipolar) ---")
    X, shape = cargar_imagenes_a_matriz_bipolar(rutas_l)
    X_r, _ = cargar_imagenes_a_matriz_bipolar(rutas_r)
    M_auto = aprendizaje_min(X, X)
    os.makedirs("Resultados/Autoasociativa_Min", exist_ok=True)
    for i, x_con_ruido in enumerate(X_r):
        y_r = recuperacion_min(M_auto, x_con_ruido)
        guardar_resultado_morfo(y_r, shape, f"restaurada_MIN_img_{i}.png", "Resultados/Autoasociativa_Min")

# ====================================================
#   HETEROASOCIATIVA BIPOLAR
# ====================================================

def ejecutar_heteroasociativa_max(rutas_l, rutas_r, clases):
    print("\n--- HETEROASOCIATIVA MAX (Bipolar) ---")
    X, _ = cargar_imagenes_a_matriz_bipolar(rutas_l)
    Y = construir_etiquetas_bipolares(len(X), len(clases))
    W = aprendizaje_max(X, Y)
    X_r, _ = cargar_imagenes_a_matriz_bipolar(rutas_r)
    for i, x in enumerate(X_r):
        y_r = recuperacion_max(W, x)
        clase_pred = np.argmax(y_r)
        print(f"Ruido_{i} -> Clase predicha: {clases[clase_pred]} | Vector salida: {y_r}")

def ejecutar_heteroasociativa_min(rutas_l, rutas_r, clases):
    print("\n--- HETEROASOCIATIVA MIN (Bipolar) ---")
    X, _ = cargar_imagenes_a_matriz_bipolar(rutas_l)
    Y = -construir_etiquetas_bipolares(len(X), len(clases))  # invertido
    M = aprendizaje_min(X, Y)
    X_r, _ = cargar_imagenes_a_matriz_bipolar(rutas_r)
    for i, x in enumerate(X_r):
        y_r = recuperacion_min(M, x)
        clase_pred = np.argmin(y_r)
        print(f"Ruido_{i} -> Clase predicha: {clases[clase_pred]} | Vector salida: {y_r}")

# ====================================================
#   MENÚ PRINCIPAL
# ====================================================

def mostrar_menu():
    print("====================================================")
    print("   PROYECTO RECONOCIMIENTO DE IMÁGENES - MAM (Bipolar)")
    print("====================================================")
    print("1. Heteroasociativa MAX (Clasificación)")
    print("2. Heteroasociativa MIN (Clasificación)")
    print("3. Autoasociativa MAX (Restauración)")
    print("4. Autoasociativa MIN (Limpieza)")
    print("5. Salir")
    print("====================================================")

def main():
    nombres_pokes = ["Charmander", "Gengar", "Mewtwo", "Pikachu", "Squirtle"]
    clases_pokes = {i: nombres_pokes[i] for i in range(len(nombres_pokes))}
    rutas_entrenamiento = [f"CFP/Fase de aprendizaje/{c}.bmp" for c in nombres_pokes]
    rutas_con_ruido = [f"CFP/Patrones espurios/{c}-Ruido.bmp" for c in nombres_pokes]

    while True:
        mostrar_menu()
        opcion = input("Seleccione una opción (1-5): ")

        if opcion == "1":
            ejecutar_heteroasociativa_max(rutas_entrenamiento, rutas_con_ruido, clases_pokes)
        elif opcion == "2":
            ejecutar_heteroasociativa_min(rutas_entrenamiento, rutas_con_ruido, clases_pokes)
        elif opcion == "3":
            ejecutar_autoasociativa_max(rutas_entrenamiento, rutas_con_ruido)
        elif opcion == "4":
            ejecutar_autoasociativa_min(rutas_entrenamiento, rutas_con_ruido)
        elif opcion == "5":
            print("Saliendo del programa...")
            sys.exit()
        else:
            print("Opción inválida, intente de nuevo.")

if __name__ == "__main__":
    main()
