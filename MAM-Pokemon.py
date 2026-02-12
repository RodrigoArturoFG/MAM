import numpy as np
from PIL import Image
import os
import sys

# ====================================================
#   FUNCIONES BASE MORFOLÓGICAS (Teoría de Ritter)
# ====================================================

def aprendizaje_max(X, Y):
    """Construcción de la matriz W usando suma-máxima (Supremo)"""
    p, n = X.shape
    m = Y.shape[1]
    # Inicializamos con un valor neutro inferior (épsilon)
    W = np.full((m, n), -10, dtype=np.int32)
    for mu in range(p):
        # Operación: y_i - x_j
        diff = Y[mu].reshape(-1, 1) - X[mu].reshape(1, -1)
        W = np.maximum(W, diff)
    return W

def aprendizaje_min(X, Y):
    """Construcción de la matriz M usando suma-mínima (Ínfimo)"""
    p, n = X.shape
    m = Y.shape[1]
    # Inicializamos con un valor neutro superior
    M = np.full((m, n), 10, dtype=np.int32)
    for mu in range(p):
        diff = Y[mu].reshape(-1, 1) - X[mu].reshape(1, -1)
        M = np.minimum(M, diff)
    return M

def recuperacion_max(W, x_test):
    """Inferencia Morfológica Superior: y = W ⊠ x"""
    m = W.shape[0]
    y_salida = np.zeros(m, dtype=np.int32)
    x_test = x_test.flatten().astype(np.int32)
    for i in range(m):
        y_salida[i] = np.max(W[i, :] + x_test)
    return y_salida

def recuperacion_min(M, x_test):
    """Inferencia Morfológica Inferior: y = M ⊞ x"""
    m = M.shape[0]
    y_salida = np.zeros(m, dtype=np.int32)
    x_test = x_test.flatten().astype(np.int32)
    for i in range(m):
        y_salida[i] = np.min(M[i, :] + x_test)
    return y_salida

# ====================================================
#   CARGA, GUARDADO Y DIAGNÓSTICO
# ====================================================

def cargar_imagenes_a_matriz(rutas_imagenes):
    """Carga imágenes BMP y las binariza (Pokémon=1, Fondo=0)"""
    datos = []
    shape_original = None
    for ruta in rutas_imagenes:
        if not os.path.exists(ruta):
            print(f"Error: No existe {ruta}")
            continue
        with Image.open(ruta).convert('L') as img:
            arr = np.array(img)
            # TRANSFORMACIÓN: Silueta (Negro < 128) -> 1, Fondo (Blanco >= 128) -> 0
            # Según la literatura, el objeto de interés debe ser el valor alto (señal)
            arr_bin = np.where(arr < 128, 1, 0).astype(np.int32)
            if shape_original is None: shape_original = arr_bin.shape
            datos.append(arr_bin.flatten())
    return np.array(datos), shape_original

def guardar_resultado_morfo(y_vector, shape_original, nombre_archivo, carpeta_destino):
    """Reconstruye la imagen (1 -> Negro, 0 -> Blanco)"""
    if not os.path.exists(carpeta_destino): os.makedirs(carpeta_destino)
    res = y_vector.reshape(shape_original)
    # Invertimos la binarización para el reporte visual (Fondo Blanco 255)
    res_img = np.where(res >= 1, 0, 255).astype(np.uint8)
    Image.fromarray(res_img).save(os.path.join(carpeta_destino, nombre_archivo))

def guardar_matrices_a_texto(rutas_imagenes, carpeta_destino="Imagenes-Matriz"):
    """Diagnóstico de 0s y 1s"""
    if not os.path.exists(carpeta_destino): os.makedirs(carpeta_destino)
    for ruta in rutas_imagenes:
        with Image.open(ruta).convert('L') as img:
            arr = np.array(img)
            arr_bin = np.where(arr < 128, 1, 0).astype(np.int32)
            nombre = os.path.basename(ruta).replace(".bmp", "")
            np.savetxt(os.path.join(carpeta_destino, f"{nombre}_bin.txt"), arr_bin, fmt='%d', delimiter=',')
            print(f"Matriz {nombre} exportada como 0/1.")

# ====================================================
#   PROCESOS DE EJECUCIÓN
# ====================================================

def ejecutar_heteroasociativa_max(rutas_l, rutas_r, clases):
    print("\n--- MAM HETEROASOCIATIVA MAX (Clasificación) ---")
    X, _ = cargar_imagenes_a_matriz(rutas_l)
    Y = np.eye(len(clases), dtype=np.int32) # One-Hot: 1 para la clase, 0 resto
    W = aprendizaje_max(X, Y)
    X_r, _ = cargar_imagenes_a_matriz(rutas_r)
    
    print(f"{'Imagen':<12} | {'Predicción MAX'}")
    print("-" * 35)
    for i, x in enumerate(X_r):
        y_r = recuperacion_max(W, x)
        idx = np.argmax(y_r)
        print(f"DEBUG MAX - Vector: {y_r}")
        print(f"Ruido_{i:<7} | {clases[idx]}")

def ejecutar_heteroasociativa_min(rutas_l, rutas_r, clases):
    print("\n--- MAM HETEROASOCIATIVA MIN (Clasificación) ---")
    X, _ = cargar_imagenes_a_matriz(rutas_l)
    # Para MIN, usamos etiquetas invertidas si es necesario, 
    # pero el One-Hot estándar funciona con argmin/argmax según el contraste.
    Y = np.eye(len(clases), dtype=np.int32)
    M = aprendizaje_min(X, Y)
    X_r, _ = cargar_imagenes_a_matriz(rutas_r)
    
    print(f"{'Imagen':<12} | {'Predicción MIN'}")
    print("-" * 35)
    for i, x in enumerate(X_r):
        y_r = recuperacion_min(M, x)
        idx = np.argmax(y_r) # Usamos argmax por el contraste 0/1
        print(f"DEBUG MIN - Vector: {y_r}")
        print(f"Ruido_{i:<7} | {clases[idx]}")

def ejecutar_autoasociativa_max(rutas_l, rutas_r):
    print("\n--- MAM AUTOASOCIATIVA MAX (Restauración) ---")
    X, shape = cargar_imagenes_a_matriz(rutas_l)
    W = aprendizaje_max(X, X)
    X_r, _ = cargar_imagenes_a_matriz(rutas_r)
    for i, x in enumerate(X_r):
        y_r = recuperacion_max(W, x)
        guardar_resultado_morfo(y_r, shape, f"MAX_{i}.png", "Resultados/Autoasociativa_Max")
    print("Resultados en Resultados/Autoasociativa_Max")

def ejecutar_autoasociativa_min(rutas_l, rutas_r):
    print("\n--- MAM AUTOASOCIATIVA MIN (Limpieza) ---")
    X, shape = cargar_imagenes_a_matriz(rutas_l)
    M = aprendizaje_min(X, X)
    X_r, _ = cargar_imagenes_a_matriz(rutas_r)
    for i, x in enumerate(X_r):
        y_r = recuperacion_min(M, x)
        guardar_resultado_morfo(y_r, shape, f"MIN_{i}.png", "Resultados/Autoasociativa_Min")
    print("Resultados en Resultados/Autoasociativa_Min")
    
# ====================================================
#   INTERFAZ DE USUARIO Y MENÚ
# ====================================================

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

def mostrar_menu():
    limpiar_pantalla()
    print("====================================================")
    print("   PROYECTO RECONOCIMIENTO DE IMÁGENES - MAM (IPN)  ")
    print("====================================================")
    print("0. Exportar Matrices de Diagnóstico (0 y 1 a .txt)")
    print("1. MAM Heteroasociativa MAX (Clasificación)")
    print("2. MAM Heteroasociativa MIN (Clasificación)")
    print("3. MAM Autoasociativa MAX (Restauración)")
    print("4. MAM Autoasociativa MIN (Limpieza)")
    print("5. Salir del programa")
    print("====================================================")

def main():
    clases_pokes = ["Charmander", "Gengar", "Mewtwo", "Pikachu", "Squirtle"]
    # Rutas relativas basadas en tu repo
    rutas_entrenamiento = [f"CFP/Fase de aprendizaje/{c}.bmp" for c in clases_pokes]
    rutas_con_ruido = [f"CFP/Patrones espurios/{c}-Ruido.bmp" for c in clases_pokes]

    while True:
        mostrar_menu()
        opcion = input("Seleccione una opción (0-5): ")

        if opcion == "0":
            guardar_matrices_a_texto(rutas_entrenamiento)
            input("\nPresione Enter para volver...")
        elif opcion == "1":
            ejecutar_heteroasociativa_max(rutas_entrenamiento, rutas_con_ruido, clases_pokes)
            input("\nPresione Enter para volver...")
        elif opcion == "2":
            ejecutar_heteroasociativa_min(rutas_entrenamiento, rutas_con_ruido, clases_pokes)
            input("\nPresione Enter para volver...")
        elif opcion == "3":
            ejecutar_autoasociativa_max(rutas_entrenamiento, rutas_con_ruido)
            input("\nPresione Enter para volver...")
        elif opcion == "4":
            ejecutar_autoasociativa_min(rutas_entrenamiento, rutas_con_ruido)
            input("\nPresione Enter para volver...")
        elif opcion == "5":
            print("\nSaliendo del programa...")
            sys.exit()

if __name__ == "__main__":
    main()
