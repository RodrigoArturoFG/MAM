import numpy as np
from PIL import Image
import os
import sys

# ====================================================
#   FUNCIONES BASE MORFOLÓGICAS (Teoría de Ritter)
# ====================================================

def aprendizaje_max(X, Y):
    p, n = X.shape
    m = Y.shape[1] if len(Y.shape) > 1 else Y.shape[0]
    W = np.full((m, n), -10, dtype=np.int32)
    for mu in range(p):
        diff = Y[mu].reshape(-1, 1) - X[mu].reshape(1, -1)
        W = np.maximum(W, diff)
    carpeta_destino="Matriz-Aprendizaje-Max"
    nombre_archivo = "Matriz_W_Max.txt"
    if not os.path.exists(carpeta_destino): os.makedirs(carpeta_destino)
    np.savetxt(os.path.join(carpeta_destino, nombre_archivo), W, fmt='%d', delimiter=',')
    print(f"Matriz W exportada como {nombre_archivo}")
    return W

def aprendizaje_min(X, Y):
    p, n = X.shape
    m = Y.shape[1] if len(Y.shape) > 1 else Y.shape[0]
    M = np.full((m, n), 10, dtype=np.int32)
    for mu in range(p):
        diff = Y[mu].reshape(-1, 1) - X[mu].reshape(1, -1)
        M = np.minimum(M, diff)
    return M

def recuperacion_max(W, x_test):
    # Producto morfológico optimizado: y = max(W + x)
    return np.max(W + x_test.reshape(1, -1), axis=1)

def recuperacion_min(M, x_test):
    # Producto morfológico optimizado: y = min(M + x)
    return np.min(M + x_test.reshape(1, -1), axis=1)

"""
def aprendizaje_max(X, Y):
    # Construcción de la matriz W usando suma-máxima (Supremo)
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
    # Construcción de la matriz M usando suma-mínima (Ínfimo)
    p, n = X.shape
    m = Y.shape[1]
    # Inicializamos con un valor neutro superior
    M = np.full((m, n), 10, dtype=np.int32)
    for mu in range(p):
        diff = Y[mu].reshape(-1, 1) - X[mu].reshape(1, -1)
        M = np.minimum(M, diff)
    return M
"""

"""
def recuperacion_max(W, x_test):
    # Inferencia Morfológica Superior: y = W ⊠ x
    m = W.shape[0]
    y_salida = np.zeros(m, dtype=np.int32)
    x_test = x_test.flatten().astype(np.int32)
    for i in range(m):
        y_salida[i] = np.max(W[i, :] + x_test)
    return y_salida

def recuperacion_min(M, x_test):
    # Inferencia Morfológica Inferior: y = M ⊞ x
    m = M.shape[0]
    y_salida = np.zeros(m, dtype=np.int32)
    x_test = x_test.flatten().astype(np.int32)
    for i in range(m):
        y_salida[i] = np.min(M[i, :] + x_test)
    return y_salida
"""

# ====================================================
#   CARGA, GUARDADO Y DIAGNÓSTICO
# ====================================================

def cargar_imagenes_a_matriz(rutas_imagenes):
    datos = []
    shape_original = None
    for ruta in rutas_imagenes:
        with Image.open(ruta).convert('L') as img:
            arr = np.array(img)
            # BIPOLAR: Pokémon (Negro) -> 1, Fondo (Blanco) -> -1
            # Esto es lo que recomienda Ritter para patrones muy parecidos
            arr_bipolar = np.where(arr < 128, 1, -1).astype(np.int32)
            if shape_original is None: shape_original = arr_bipolar.shape
            datos.append(arr_bipolar.flatten())
    return np.array(datos), shape_original


def guardar_resultado_morfo(y_vector, shape_original, nombre_archivo, carpeta_destino):
    if not os.path.exists(carpeta_destino): os.makedirs(carpeta_destino)
    res = y_vector.reshape(shape_original)
    
    # Lógica de decisión bipolar:
    # Si el valor recuperado es positivo, el píxel pertenece al Pokémon.
    res_img = np.where(res > 0, 0, 255).astype(np.uint8)
    
    ruta_final = os.path.join(carpeta_destino, nombre_archivo).replace("\\", "/")
    Image.fromarray(res_img).save(ruta_final)
    print(f"Imagen guardada: {ruta_final}")

"""
def guardar_resultado_morfo(y_vector, shape_original, nombre_archivo, carpeta_destino):
    # Reconstruye la imagen (1 -> Negro, 0 -> Blanco)
    if not os.path.exists(carpeta_destino): os.makedirs(carpeta_destino)
    res = y_vector.reshape(shape_original)
    # Invertimos la binarización para el reporte visual (Fondo Blanco 255)
    res_img = np.where(res >= 1, 0, 255).astype(np.uint8)
    Image.fromarray(res_img).save(os.path.join(carpeta_destino, nombre_archivo))
"""

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
    X, _ = cargar_imagenes_a_matriz(rutas_l)
    # ETIQUETAS BIPOLARES: 1 para la clase, -1 para las demás
    Y = (np.eye(len(clases), dtype=np.int32) * 2) - 1
    W = aprendizaje_max(X, Y)
    X_r, _ = cargar_imagenes_a_matriz(rutas_r)
    
    for i, x in enumerate(X_r):
        y_r = recuperacion_max(W, x)
        print(f"DEBUG MAX - Vector: {y_r}")
        print(f"Ruido_{i} -> {clases[np.argmax(y_r)]}")

def ejecutar_heteroasociativa_min(rutas_l, rutas_r, clases):
    X, _ = cargar_imagenes_a_matriz(rutas_l)
    # ETIQUETAS BIPOLARES PARA MIN: -1 para la clase, 1 para las demás
    Y = 1 - (np.eye(len(clases), dtype=np.int32) * 2)
    M = aprendizaje_min(X, Y)
    X_r, _ = cargar_imagenes_a_matriz(rutas_r)
    
    for i, x in enumerate(X_r):
        y_r = recuperacion_min(M, x)
        print(f"DEBUG MIN - Vector: {y_r}")
        # En MIN, el valor más PEQUEÑO identifica la clase
        print(f"Ruido_{i} -> {clases[np.argmin(y_r)]}")

def ejecutar_autoasociativa_max(rutas_l, rutas_r):
    """Restauración: Robusta a ruido sustractivo (Puntos negros)"""
    print("\n--- INICIANDO MAM AUTOASOCIATIVA MAX (Bipolar) ---")
    
    # 1. Carga Bipolar (Pokémon 1, Fondo -1)
    X, shape = cargar_imagenes_a_matriz(rutas_l)
    X_r, _ = cargar_imagenes_a_matriz(rutas_r)

    # 2. Aprendizaje Autoasociativo (Y = X)
    # Matriz cuadrada de (2500, 2500)
    W_auto = aprendizaje_max(X, X)

    # 3. Recuperación Matricial Optimizada
    os.makedirs("Resultados/Autoasociativa_Max", exist_ok=True)
    for i, x_con_ruido in enumerate(X_r):
        # Producto morfológico veloz: max(W + x)
        y_r = recuperacion_max(W_auto, x_con_ruido)
        
        # 4. RECONSTRUCCIÓN VISUAL:
        # En lógica bipolar, valores > 0 son Pokémon (Negro: 0)
        # Valores <= 0 son Fondo (Blanco: 255)
        guardar_resultado_morfo(
            y_r, 
            shape, 
            f"restaurada_MAX_img_{i}.png", 
            "Resultados/Autoasociativa_Max"
        )
    print("Evidencias guardadas en: Resultados/Autoasociativa_Max")

def ejecutar_autoasociativa_min(rutas_l, rutas_r):
    """Limpieza: Robusta a ruido aditivo (Manchas blancas)"""
    print("\n--- INICIANDO MAM AUTOASOCIATIVA MIN (Bipolar) ---")
    
    # 1. Carga Bipolar
    X, shape = cargar_imagenes_a_matriz(rutas_l)
    X_r, _ = cargar_imagenes_a_matriz(rutas_r)

    # 2. Aprendizaje Autoasociativo (Y = X)
    M_auto = aprendizaje_min(X, X)

    # 3. Recuperación Matricial Optimizada
    os.makedirs("Resultados/Autoasociativa_Min", exist_ok=True)
    for i, x_con_ruido in enumerate(X_r):
        # Producto morfológico veloz: min(M + x)
        y_r = recuperacion_min(M_auto, x_con_ruido)
        
        # Reconstrucción visual idéntica
        guardar_resultado_morfo(
            y_r, 
            shape, 
            f"restaurada_MIN_img_{i}.png", 
            "Resultados/Autoasociativa_Min"
        )
    print("Evidencias guardadas en: Resultados/Autoasociativa_Min")
    
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
    nombres_pokes = ["Charmander", "Gengar", "Mewtwo", "Pikachu", "Squirtle"]
    clases_pokes = {
    0: np.array([255, 0, 0, 0, 0]), # Pokemon 1
    1: np.array([0, 255, 0, 0, 0]), # Pokemon 2
    2: np.array([0, 0, 255, 0, 0]), # Pokemon 3
    3: np.array([0, 0, 0, 255, 0]), # Pokemon 4
    4: np.array([0, 0, 0, 0, 255])  # Pokemon 5
    }
    # Rutas relativas basadas en tu repo
    rutas_entrenamiento = [f"CFP/Fase de aprendizaje/{c}.bmp" for c in nombres_pokes]
    rutas_con_ruido = [f"CFP/Patrones espurios/{c}-Ruido.bmp" for c in nombres_pokes]

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