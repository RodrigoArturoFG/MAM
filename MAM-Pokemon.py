import numpy as np
from PIL import Image
import os
import sys

# Preprocesamiento de imagenes
def cargar_imagenes_a_matriz(rutas_imagenes):
    """
    Carga imágenes desde una lista de rutas, las convierte a escala de grises,
    las aplana (vector) y guarda sus dimensiones originales.
    """
    datos_aplanados = []
    shape_original = None
    
    for ruta in rutas_imagenes:
        with Image.open(ruta).convert('L') as img:
            arr = np.array(img)
            if shape_original is None:
                shape_original = arr.shape  # Guarda (alto, ancho) de la primera imagen
            datos_aplanados.append(arr.flatten())
            
    return np.array(datos_aplanados), shape_original

# Memoría asociativa morfológica de tipo Max
def aprendizaje_max(X, Y):
    """
    X: Matriz de entrada (p patrones x n pixeles)
    Y: Matriz de salida (p patrones x m pixeles o etiquetas)
    """
    num_patrones, n = X.shape
    m = Y.shape[1] # Cantidad de elementos en la salida
    W = np.full((m, n), -np.inf) # Inicializamos con valor muy bajo para el Max
    
    # Optimización con Numpy para evitar el doble for lento:
    for mu in range(num_patrones):
        # Para cada patrón, calculamos la matriz de diferencias (Y - X)
        # y actualizamos W con el máximo actual
        diferencia = Y[mu].reshape(-1, 1) - X[mu].reshape(1, -1)
        W = np.maximum(W, diferencia)
        
    return W

def recuperacion_max(W, x_test):
    """
    W: Matriz de memoria
    x_test: Vector de imagen con ruido (aplanado)
    """
    m, n = W.shape
    y_salida = np.zeros(m)
    for i in range(m):
        # Operación fundamental: Max(W_ij + x_j)
        y_salida[i] = np.max(W[i, :] + x_test)
    return y_salida

# Memoría asociativa morfológica de tipo Min
def aprendizaje_min(X, Y):
    """
    X: Matriz de entrada (p patrones x n pixeles)
    Y: Matriz de salida (p patrones x m pixeles o etiquetas)
    """
    num_patrones, n = X.shape
    m = Y.shape[1] # Cantidad de elementos en la salida
    
    # Inicializamos con infinito positivo (np.inf) porque buscaremos el MIN
    M = np.full((m, n), np.inf) 
    
    # Optimización con Numpy (idéntico al procedimiento de la Max):
    for mu in range(num_patrones):
        # Para cada patrón, calculamos la matriz de diferencias (Y - X)
        # y actualizamos M con el mínimo actual
        diferencia = Y[mu].reshape(-1, 1) - X[mu].reshape(1, -1)
        M = np.minimum(M, diferencia)
        
    return M

def recuperacion_min(M, x_test):
    """
    M: Matriz de memoria generada en el aprendizaje
    x_test: Vector de imagen con ruido (aplanado)
    """
    m, n = M.shape
    y_salida = np.zeros(m)
    for i in range(m):
        # Operación fundamental: Min(M_ij + x_j)
        y_salida[i] = np.min(M[i, :] + x_test)
        
    return y_salida

# Función para guardar resultados morfológicos como imágenes
def guardar_resultado_morfo(y_vector, shape_original, nombre_archivo, carpeta_destino="resultados"):
    """
    Convierte un vector de salida de la MAM en imagen y la guarda en una ruta relativa.
    
    y_vector: El vector (1D) resultante de la fase de recuperación.
    shape_original: Tupla (alto, ancho) de la imagen, ej: (50, 50).
    nombre_archivo: Nombre del archivo de salida (ej: "pikachu_recuperado.png").
    carpeta_destino: Carpeta donde se guardará el resultado.
    """
    
    # 1. Crear la carpeta si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    """
    # Variante para reconstruir la imagen normalizando el vector (opcional, dependiendo de la escala de valores):
    # Normalizamos el vector al rango [0, 255] para visualizarlo como imagen
    vector_normalizado = 255 * (y_vector - np.min(y_vector)) / (np.ptp(y_vector) + 1e-5)
    imagen_array = vector_normalizado.reshape(shape_original).astype(np.uint8)
    """
    
    # En este caso, asumimos que el vector ya está en el rango adecuado (0-255) después de las operaciones morfológicas.
    # 2. Reconstruir la forma de la imagen (Reshape de 1D a 2D)
    # Importante: Aplicamos clip(0, 255) por si las sumas morfológicas 
    # generaron valores fuera del rango de color estándar.
    imagen_array = y_vector.reshape(shape_original)
    imagen_array = np.clip(imagen_array, 0, 255).astype(np.uint8)
    
    # 3. Crear el objeto de imagen desde el array de Numpy
    img_resultante = Image.fromarray(imagen_array)
    
    # 4. Guardamos la imagen resultante en la ruta especificada
    ruta_completa = os.path.join(carpeta_destino, nombre_archivo)
    img_resultante.save(ruta_completa)
    
    print(f"Imagen guardada exitosamente en: {ruta_completa}")

import numpy as np
import os

# --- FUNCIONES DE SOPORTE HETEROASOCIATIVAS ---
def ejecutar_heteroasociativa_max(rutas_limpias, rutas_ruido, nombres_clases):
    """Clasificación con robustez a ruido sustractivo (Pimienta/Negro)"""
    print("\n--- INICIANDO MAM HETEROASOCIATIVA MAX ---")
    
    # 1. Carga y Preparación
    X_limpias, _ = cargar_imagenes_a_matriz(rutas_limpias)
    num_clases = len(nombres_clases)
    Y_etiquetas = np.eye(num_clases) * 255
    X_ruido, _ = cargar_imagenes_a_matriz(rutas_ruido)

    # 2. Aprendizaje y Recuperación
    W_hetero = aprendizaje_max(X_limpias, Y_etiquetas)
    
    print(f"{'Imagen':<15} | {'Predicción MAX'}")
    print("-" * 35)
    for i, x_test in enumerate(X_ruido):
        y_rec = recuperacion_max(W_hetero, x_test)
        idx = np.argmax(y_rec)
        print(f"Ruido_{i:<10} | {nombres_clases[idx]}")

def ejecutar_heteroasociativa_min(rutas_limpias, rutas_ruido, nombres_clases):
    """Clasificación con robustez a ruido aditivo (Sal/Blanco)"""
    print("\n--- INICIANDO MAM HETEROASOCIATIVA MIN ---")
    
    # 1. Carga y Preparación
    X_limpias, _ = cargar_imagenes_a_matriz(rutas_limpias)
    num_clases = len(nombres_clases)
    Y_etiquetas = np.eye(num_clases) * 255
    X_ruido, _ = cargar_imagenes_a_matriz(rutas_ruido)

    # 2. Aprendizaje y Recuperación
    M_hetero = aprendizaje_min(X_limpias, Y_etiquetas)
    
    print(f"{'Imagen':<15} | {'Predicción MIN'}")
    print("-" * 35)
    for i, x_test in enumerate(X_ruido):
        y_rec = recuperacion_min(M_hetero, x_test)
        idx = np.argmax(y_rec)
        print(f"Ruido_{i:<10} | {nombres_clases[idx]}")


# --- FUNCIONES DE SOPORTE AUTOASOCIATIVAS ---
def ejecutar_autoasociativa_max(rutas_entrenamiento, rutas_con_ruido):
    """Restauración de imagen con robustez a ruido sustractivo (Pimienta/Negro)"""
    print("\n--- INICIANDO MAM AUTOASOCIATIVA MAX ---")
    
    # 1. Carga
    X_train, shape_original = cargar_imagenes_a_matriz(rutas_entrenamiento)
    X_test_ruido, _ = cargar_imagenes_a_matriz(rutas_con_ruido)

    # 2. Aprendizaje (Y = X)
    W_auto = aprendizaje_max(X_train, X_train)

    # 3. Recuperación y Guardado
    os.makedirs("Resultados/Autoasociativa_Max", exist_ok=True)
    for i, x_con_ruido in enumerate(X_test_ruido):
        y_rec = recuperacion_max(W_auto, x_con_ruido)
        guardar_resultado_morfo(
            y_rec, shape_original, f"restaurada_MAX_img_{i}.png", "Resultados/Autoasociativa_Max"
        )
    print("Imágenes restauradas guardadas en: Resultados/Autoasociativa_Max")

def ejecutar_autoasociativa_min(rutas_entrenamiento, rutas_con_ruido):
    """Limpieza de imagen con robustez a ruido aditivo (Sal/Blanco)"""
    print("\n--- INICIANDO MAM AUTOASOCIATIVA MIN ---")
    
    # 1. Carga
    X_train, shape_original = cargar_imagenes_a_matriz(rutas_entrenamiento)
    X_test_ruido, _ = cargar_imagenes_a_matriz(rutas_con_ruido)

    # 2. Aprendizaje (Y = X)
    M_auto = aprendizaje_min(X_train, X_train)

    # 3. Recuperación y Guardado
    os.makedirs("Resultados/Autoasociativa_Min", exist_ok=True)
    for i, x_con_ruido in enumerate(X_test_ruido):
        y_rec = recuperacion_min(M_auto, x_con_ruido)
        guardar_resultado_morfo(
            y_rec, shape_original, f"restaurada_MIN_img_{i}.png", "Resultados/Autoasociativa_Min"
        )
    print("Imágenes restauradas guardadas en: Resultados/Autoasociativa_Min")


# --- INTERFAZ DE USUARIO EN CONSOLA ---

# Función para limpiar la pantalla de la consola (compatible con Windows y Unix)
def limpiar_pantalla():
    """Limpia la consola según el sistema operativo"""
    os.system('cls' if os.name == 'nt' else 'clear')

# Función para mostrar el menú principal
def mostrar_menu():
    limpiar_pantalla()
    print("========================================================================================================")
    print("                     PROYECTO RECONOCIMIENTO DE IMÁGENES - MAM (IPN - ESCOM)  ")
    print("========================================================================================================")
    print("1. MAM Heteroasociativa MAX (Clasificación - Ruido Negro)")
    print("2. MAM Heteroasociativa MIN (Clasificación - Ruido Blanco)")
    print("3. MAM Autoasociativa MAX (Restauración - Ruido Negro)")
    print("4. MAM Autoasociativa MIN (Limpieza - Ruido Blanco)")
    print("5. Salir del programa")
    print("========================================================================================================")

# Programa principal
def ejecutar_programa():
    # --- CONFIGURACIÓN DE RUTAS ---
    clases_pokemon = ["Charmander", "Gengar", "Mewtwo", "Pikachu", "Squirtle"]
    rutas_entrenamiento = [
        "CFP/Fase de aprendizaje/Charmander.bmp",
        "CFP/Fase de aprendizaje/Gengar.bmp",
        "CFP/Fase de aprendizaje/Mewtwo.bmp", 
        "CFP/Fase de aprendizaje/Pikachu.bmp",   
        "CFP/Fase de aprendizaje/Squirtle.bmp" 
    ]
    rutas_con_ruido = [
        "CFP/Patrones espurios/Charmander-Ruido.bmp", 
        "CFP/Patrones espurios/Gengar-Ruido.bmp", 
        "CFP/Patrones espurios/Mewtwo-Ruido.bmp", 
        "CFP/Patrones espurios/Pikachu-Ruido.bmp", 
        "CFP/Patrones espurios/Squirtle-Ruido.bmp"
    ]

    while True:
        mostrar_menu()
        opcion = input("Seleccione una opción (1-5): ")

        if opcion == "1":
            ejecutar_heteroasociativa_max(rutas_entrenamiento, rutas_con_ruido, clases_pokemon)
            input("\nPresione Enter para volver al menú...")
        
        elif opcion == "2":
            ejecutar_heteroasociativa_min(rutas_entrenamiento, rutas_con_ruido, clases_pokemon)
            input("\nPresione Enter para volver al menú...")
            
        elif opcion == "3":
            ejecutar_autoasociativa_max(rutas_entrenamiento, rutas_con_ruido)
            input("\nPresione Enter para volver al menú...")
            
        elif opcion == "4":
            ejecutar_autoasociativa_min(rutas_entrenamiento, rutas_con_ruido)
            input("\nPresione Enter para volver al menú...")
            
        elif opcion == "5":
            print("\nSaliendo del programa...")
            sys.exit() # Sale directamente sin confirmación
            
        else:
            print("\nOpción no válida. Intente de nuevo.")
            input("Presione Enter para continuar...")

# --- PUNTO DE ENTRADA DEL PROGRAMA ---
if __name__ == "__main__":
    ejecutar_programa()
