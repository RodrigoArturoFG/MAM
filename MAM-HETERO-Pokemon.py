import numpy as np
from PIL import Image
import os

# --- 1. CARGA DE DATOS ---
def cargar_imagenes_pokemon(rutas_lista):
    """Carga imágenes BMP de 50x50 y las convierte en vectores para X."""
    datos_aplanados = []
    for ruta in rutas_lista:
        with Image.open(ruta).convert('L') as img:
            # Aseguramos que sea 50x50 ya que es el tamaño esperado de las imagenes
            img = img.resize((50, 50)) 
            datos_aplanados.append(np.array(img).flatten())
    return np.array(datos_aplanados)

def cargar_imagenes_pokemon_binaria(rutas_lista):
    """
    Carga BMP y binariza: 
    Blanco (255) -> 0
    Cualquier otro -> 1
    """
    datos_aplanados = []
    for ruta in rutas_lista:
        with Image.open(ruta).convert('L') as img:
            img = img.resize((50, 50))
            arr = np.array(img)
            
            # Lógica: 0 si es 255 (blanco), 1 en cualquier otro caso
            # Esto pone al Pokémon como '1' y al fondo como '0'
            binario = np.where(arr >= 250, 0, 1) 
            
            datos_aplanados.append(binario.flatten())
    return np.array(datos_aplanados)


# --- 2. CONFIGURACIÓN DE ETIQUETAS (5 Pokémon) ---
# Creamos la matriz Y de 5x5 con etiquetas estáticas
n_pokemon = 5
# Etiquetas One-Hot en rango [0, 1]
Y_etiquetas = np.eye(n_pokemon) 

# --- 3. FUNCIONES MAX - MIN ---
def aprendizaje_max_ciclos(X, Y):
    num_patrones, n = X.shape
    m = Y.shape[1]
    W = np.full((m, n), -np.inf)
    for mu in range(num_patrones):
        for i in range(m):
            for j in range(n):
                diferencia = Y[mu, i] - X[mu, j]
                if diferencia > W[i, j]:
                    W[i, j] = diferencia
    return W

def aprendizaje_min_ciclos(X, Y):
    num_patrones, n = X.shape
    m = Y.shape[1]
    M = np.full((m, n), np.inf)
    for mu in range(num_patrones):
        for i in range(m):
            for j in range(n):
                diferencia = Y[mu, i] - X[mu, j]
                if diferencia < M[i, j]:
                    M[i, j] = diferencia
    return M

def recuperacion_max_ciclos(W, x_test):
    m, n = W.shape
    y_salida = np.full(m, -np.inf)
    for i in range(m):
        for j in range(n):
            valor = W[i, j] + x_test[j]
            if valor > y_salida[i]:
                y_salida[i] = valor
    return y_salida

def recuperacion_min_ciclos(M, x_test):
    m, n = M.shape
    y_salida = np.full(m, np.inf)
    for i in range(m):
        for j in range(n):
            valor = M[i, j] + x_test[j]
            if valor < y_salida[i]:
                y_salida[i] = valor
    return y_salida

# --- 4. MENÚ INTERACTIVO ---
def menu():
    nombres_pokemon = ["Charmander", "Gengar", "Mewtwo", "Pikachu", "Squirtle"]
    rutas_entrenamiento = [f"CFP/Fase de aprendizaje/{c}.bmp" for c in nombres_pokemon]
    rutas_con_ruido = [f"CFP/Patrones espurios/{c}-Ruido.bmp" for c in nombres_pokemon]
    X_train = None
    W, M = None, None

    while True:
        print("\n--- PROYECTO RECONOCIMIENTO POKÉMON (MAM) ---")
        print("1. Cargar Imágenes y Entrenar Memorias")
        print("2. Clasificar con MAM MAX (Ruido Sustractivo)")
        print("3. Clasificar con MAM MIN (Ruido Aditivo)")
        print("4. Salir")
        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            try:
                print("Entrenando con conjunto fundamental de patrones...")
                #X_train = cargar_imagenes_pokemon(rutas_entrenamiento)
                X_train = cargar_imagenes_pokemon_binaria(rutas_entrenamiento)
                W = aprendizaje_max_ciclos(X_train, Y_etiquetas)
                M = aprendizaje_min_ciclos(X_train, Y_etiquetas)
                print("¡Entrenamiento completado!")
            except Exception as e:
                print(f"Error: {e}")

        elif opcion == "2" or opcion == "3":
            if W is None or M is None:
                print("Error: Primero debes entrenar (Opción 1).")
                continue
            
            tipo = "MAX" if opcion == "2" else "MIN"
            print(f"\n--- ANÁLISIS DE VECTORES MAM {tipo} ---")
            # Encabezado de tabla ajustado para ver los vectores
            print(f"{'Imagen Prueba':<15} | {'Vector de Salida (Activación)':<45} | {'Ganador':<12}")
            print("-" * 85)

            try:
                #X_test_lote = cargar_imagenes_pokemon(rutas_con_ruido)
                X_test_lote = cargar_imagenes_pokemon_binaria(rutas_con_ruido)
                
                for i in range(len(rutas_con_ruido)):
                    x_vector = X_test_lote[i]
                    
                    if opcion == "2":
                        y_res = recuperacion_max_ciclos(W, x_vector)
                    else:
                        y_res = recuperacion_min_ciclos(M, x_vector)
                    
                    # Clasificación
                    idx_ganador = np.argmax(y_res)
                    nombre_detectado = nombres_pokemon[idx_ganador]
                    
                    # Formateamos el vector para que se vea limpio (sin decimales innecesarios)
                    vector_str = "[" + ", ".join([f"{val:g}" for val in y_res]) + "]"
                    
                    print(f"{rutas_con_ruido[i]:<15} | {vector_str:<45} | {nombre_detectado:<12}")
                
                print("-" * 85)

            except Exception as e:
                print(f"Error: {e}")

        elif opcion == "4":
            break



if __name__ == "__main__":
    menu()

