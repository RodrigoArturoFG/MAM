import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv
from skimage.metrics import structural_similarity as ssim
import math

# ====================================================
#   FUNCIONES DE EVALUACIÓN
# ====================================================

from skimage.metrics import structural_similarity as ssim

def evaluar_ssim(original, reconstruido, size=50):
    original = original.reshape((size, size))
    reconstruido = reconstruido.reshape((size, size))

    # Detectar rango automáticamente según dtype y valores
    if np.issubdtype(original.dtype, np.integer):
        # Para imágenes tipo uint8 (0–255)
        data_range = 255
    else:
        # Para imágenes float normalizadas (ej. 0–1)
        data_range = original.max() - original.min()
        if data_range == 0:
            data_range = 1.0  # evitar división por cero

    score, _ = ssim(original, reconstruido, data_range=data_range, full=True)
    return score

def evaluar_psnr(original, reconstruido):
    mse = np.mean((original - reconstruido) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # imágenes normalizadas en [0,1]
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# ====================================================
#   AÑADIR RUIDO ARTIFICIAL
# ====================================================

def aplicar_ruido(img_vector, nivel=0.1):
    """
    Aplica ruido aleatorio a un vector binario.
    nivel = proporción de píxeles alterados (0.1 = 10%)
    """
    noisy = img_vector.copy()
    n = len(noisy)
    num_ruido = int(n * nivel)
    idx = np.random.choice(n, num_ruido, replace=False)
    noisy[idx] = 1 - noisy[idx]  # invertir píxeles
    return noisy

# ====================================================
#   GUARDAR LOG EXTENDIDO
# ====================================================

def guardar_log_csv_ext(distancias, clases, ruta_salida="Resultados/Autoasociativa_Log_Ext.csv"):
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    with open(ruta_salida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ruido","Clase",
                         "Distancia_MAX","SSIM_MAX","PSNR_MAX",
                         "Distancia_MIN","SSIM_MIN","PSNR_MIN"])
        for d in distancias:
            writer.writerow([d["ruido"], clases[d["clase"]],
                             d.get("dist_max"), d.get("ssim_max"), d.get("psnr_max"),
                             d.get("dist_min"), d.get("ssim_min"), d.get("psnr_min")])
    print(f"Log extendido guardado en {ruta_salida}")

# ====================================================
#   FUNCIONES DE APRENDIZAJE Y RECUPERACIÓN
# ====================================================

def aprendizaje_max(X, Y):
    print("Iniciando aprendizaje MAX...")
    p, n = X.shape
    m = Y.shape[1]
    W = np.full((m, n), -999, dtype=np.float32)
    for mu in range(p):
        for i in range(m):
            for j in range(n):
                W[i,j] = max(W[i,j], Y[mu,i] - X[mu,j])
    return W

def aprendizaje_min(X, Y):
    print("Iniciando aprendizaje MIN...")
    p, n = X.shape
    m = Y.shape[1]
    M = np.full((m, n), 999, dtype=np.float32)
    for mu in range(p):
        for i in range(m):
            for j in range(n):
                M[i,j] = min(M[i,j], Y[mu,i] - X[mu,j])
    return M

def recuperacion_max(W, x_test):
    print("Ejecutando recuperación MAX...")
    m, n = W.shape
    y = np.zeros(m, dtype=np.float32)
    for i in range(m):
        valores = []
        for j in range(n):
            valores.append(W[i,j] + x_test[j])
        y[i] = min(valores)
    return y

def recuperacion_min(M, x_test):
    print("Ejecutando recuperación MIN...")
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
#   GUARDAR LOG DE DISTANCIAS EN CSV
# ====================================================

def guardar_log_csv(distancias, clases, ruta_salida="Resultados\Autoasociativa_Log.csv"):
    """
    Guarda en un CSV las distancias entre reconstrucciones y originales.
    - distancias: lista de diccionarios con info de cada prueba
    - clases: nombres de las clases
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    with open(ruta_salida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ruido", "Clase", "Distancia_MAX", "Distancia_MIN"])
        for d in distancias:
            writer.writerow([d["ruido"], clases[d["clase"]], d["dist_max"], d["dist_min"]])
    print(f"Log guardado en {ruta_salida}")


# ====================================================
#   MOSTRAR COMPARACIÓN DE IMÁGENES
# ====================================================
def mostrar_comparacion(original, ruido, reconstruido, clase, metodo, size=50):
    """
    Muestra original, ruido y reconstrucción en una sola figura.
    Además guarda la comparación en un archivo PNG con nombre dinámico.
    """
    print(f"Mostrando comparación para {clase} - Método: {metodo}")
    fig, axs = plt.subplots(1, 3, figsize=(9,3))

    axs[0].imshow(original.reshape((size, size)), cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(ruido.reshape((size, size)), cmap='gray')
    axs[1].set_title("Ruido")
    axs[1].axis("off")

    axs[2].imshow(reconstruido.reshape((size, size)), cmap='gray')
    axs[2].set_title(f"Reconstruido ({metodo})")
    axs[2].axis("off")

    plt.suptitle(f"Autoasociativa - {clase}")
    plt.tight_layout()

    # Carpeta según método
    carpeta = f"Resultados\Autoasociativa_{metodo}"
    os.makedirs(carpeta, exist_ok=True)

    # Nombre dinámico: resultado_clase_metodo.png
    filename = os.path.join(carpeta, f"resultado_{clase}_{metodo}.png")

    # Guardar antes de mostrar
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"Comparación guardada en: {os.path.abspath(filename)}")

# ====================================================
#   OPCIONES DEL MENÚ
# ====================================================

def opcion_entrenamiento(rutas_l, size=50):
    print("\nEntrenando memoria autoasociativa con CFP...")
    X = cargar_dataset_imagenes(rutas_l, size=size, umbral=128)
    W = aprendizaje_max(X, X)
    M = aprendizaje_min(X, X)
    print("Entrenamiento completado!!!!")
    return X, W, M

def opcion_autoasociativa_max(X, W, rutas_r, clases, size=50):
    print("\nEjecutando autoasociativa MAX (Restauración)...")
    X_r = cargar_dataset_imagenes(rutas_r, size=size, umbral=128)
    distancias_log = []

    for i, x in enumerate(X_r):
        y_out_max = recuperacion_max(W, x)

        # Ya no se necesita guardar reconstrucción como BMP, esto se hace dentro de mostrar_comparacion
        # guardar_resultado_imagen(y_out_max, size, "Resultados\Autoasociativa_Max", f"Ruido_{i}_MAX.bmp")

        # Mostrar comparaciones lado a lado
        mostrar_comparacion(X[i], x, y_out_max, clases[i], "MAX", size)

        # Calcular distancia al original
        dist_max = np.linalg.norm(y_out_max - X[i])
        distancias_log.append({"ruido": i, "clase": i, "dist_max": dist_max, "dist_min": None})

        print(f"\nRuido_{i} reconstruido con MAX -> comparación mostrada y guardada en Autoasociativa_Max")

    # Guardar log en CSV
    guardar_log_csv(distancias_log, clases)

# Evaluar funciones de aprendizaje y recuperación
def opcion_autoasociativa_max_eval(X, W, rutas_r, clases, size=50, umbral=128, nivel_ruido=0.0):
    print("\nEjecutando autoasociativa MAX (Restauración)...")
    X_r = cargar_dataset_imagenes(rutas_r, size=size, umbral=umbral)
    distancias_log = []

    for i, x in enumerate(X_r):
        if nivel_ruido > 0:
            x = aplicar_ruido(x, nivel=nivel_ruido)

        y_out_max = recuperacion_max(W, x)
        mostrar_comparacion(X[i], x, y_out_max, clases[i], "MAX", size)

        dist_max = np.linalg.norm(y_out_max - X[i])
        ssim_max = evaluar_ssim(X[i], y_out_max, size)
        psnr_max = evaluar_psnr(X[i], y_out_max)

        distancias_log.append({"ruido": i, "clase": i,
                               "dist_max": dist_max, "ssim_max": ssim_max, "psnr_max": psnr_max,
                               "dist_min": None, "ssim_min": None, "psnr_min": None})

    guardar_log_csv_ext(distancias_log, clases)


def opcion_autoasociativa_min(X, M, rutas_r, clases, size=50):
    print("\nEjecutando autoasociativa MIN (Limpieza)...")
    X_r = cargar_dataset_imagenes(rutas_r, size=size, umbral=128)
    distancias_log = []

    for i, x in enumerate(X_r):
        y_out_min = recuperacion_min(M, x)

        # Guardar reconstrucción como BMP
        guardar_resultado_imagen(y_out_min, size, "Resultados\Autoasociativa_Min", f"Ruido_{i}_MIN.bmp")

        # Mostrar comparaciones lado a lado
        mostrar_comparacion(X[i], x, y_out_min, clases[i], "MIN", size)

        # Calcular distancia al original
        dist_min = np.linalg.norm(y_out_min - X[i])
        distancias_log.append({"ruido": i, "clase": i, "dist_max": None, "dist_min": dist_min})

        print(f"\nRuido_{i} reconstruido con MIN -> comparación mostrada y guardada en Autoasociativa_Min")

    # Guardar log en CSV
    guardar_log_csv(distancias_log, clases)

# Evaluar funciones de aprendizaje y recuperación
def opcion_autoasociativa_min_eval(X, M, rutas_r, clases, size=50, umbral=128, nivel_ruido=0.0):
    print("\nEjecutando autoasociativa MIN (Limpieza)...")
    X_r = cargar_dataset_imagenes(rutas_r, size=size, umbral=umbral)
    distancias_log = []

    for i, x in enumerate(X_r):
        # Aplicar ruido artificial si se especifica
        if nivel_ruido > 0:
            x = aplicar_ruido(x, nivel=nivel_ruido)

        # Recuperación MIN
        y_out_min = recuperacion_min(M, x)

        # Mostrar comparaciones lado a lado
        mostrar_comparacion(X[i], x, y_out_min, clases[i], "MIN", size)

        # Calcular métricas
        dist_min = np.linalg.norm(y_out_min - X[i])
        ssim_min = evaluar_ssim(X[i], y_out_min, size)
        psnr_min = evaluar_psnr(X[i], y_out_min)

        # Guardar resultados en log
        distancias_log.append({
            "ruido": i, "clase": i,
            "dist_max": None, "ssim_max": None, "psnr_max": None,
            "dist_min": dist_min, "ssim_min": ssim_min, "psnr_min": psnr_min
        })

        print(f"\nRuido_{i} reconstruido con MIN -> comparación mostrada y guardada en Autoasociativa_Min")

    # Guardar log extendido en CSV
    guardar_log_csv_ext(distancias_log, clases)

# ====================================================
#   MENÚ PRINCIPAL
# ====================================================
def menu():
    nombres_pokes = ["Charmander", "Gengar", "Mewtwo", "Pikachu", "Squirtle"]
    clases_pokes = {i: nombres_pokes[i] for i in range(len(nombres_pokes))}
    rutas_entrenamiento = [f"CFP/Fase de aprendizaje/{c}.bmp" for c in nombres_pokes]
    rutas_con_ruido = [f"CFP/Patrones espurios/{c}-Ruido.bmp" for c in nombres_pokes]

    X, W, M = None, None, None

    # Parámetros configurables
    size = 50
    umbral = 128
    nivel_ruido = 0.0

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=====================================================")
        print("MAM - AUTOASOCIATIVA - PRUEBA CON IMÁGENES DE POKÉMON")
        print("=====================================================")
        print(f"Parámetros actuales -> size={size}, umbral={umbral}, nivel_ruido={nivel_ruido*100:.0f}%")
        print("-----------------------------------------------------")
        print("1. Entrenamiento con CFP")
        print("2. Autoasociativa MAX (Restauración)")
        print("3. Autoasociativa MIN (Limpieza)")
        print("4. Cambiar parámetros")
        print("5. Salir")
        print("=====================================================")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            X, W, M = opcion_entrenamiento(rutas_entrenamiento, size=size)
        elif opcion == "2":
            if W is None:
                print("\nPrimero debe entrenar la memoria (opción 1).")
            else:
                opcion_autoasociativa_max_eval(X, W, rutas_con_ruido, clases_pokes,
                                          size=size, umbral=umbral, nivel_ruido=nivel_ruido)
        elif opcion == "3":
            if M is None:
                print("\nPrimero debe entrenar la memoria (opción 1).")
            else:
                opcion_autoasociativa_min_eval(X, M, rutas_con_ruido, clases_pokes,
                                          size=size, umbral=umbral, nivel_ruido=nivel_ruido)
        elif opcion == "4":
            try:
                size = int(input("\nNuevo tamaño de imagen (ej. 50, 100): "))
                umbral = int(input("\nNuevo umbral de binarización (0-255): "))
                nivel_ruido = float(input("\nNivel de ruido artificial (ej. 0.1 para 10%): "))
            except ValueError:
                print("\nEntrada inválida, manteniendo parámetros anteriores.")
        elif opcion == "5":
            print("\nSaliendo del programa...")
            break
        else:
            print("\nOpción inválida.")

        input("\nPresione Enter para continuar...")


if __name__ == "__main__":
    menu()
