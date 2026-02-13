import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def cargar_csv(archivo="caracteristicas.csv"):
    df = pd.read_csv(archivo, encoding="utf-8")
    X = df.drop("Clase", axis=1).values
    y = df["Clase"].values
    # Reemplazar NaN por 0 para evitar errores
    X = np.nan_to_num(X, nan=0.0)
    return X, y

def analizar_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
    print("Varianza total explicada:", np.sum(pca.explained_variance_ratio_))

    plt.figure(figsize=(8,6))
    for clase in np.unique(y):
        idx = y == clase
        plt.scatter(X_pca[idx,0], X_pca[idx,1], label=clase)
    plt.title("PCA de características Pokémon (extendidas)")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.legend()
    plt.tight_layout()

    os.makedirs("Resultados", exist_ok=True)
    plt.savefig("Resultados/pca_pokemon.png", dpi=300)
    plt.show()

def analizar_tsne(X, y):
    n_samples = X.shape[0]
    perplexity = min(30, max(2, n_samples - 1))  # Ajuste automático
    print(f"Usando perplexity={perplexity} para {n_samples} muestras")

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8,6))
    for clase in np.unique(y):
        idx = y == clase
        plt.scatter(X_tsne[idx,0], X_tsne[idx,1], label=clase)
    plt.title("t-SNE de características Pokémon (extendidas)")
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.legend()
    plt.tight_layout()

    os.makedirs("Resultados", exist_ok=True)
    plt.savefig("Resultados/tsne_pokemon.png", dpi=300)
    plt.show()

    print(f"Gráfica t-SNE guardada en: {os.path.abspath('Resultados/tsne_pokemon.png')}")


# Datos de distancias
def comparar_distancias():
    clases = ["Charmander", "Gengar", "Mewtwo", "Pikachu", "Squirtle"]
    dist_min = [38.73, 45.14, 36.59, 36.39, 42.78]
    dist_max = [39.90, 32.47, 41.87, 42.05, 18.57]

    x = range(len(clases))
    plt.figure(figsize=(8,6))

    plt.bar([i-0.2 for i in x], dist_min, width=0.4, label="MIN", color="skyblue")
    plt.bar([i+0.2 for i in x], dist_max, width=0.4, label="MAX", color="salmon")

    plt.xticks(x, clases)
    plt.ylabel("Distancia al original")
    plt.title("Comparación Autoasociativa MIN vs MAX")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Resultados\comparacion_min_max.png", dpi=300)
    plt.show()

# ====================================================
#   MENÚ PRINCIPAL
# ====================================================

def menu():
    X, y = cargar_csv("caracteristicas.csv")

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=====================================================")
        print("ANÁLISIS DE CARACTERÍSTICAS - PCA / t-SNE")
        print("=====================================================")
        print("1. Analizar con PCA")
        print("2. Analizar con t-SNE")
        print("3. Comparar distancias MIN vs MAX")
        print("4. Salir")
        print("=====================================================")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            analizar_pca(X, y)
        elif opcion == "2":
            analizar_tsne(X, y)
        elif opcion == "3":
            comparar_distancias()
        elif opcion == "4":
            print("Saliendo del programa...")
            break
        else:
            print("Opción inválida.")

        input("\nPresione Enter para continuar...")

if __name__ == "__main__":
    menu()
