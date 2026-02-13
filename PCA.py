import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def cargar_csv(archivo="caracteristicas.csv"):
    df = pd.read_csv(archivo, encoding="utf-8")
    X = df.drop("Clase", axis=1).values
    y = df["Clase"].values
    # Reemplazar NaN por 0
    X = np.nan_to_num(X, nan=0.0)
    return X, y

def analizar_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Mostrar varianza explicada
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
    plt.savefig("pca_pokemon.png")
    plt.show()

def analizar_tsne(X, y):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
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
    plt.savefig("tsne_pokemon.png")
    plt.show()

def main():
    X, y = cargar_csv("caracteristicas.csv")
    analizar_pca(X, y)
    analizar_tsne(X, y)

if __name__ == "__main__":
    main()
