import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from mlp.losses import one_hot_encode


def load_mnist(
    test_size=10000,
    val_size=10000,
    random_state=42
):
    """
    Carrega e prepara o dataset MNIST.

    O MNIST possui imagens 28x28 de dígitos manuscritos.
    Cada imagem é transformada em um vetor com 784 valores.

    Retorno:
        X_train, y_train
        X_val, y_val
        X_test, y_test
    """

    print("Carregando MNIST...")

    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto"
    )

    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)

    # Normalização dos pixels: de 0-255 para 0-1
    X = X / 255.0

    # Primeiro separamos o conjunto de teste
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Depois separamos validação a partir do conjunto de treino
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val
    )

    # Converte os rótulos para one-hot encoding
    y_train = one_hot_encode(y_train, num_classes=10)
    y_val = one_hot_encode(y_val, num_classes=10)
    y_test = one_hot_encode(y_test, num_classes=10)

    print("MNIST carregado com sucesso.")
    print(f"Treino: {X_train.shape}")
    print(f"Validação: {X_val.shape}")
    print(f"Teste: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test