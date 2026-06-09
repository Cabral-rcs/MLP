import time

from mlp.data import load_mnist
from mlp.network import MLP


def main():
    print("Iniciando treinamento da MLP no MNIST...")
    start_time = time.time()

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

    model = MLP(
        layer_sizes=[784, 128, 64, 10],
        activation="relu",
        learning_rate=0.1,
        seed=42
    )

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=10,
        batch_size=64,
        verbose=True
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\nResultado final no conjunto de teste:")
    print(f"Loss teste: {test_loss:.4f}")
    print(f"Acurácia teste: {test_acc:.4f}")
    print(f"Tempo total: {elapsed_time:.2f} segundos")

    return history, model


if __name__ == "__main__":
    main()