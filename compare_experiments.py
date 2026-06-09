import csv
import os
import time

from mlp.data import load_mnist
from mlp.network import MLP


def run_experiment(
    name,
    layer_sizes,
    activation,
    learning_rate,
    epochs,
    batch_size,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):
    print("\n" + "=" * 60)
    print(f"Iniciando experimento: {name}")
    print("=" * 60)

    start_time = time.time()

    model = MLP(
        layer_sizes=layer_sizes,
        activation=activation,
        learning_rate=learning_rate,
        seed=42
    )

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)

    elapsed_time = time.time() - start_time

    result = {
        "nome": name,
        "arquitetura": str(layer_sizes),
        "ativacao": activation,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_loss_final": history["train_loss"][-1],
        "train_accuracy_final": history["train_accuracy"][-1],
        "val_loss_final": history["val_loss"][-1],
        "val_accuracy_final": history["val_accuracy"][-1],
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "tempo_segundos": elapsed_time
    }

    print("\nResultado do experimento:")
    print(f"Loss teste: {test_loss:.4f}")
    print(f"Acurácia teste: {test_acc:.4f}")
    print(f"Tempo: {elapsed_time:.2f} segundos")

    return result


def save_results(results, filepath="results/experimentos.csv"):
    os.makedirs("results", exist_ok=True)

    fieldnames = [
        "nome",
        "arquitetura",
        "ativacao",
        "learning_rate",
        "epochs",
        "batch_size",
        "train_loss_final",
        "train_accuracy_final",
        "val_loss_final",
        "val_accuracy_final",
        "test_loss",
        "test_accuracy",
        "tempo_segundos"
    ]

    with open(filepath, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResultados salvos em: {filepath}")


def main():
    print("Carregando dados uma única vez...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

    experiments = [
        {
            "name": "baseline_128_64_lr_01",
            "layer_sizes": [784, 128, 64, 10],
            "activation": "relu",
            "learning_rate": 0.1,
            "epochs": 10,
            "batch_size": 64
        },
        {
            "name": "maior_256_128_lr_01",
            "layer_sizes": [784, 256, 128, 10],
            "activation": "relu",
            "learning_rate": 0.1,
            "epochs": 10,
            "batch_size": 64
        },
        {
            "name": "baseline_128_64_lr_005",
            "layer_sizes": [784, 128, 64, 10],
            "activation": "relu",
            "learning_rate": 0.05,
            "epochs": 10,
            "batch_size": 64
        }
    ]

    results = []

    for experiment in experiments:
        result = run_experiment(
            name=experiment["name"],
            layer_sizes=experiment["layer_sizes"],
            activation=experiment["activation"],
            learning_rate=experiment["learning_rate"],
            epochs=experiment["epochs"],
            batch_size=experiment["batch_size"],
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test
        )

        results.append(result)

    save_results(results)

    print("\nResumo final dos experimentos:")
    for result in results:
        print(
            f"{result['nome']} | "
            f"acc teste: {result['test_accuracy']:.4f} | "
            f"loss teste: {result['test_loss']:.4f} | "
            f"tempo: {result['tempo_segundos']:.2f}s"
        )


if __name__ == "__main__":
    main()