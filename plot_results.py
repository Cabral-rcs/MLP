import csv
import os

import matplotlib.pyplot as plt


def read_training_history(filepath="results/historico_treino.csv"):
    epochs = []
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    with open(filepath, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_accuracy.append(float(row["train_accuracy"]))
            val_loss.append(float(row["val_loss"]))
            val_accuracy.append(float(row["val_accuracy"]))

    return epochs, train_loss, train_accuracy, val_loss, val_accuracy


def read_experiments(filepath="results/experimentos.csv"):
    names = []
    test_accuracies = []
    test_losses = []
    times = []

    with open(filepath, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            names.append(row["nome"])
            test_accuracies.append(float(row["test_accuracy"]))
            test_losses.append(float(row["test_loss"]))
            times.append(float(row["tempo_segundos"]))

    return names, test_accuracies, test_losses, times


def plot_loss_curve():
    epochs, train_loss, _, val_loss, _ = read_training_history()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Treino")
    plt.plot(epochs, val_loss, marker="o", label="Validação")
    plt.title("Evolução da Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/loss_por_epoca.png", dpi=300)
    plt.close()

    print("Gráfico salvo em: results/loss_por_epoca.png")


def plot_accuracy_curve():
    epochs, _, train_accuracy, _, val_accuracy = read_training_history()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracy, marker="o", label="Treino")
    plt.plot(epochs, val_accuracy, marker="o", label="Validação")
    plt.title("Evolução da Acurácia")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/acuracia_por_epoca.png", dpi=300)
    plt.close()

    print("Gráfico salvo em: results/acuracia_por_epoca.png")


def plot_experiment_accuracy():
    names, test_accuracies, _, _ = read_experiments()

    plt.figure(figsize=(10, 5))
    plt.bar(names, test_accuracies)
    plt.title("Acurácia dos Experimentos no Teste")
    plt.xlabel("Experimento")
    plt.ylabel("Acurácia")
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0.95, 0.98)
    plt.tight_layout()
    plt.savefig("results/comparacao_acuracia.png", dpi=300)
    plt.close()

    print("Gráfico salvo em: results/comparacao_acuracia.png")


def plot_experiment_time():
    names, _, _, times = read_experiments()

    plt.figure(figsize=(10, 5))
    plt.bar(names, times)
    plt.title("Tempo de Execução dos Experimentos")
    plt.xlabel("Experimento")
    plt.ylabel("Tempo em segundos")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("results/comparacao_tempo.png", dpi=300)
    plt.close()

    print("Gráfico salvo em: results/comparacao_tempo.png")


def main():
    os.makedirs("results", exist_ok=True)

    plot_loss_curve()
    plot_accuracy_curve()
    plot_experiment_accuracy()
    plot_experiment_time()

    print("\nTodos os gráficos foram gerados com sucesso.")


if __name__ == "__main__":
    main()