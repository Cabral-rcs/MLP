import numpy as np

from mlp.activations import get_activation
from mlp.losses import softmax, cross_entropy, accuracy
from mlp.optimizers import SGD


class MLP:
    """
    Rede Neural Multilayer Perceptron implementada do zero com NumPy.

    Esta versão possui:
    - inicialização dos pesos
    - forward pass
    - backpropagation
    - atualização com SGD
    - treinamento com mini-batches
    - avaliação
    - predição
    """

    def __init__(
        self,
        layer_sizes,
        activation="relu",
        learning_rate=0.01,
        seed=42
    ):
        """
        Inicializa a rede neural.

        Parâmetros:
            layer_sizes: lista com a quantidade de neurônios em cada camada.
                Exemplo: [784, 128, 64, 10]

            activation: função de ativação usada nas camadas ocultas.
                Exemplo: "relu", "sigmoid" ou "tanh"

            learning_rate: taxa de aprendizado usada pelo SGD.

            seed: valor para controlar a aleatoriedade dos pesos.
        """

        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.activation, self.activation_derivative = get_activation(activation)

        self.weights = []
        self.biases = []

        self.optimizer = SGD(learning_rate=learning_rate)

        np.random.seed(seed)
        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Inicializa os pesos e vieses da rede.

        Para ReLU, usamos inicialização He.
        Para sigmoid e tanh, usamos uma inicialização próxima da Xavier.
        """

        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]

            if self.activation_name == "relu":
                scale = np.sqrt(2 / input_size)
            else:
                scale = np.sqrt(1 / input_size)

            weight = np.random.randn(input_size, output_size) * scale
            bias = np.zeros((1, output_size))

            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        """
        Executa o forward pass da rede.

        Parâmetros:
            X: matriz de entrada com shape (n_amostras, n_features)

        Retorno:
            probabilidades da última camada, após softmax.
        """

        self.activations_cache = [X]
        self.z_cache = []

        A = X

        for i in range(len(self.weights)):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.z_cache.append(Z)

            is_last_layer = i == len(self.weights) - 1

            if is_last_layer:
                A = softmax(Z)
            else:
                A = self.activation(Z)

            self.activations_cache.append(A)

        return A

    def backward(self, y_true):
        """
        Executa o backpropagation.

        Parâmetros:
            y_true: rótulos verdadeiros em one-hot encoding

        Retorno:
            dicionário com os gradientes dW e db
        """

        m = y_true.shape[0]

        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)

        dZ = self.activations_cache[-1] - y_true

        for i in reversed(range(len(self.weights))):
            A_prev = self.activations_cache[i]

            dW[i] = np.dot(A_prev.T, dZ) / m
            db[i] = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                dZ = dA_prev * self.activation_derivative(self.z_cache[i - 1])

        gradients = {
            "dW": dW,
            "db": db
        }

        return gradients

    def update_parameters(self, gradients):
        """
        Atualiza os pesos e vieses usando o otimizador configurado.
        """

        self.weights, self.biases = self.optimizer.update(
            self.weights,
            self.biases,
            gradients
        )

    def train_step(self, X, y_true):
        """
        Executa uma etapa de treinamento em um batch.
        """

        y_pred = self.forward(X)

        loss = cross_entropy(y_pred, y_true)
        acc = accuracy(y_pred, y_true)

        gradients = self.backward(y_true)
        self.update_parameters(gradients)

        return loss, acc

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=20,
        batch_size=64,
        verbose=True
    ):
        """
        Treina a rede neural usando mini-batches.

        Parâmetros:
            X_train: dados de treino
            y_train: rótulos de treino em one-hot
            X_val: dados de validação, opcional
            y_val: rótulos de validação em one-hot, opcional
            epochs: quantidade de épocas
            batch_size: tamanho de cada mini-batch
            verbose: se True, mostra o progresso no terminal

        Retorno:
            histórico com loss e acurácia de treino e validação
        """

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)

            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            batch_losses = []
            batch_accuracies = []

            for start in range(0, n_samples, batch_size):
                end = start + batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                batch_loss, batch_acc = self.train_step(X_batch, y_batch)

                batch_losses.append(batch_loss)
                batch_accuracies.append(batch_acc)

            train_loss = np.mean(batch_losses)
            train_acc = np.mean(batch_accuracies)

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)

                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

                if verbose:
                    print(
                        f"Época {epoch + 1}/{epochs} | "
                        f"loss treino: {train_loss:.4f} | "
                        f"acc treino: {train_acc:.4f} | "
                        f"loss val: {val_loss:.4f} | "
                        f"acc val: {val_acc:.4f}"
                    )
            else:
                if verbose:
                    print(
                        f"Época {epoch + 1}/{epochs} | "
                        f"loss treino: {train_loss:.4f} | "
                        f"acc treino: {train_acc:.4f}"
                    )

        return history

    def evaluate(self, X, y_true):
        """
        Avalia a rede em um conjunto de dados.

        Parâmetros:
            X: dados de entrada
            y_true: rótulos verdadeiros em one-hot

        Retorno:
            loss e acurácia
        """

        y_pred = self.forward(X)

        loss = cross_entropy(y_pred, y_true)
        acc = accuracy(y_pred, y_true)

        return loss, acc

    def predict(self, X):
        """
        Retorna a classe prevista para cada amostra.
        """

        probabilities = self.forward(X)
        predictions = np.argmax(probabilities, axis=1)

        return predictions

    def summary(self):
        """
        Mostra um resumo simples da arquitetura da rede.
        """

        print("Arquitetura da MLP:")

        for i in range(len(self.layer_sizes) - 1):
            print(
                f"Camada {i + 1}: "
                f"{self.layer_sizes[i]} -> {self.layer_sizes[i + 1]}"
            )

        print(f"Ativação nas camadas ocultas: {self.activation_name}")