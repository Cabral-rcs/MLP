import numpy as np

from mlp.activations import get_activation
from mlp.losses import softmax


class MLP:
    """
    Rede Neural Multilayer Perceptron implementada do zero com NumPy.

    Nesta primeira versão, a classe faz:
    - inicialização dos pesos
    - forward pass
    - predição

    O treinamento e o backpropagation serão adicionados nas próximas etapas.
    """

    def __init__(self, layer_sizes, activation="relu", seed=42):
        """
        Inicializa a rede neural.

        Parâmetros:
            layer_sizes: lista com a quantidade de neurônios em cada camada.
                Exemplo: [784, 128, 64, 10]

            activation: função de ativação usada nas camadas ocultas.
                Exemplo: "relu", "sigmoid" ou "tanh"

            seed: valor para controlar a aleatoriedade dos pesos.
        """
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.activation, self.activation_derivative = get_activation(activation)

        self.weights = []
        self.biases = []

        np.random.seed(seed)
        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Inicializa os pesos e vieses da rede.

        Usamos inicialização He para ReLU, porque ela ajuda a manter os valores
        em uma escala mais saudável durante o forward pass.

        Para sigmoid e tanh, usamos uma escala baseada em Xavier.
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

    def predict(self, X):
        """
        Retorna a classe prevista para cada amostra.

        Parâmetros:
            X: matriz de entrada

        Retorno:
            array com as classes previstas.
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