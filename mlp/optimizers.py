class SGD:
    """
    Otimizador Stochastic Gradient Descent.

    Ele atualiza os pesos e vieses da rede usando os gradientes calculados
    pelo backpropagation.

    Fórmula:
        parametro = parametro - learning_rate * gradiente
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, biases, gradients):
        """
        Atualiza pesos e vieses.

        Parâmetros:
            weights: lista com matrizes de pesos
            biases: lista com vetores de vieses
            gradients: dicionário contendo dW e db
        """

        for i in range(len(weights)):
            weights[i] -= self.learning_rate * gradients["dW"][i]
            biases[i] -= self.learning_rate * gradients["db"][i]

        return weights, biases