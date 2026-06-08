import numpy as np


def relu(x):
    """
    Função de ativação ReLU.

    A ReLU mantém os valores positivos e transforma valores negativos em zero.

    Exemplo:
    [-2, -1, 0, 1, 2] -> [0, 0, 0, 1, 2]
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivada da função ReLU.

    Retorna:
    - 1 para valores maiores que 0
    - 0 para valores menores ou iguais a 0
    """
    return (x > 0).astype(float)


def sigmoid(x):
    """
    Função de ativação Sigmoid.

    Transforma os valores para o intervalo entre 0 e 1.
    """
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivada da função Sigmoid.
    """
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    """
    Função de ativação Tangente Hiperbólica.

    Transforma os valores para o intervalo entre -1 e 1.
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Derivada da função Tanh.
    """
    return 1 - np.tanh(x) ** 2


def get_activation(name):
    """
    Retorna a função de ativação e sua derivada com base no nome informado.

    Exemplo:
    activation, activation_derivative = get_activation("relu")
    """
    activations = {
        "relu": (relu, relu_derivative),
        "sigmoid": (sigmoid, sigmoid_derivative),
        "tanh": (tanh, tanh_derivative),
    }

    if name not in activations:
        raise ValueError(
            f"Ativação '{name}' não encontrada. Use: relu, sigmoid ou tanh."
        )

    return activations[name]