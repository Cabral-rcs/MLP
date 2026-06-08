import numpy as np


def softmax(logits):
    """
    Aplica a função softmax nos valores de saída da rede.

    A softmax transforma os logits em probabilidades.
    Cada linha da saída soma aproximadamente 1.

    Parâmetros:
        logits: array com shape (n_amostras, n_classes)

    Retorno:
        probabilidades com shape (n_amostras, n_classes)
    """
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits_shifted)
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    return probabilities


def cross_entropy(y_pred, y_true):
    """
    Calcula a perda cross-entropy.

    Parâmetros:
        y_pred: probabilidades previstas pela rede, shape (n_amostras, n_classes)
        y_true: rótulos verdadeiros em one-hot, shape (n_amostras, n_classes)

    Retorno:
        valor médio da perda
    """
    epsilon = 1e-12
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)

    loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_true.shape[0]

    return loss


def accuracy(y_pred, y_true):
    """
    Calcula a acurácia da classificação.

    Parâmetros:
        y_pred: probabilidades previstas pela rede, shape (n_amostras, n_classes)
        y_true: rótulos verdadeiros em one-hot, shape (n_amostras, n_classes)

    Retorno:
        acurácia entre 0 e 1
    """
    predicted_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)

    return np.mean(predicted_classes == true_classes)


def one_hot_encode(y, num_classes):
    """
    Transforma rótulos inteiros em representação one-hot.

    Exemplo:
        y = [2, 0, 1]
        num_classes = 3

        resultado:
        [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]]

    Parâmetros:
        y: array com rótulos inteiros
        num_classes: quantidade total de classes

    Retorno:
        matriz one-hot com shape (n_amostras, num_classes)
    """
    y = np.array(y, dtype=int)
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1

    return one_hot