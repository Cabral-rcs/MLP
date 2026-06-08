# MLP do Zero com NumPy - MNIST

Este projeto implementa uma rede neural MLP do zero utilizando NumPy, com o objetivo de classificar dígitos manuscritos do dataset MNIST.

## Objetivo

Construir uma rede neural multicamadas sem utilizar frameworks de deep learning, como PyTorch ou TensorFlow.

A meta é alcançar pelo menos 92% de acurácia no conjunto de teste.

## Estrutura do Projeto

```txt
.
├── README.md
├── requirements.txt
├── mlp/
│   ├── __init__.py
│   ├── network.py
│   ├── activations.py
│   ├── losses.py
│   └── optimizers.py
├── notebooks/
│   └── experimentos.ipynb
└── results/