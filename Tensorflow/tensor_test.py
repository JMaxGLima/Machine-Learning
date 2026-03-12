"""
Módulo de treinamento de rede neural para classificação de dígitos MNIST.

Este módulo implementa uma rede neural convolucional simples usando TensorFlow/Keras
para classificar imagens de dígitos manuscritos do dataset MNIST. O código demonstra
o fluxo completo de aprendizado de máquina: carregamento de dados, criação do modelo,
compilação, treinamento e avaliação.

Estrutura do módulo:
- load_data(): Carrega e pré-processa o dataset MNIST
- create_model(): Cria arquitetura da rede neural
- get_predictions(): Obtém previsões do modelo
- get_loss_fn(): Configura função de perda
- compile_model(): Compila o modelo para treinamento
- train_model(): Treina e avalia o modelo
- main(): Função principal que orquestra todo o processo

"""

import tensorflow as tf

def load_data():
    """
    Carrega e pré-processa o dataset MNIST.
    
    Returns:
        tuple: Uma tupla contendo (x_train, y_train, x_test, y_test)
            - x_train: Imagens de treinamento normalizadas (float32)
            - y_train: Labels de treinamento (int)
            - x_test: Imagens de teste normalizadas (float32)
            - y_test: Labels de teste (int)
    """
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

def create_model():
    """
    Cria um modelo de rede neural sequencial para classificação MNIST.
    
    O modelo consiste em:
    - Camada Flatten: Transforma imagens 28x28 em vetores 784
    - Camada Dense: 128 neurônios com ativação ReLU
    - Camada Dropout: 20% de dropout para regularização
    - Camada Dense de saída: 10 neurônios (um para cada dígito)
    
    Returns:
        tf.keras.models.Sequential: Modelo Keras não compilado
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    return model

def get_predictions(model, x_train):
    """
    Obtém previsões brutas e probabilidades do modelo para uma amostra.
    
    Args:
        model: Modelo Keras treinado ou não
        x_train: Dados de treinamento (numpy array)
    
    Returns:
        tuple: (predictions, probabilities)
            - predictions: Logits brutos do modelo (numpy array)
            - probabilities: Probabilidades normalizadas com softmax (numpy array)
    """
    predictions = model(x_train[:1]).numpy()
    probabilities = tf.nn.softmax(predictions).numpy()
    return predictions, probabilities

def get_loss_fn(y_train, predictions):
    """
    Cria a função de perda e calcula a perda para uma amostra.
    
    Args:
        y_train: Labels de treinamento
        predictions: Previsões do modelo
    
    Returns:
        tuple: (loss_fn, loss)
            - loss_fn: Função de perda SparseCategoricalCrossentropy
            - loss: Valor da perda calculada para a primeira amostra
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = loss_fn(y_train[:1], predictions).numpy()
    return loss_fn, loss

def compile_model(model, loss_fn):
    """
    Compila o modelo com otimizador, função de perda e métricas.
    
    Args:
        model: Modelo Keras a ser compilado
        loss_fn: Função de perda para otimização
    """
    model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

def train_model(model, x_train, y_train, x_test, y_test):
    """
    Treina o modelo e avalia seu desempenho.
    
    Args:
        model: Modelo Keras compilado
        x_train: Dados de treinamento
        y_train: Labels de treinamento
        x_test: Dados de teste
        y_test: Labels de teste
    
    Returns:
        tf.keras.Sequential: Modelo com camada softmax para probabilidades
    """
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test,  y_test, verbose=2)
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    probability_model(x_test[:5])
    return probability_model

def main():
    """
    Função principal que orquestra o fluxo completo de treinamento.
    
    Executa as seguintes etapas:
    1. Carrega e pré-processa os dados MNIST
    2. Cria a arquitetura do modelo
    3. Realiza previsões de teste
    4. Configura a função de perda
    5. Compila o modelo
    6. Treina e avalia o modelo
    """
    print(" Versão do TensorFlow:", tf.__version__)
    x_train, y_train, x_test, y_test = load_data()
    print("Dados carregados com sucesso")
    print("Formato dos dados de treino:", x_train.shape)
    print("Formato das labels de treino:", y_train.shape)
    print("Formato dos dados de teste:", x_test.shape)
    print("Formato das labels de teste:", y_test.shape)
    
    model = create_model()
    print("Modelo criado com sucesso")
    model.summary()
    
    predictions, probabilities = get_predictions(model, x_train)
    print("Previsões:", predictions)
    print("Probabilidades:", probabilities)
    
    loss_fn, loss = get_loss_fn(y_train, predictions)
    print("Função de perda:", loss_fn)
    print("Perda:", loss)
    
    compile_model(model, loss_fn)
    print("Modelo compilado com sucesso")
    
    train_model(model, x_train, y_train, x_test, y_test)
    print("Modelo treinado com sucesso")

if __name__ == "__main__":
    main()
