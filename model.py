from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_model(input_size, hl, lr, name):
    # 4) Definir a arquitetura de rede neural artificial com Tensorflow
    model = Sequential(
        [Dense(input_size, activation='relu')] + # Input
        [Dense(x, activation='relu') for x in hl] +
        [Dense(1, activation='sigmoid')], # Saida
        name
    )
    # 5) Definir um otimizador
    model.compile(optimizer=Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model
