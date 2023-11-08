from ucimlrepo import fetch_ucirepo
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.3
HIDDEN_LAYERS = [
    40, 40, 50
]

# 1) Load dataset and remove rows with missing vars
breast_cancer_wisconsin_original = fetch_ucirepo(id=15)

X = breast_cancer_wisconsin_original.data.features
y = breast_cancer_wisconsin_original.data.targets

for i, row in X.iterrows():
    for column in row:
        if math.isnan(column):
            X = X.drop(row.name)
            y = y.drop(row.name)

for i, row in y.iterrows():
    for column in row:
        if math.isnan(column):
            print("removing ", row.name)
            X = X.drop(row.name)
            y = y.drop(row.name)

X[:30].plot(kind = 'bar')
# plt.show()

y[:30].plot(kind = 'bar')
# plt.show()

# 2) Normaliza dataset

# Normalize X using maximum absolute scaling
for col in X.columns:
    X[col] = X[col] / X[col].abs().max()


# Normalize Y using min-max scaling
for col in y.columns:
    y[col] = (y[col] - y[col].min()) / (y[col].max() - y[col].min())

X[:30].plot(kind = 'bar')
# plt.show()

y[:30].plot(kind = 'bar')
# plt.show()

# 3) Dividir os dados em conjunto treinamento e teste utilizando método holdout
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

print(len(X.columns))

# 4) Definir a arquitetura de rede neural artificial com Tensorflow
model = Sequential(
    [Dense(len(X.columns), activation='relu')] + # Input
    [Dense(x, activation='relu') for x in HIDDEN_LAYERS] +
    [Dense(1, activation='sigmoid')] # Saida
)

# 5) Definir um otimizador
model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# 6) Treinar o modelo
model.fit(X_train, y_train, epochs=10,
    batch_size=2000,
    validation_split=0.2)

# 7) Avaliar o modelo
results = model.evaluate(X_test, y_test, verbose=0)
print('test loss, test acc:', results)