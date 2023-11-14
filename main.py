from ucimlrepo import fetch_ucirepo
import math
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, roc_curve, auc
from model import build_model
import matplotlib.pyplot as plt
from plot import save_or_show

TEST_SIZE = 0.3
HIDDEN_LAYERS = [
    20, 20
]
random_state = 12

keras.utils.set_random_seed(random_state)

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

# X[:30].plot(kind = 'bar')
# plt.show()

# y[:30].plot(kind = 'bar')
# plt.show()

# 2) Normaliza dataset

# Normalize X using maximum absolute scaling
for col in X.columns:
    X[col] = X[col] / X[col].abs().max()


# Normalize Y using min-max scaling
for col in y.columns:
    y[col] = (y[col] - y[col].min()) / (y[col].max() - y[col].min())

y = y.astype('int32')

# X[:30].plot(kind = 'bar')
# plt.show()

# y[:30].plot(kind = 'bar')
# plt.show()

# 3) Dividir os dados em conjunto treinamento e teste utilizando método holdout
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=random_state)

print(len(X.columns))

# 4) Definir a arquitetura de rede neural artificial com Tensorflow
# 5) Definir um otimizador
input_size = len(X.columns)
models = [
    build_model(input_size, HIDDEN_LAYERS, 0.001),
    build_model(input_size, HIDDEN_LAYERS, 0.01),
    build_model(input_size, HIDDEN_LAYERS, 0.1),
    build_model(input_size, HIDDEN_LAYERS, 0.3),
]
model_preds = []

for i, model in enumerate(models):
    print("=" * 10 + " Modelo %d " % (i + 1) + "="*10)

    # 6) Treinar o modelo
    history = model.fit(X_train, y_train, epochs=50,
        batch_size=2000,
        validation_split=0.2)

    # 7) Avaliar o modelo
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype('int32')
    model_preds.append(y_pred)

    # 8) Valores de acuracidade

    fig, ((t_ax, cm_ax), (acc_ax, loss_ax)) = plt.subplots(2, 2)
    # Acuracidade
    acc = accuracy_score(y_test, y_pred_class)

    # Matriz confusão
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_class, ax=cm_ax)
    cm_ax.set_title(f'model {i + 1} confusion matrix')

    t_ax.text(0, 0.4, f'Test accuracy {acc:.4f}')

    # Acuracidade no treinamento
    acc_ax.plot(history.history['accuracy'])
    acc_ax.plot(history.history['val_accuracy'])
    acc_ax.set_title(f'model {i + 1} training accuracy')
    acc_ax.set_ylabel('accuracy')
    acc_ax.set_xlabel('epoch')
    acc_ax.legend(['train', 'val'], loc='upper left')

    # Perda no treinamento
    loss_ax.plot(history.history['loss'])
    loss_ax.plot(history.history['val_loss'])
    loss_ax.set_title(f'model {i + 1} training loss')
    loss_ax.set_ylabel('loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.legend(['train', 'val'], loc='upper left')

    plt.tight_layout()
    save_or_show(f'model{i + 1}.png')

# ROC Curve
model_roc_curves = [roc_curve(y_test, p.ravel()) for p in model_preds]
aucs = [auc(rc[0], rc[1]) for rc in model_roc_curves]

plt.plot([0, 1], [0, 1], 'k--')
for i, rc, a in zip(range(len(model_roc_curves)), model_roc_curves, aucs):
    plt.plot(rc[0], rc[1], label=f'model {i + 1}(area = {a:.3f})')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
save_or_show(f'roc.png')