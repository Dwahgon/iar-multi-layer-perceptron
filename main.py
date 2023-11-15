import numpy as np
from args import args # Keep this first
from ucimlrepo import fetch_ucirepo
import math
from tensorflow import keras
from keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, roc_curve, auc
from model import build_model
import matplotlib.pyplot as plt
from plot import save_or_show

keras.utils.set_random_seed(args.random_state)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

print(len(X.columns))

# 4) Definir a arquitetura de rede neural artificial com Tensorflow
# 5) Definir um otimizador
models = [build_model(len(X.columns), args.hidden_layers, learning_rate, f'mlp_lr_{learning_rate:.3f}') for learning_rate in args.learning_rates]
model_names = [model.name for model in models]

# 6) Treinar o modelo
histories = [model.fit(X_train, y_train, epochs=50,
        batch_size=2000,
        validation_split=0.2) for model in models]

# 7) Avaliar o modelo
# evaluations = [model.evaluate(X_test) for model in models]

# 8) Medidads de acurácidade
y_preds = [model.predict(X_test) for model in models]
y_preds_class = [(y_pred > 0.5).astype('int32') for y_pred in y_preds]
accuracies = [accuracy_score(y_test, y_pred_class) for y_pred_class in y_preds_class]
losses = [binary_crossentropy(y_test, y_pred).numpy()[0] for y_pred in y_preds]

## Graphs
fig, ((bars_ax, train_acc_ax, train_loss_ax), (nax, val_acc_ax, val_loss_ax)) = plt.subplots(2, 3, figsize=(20, 10))
fig.delaxes(nax)

# Bars
width = .25
x = np.arange(len(models))  # the label locations
multiplier = 0
for name, values in [('accuracy', accuracies), ('binary_crossentropy', losses)]:
    offset = width * multiplier
    rects = bars_ax.bar(x + offset, [float(f'{x:.4f}') for x in values], width, label=name)
    bars_ax.bar_label(rects, padding=3)
    multiplier += 1
bars_ax.set_title('Acurácias e Perdas')
bars_ax.set_xticks(x + width, model_names)
bars_ax.set_ylim(0, 1.2)
bars_ax.legend(loc='best', ncols=2)
# bars_ax.set_ylim(0, 1)

# Training graph
def plot_stats_graph(values, axis, name, ylabel):
    for value in values:
        axis.plot(value)
    axis.set_title(name)
    axis.set_ylabel(ylabel)
    axis.set_xlabel('epoch')
    axis.legend(model_names, loc='best')
plot_stats_graph([history.history['accuracy'] for history in histories], train_acc_ax, 'Acurácidade no treinamento', 'accuracy')
plot_stats_graph([history.history['loss'] for history in histories], train_loss_ax, 'Perda no treinamento', 'loss')
plot_stats_graph([history.history['val_accuracy'] for history in histories], val_acc_ax, 'Acurácidade na validação', 'accuracy')
plot_stats_graph([history.history['val_loss'] for history in histories], val_loss_ax, 'Perda na validação', 'loss')
plt.tight_layout()
save_or_show('results.png')

# confusion matrix
for model_name, y_pred in zip(model_names, y_preds_class):
    plt.clf()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    save_or_show(f'{model_name}_cm.png')

# ROC Curve
model_roc_curves = [roc_curve(y_test, p.ravel()) for p in y_preds]
aucs = [auc(rc[0], rc[1]) for rc in model_roc_curves]

plt.clf()
plt.figure(figsize=(7,7))
plt.plot([0, 1], [0, 1], 'k--')
for i, rc, a in zip(range(len(model_roc_curves)), model_roc_curves, aucs):
    plt.plot(rc[0], rc[1], label=f'model {i + 1}(area = {a:.3f})')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
save_or_show(f'roc.png')