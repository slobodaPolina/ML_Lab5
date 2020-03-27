import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from core import AdaBoost
import data


def cross_validation(X, Y, ids_batches, boost):
    y_classified = []
    y_actual = []
    # перебираю возможные обучающие сеты, выкидывая test_num
    # обучаю, тестирую на тестовых данных
    for test_num in range(len(ids_batches)):
        X_train, Y_train = data.train_dataset(X, Y, ids_batches, test_num)
        boost.fit(X_train, Y_train)
        for i in ids_batches[test_num]:
            y_prediction = boost.classify(X[i])
            y_classified.append(y_prediction)
            y_actual.append(Y[i])
    # оценка, насколько массивы реальных и предсказанных значений расходятся
    return accuracy_score(y_actual, y_classified)


for path in ['chips', 'geyser']:
    features, values = data.read_data('resources/' + path + '.csv')
    # массив из 5 (по умолчанию) массивов со смешанными значениями от 0 до len(Y) (количества строк в датасете)
    ids_batches = data.split_indices_data(len(values))
    points = []
    for models_amount in range(5, 30):
        boost = AdaBoost(models_amount)
        points.append({'models': models_amount, 'score': cross_validation(features, values, ids_batches, boost)})
    plt.plot([point['models'] for point in points], [point['score'] for point in points])
    plt.xlabel('models amount')
    plt.ylabel('accuracy score')
    plt.legend()
    plt.show()


def draw_plot(X, Y, boost, title):
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.Paired)
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = boost.predict(xy).reshape(XX.shape)
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.title(title)
    plt.show()


initial_models = 10
for path in ['chips', 'geyser']:
    features, values = data.read_data('resources/' + path + '.csv')
    boost = AdaBoost(initial_models)
    # обучение на всем датасете
    boost.fit(features, values)
    for i in range(15):
        # рисуем график для данного числа моделей и переходим к следующей модели (увеличиваем их количество) - от 10 до 25
        draw_plot(features, values, boost, path + " with " + str(initial_models + i) + " models")
        boost.next_model()
