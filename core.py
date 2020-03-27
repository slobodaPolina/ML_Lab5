import numpy as np
import math
INT_MAX = 2e9


# вопрос, состоящий из сравнения с value (больше, меньше) и feature_idx - какое значение сравниваем
class Question:
    def __init__(self, feature_idx, value):
        self.feature_idx = feature_idx
        self.value = value


# слабое дерево (Decision stump)
class Tree:
    def __init__(self, significance, left_cls, right_cls, question):
        self.significance = significance
        self.question = question
        self.left_cls = left_cls
        self.right_cls = right_cls

    # классификация - правый или левый класс
    def classify(self, row):
        clazz = self.left_cls if self._accept(row) else self.right_cls
        return clazz, self.significance

    def _accept(self, row):
        return row[self.question.feature_idx] < self.question.value


class AdaBoost:
    def __init__(self, models_number):
        self.models_number = models_number
        self.class_quantity = 2
        self.forest = []

    # lists of features and values
    def fit(self, X, Y):
        self.features = X
        self.values = Y
        # инициируем веса. значально они все равны и в сумме 1 (то есть по 1/n)
        self.weights = self._init_weights()
        self.indices = np.arange(len(self.values))
        # сразу же запускаем все начальные модели
        for _ in range(self.models_number):
            self.next_model()

    # добавление следующей модели (следующего шага)
    def next_model(self):
        # выбирается вопрос (по какой фиче сравнивать, какой порог)
        q = self._get_question(self.indices)
        # какой класс выбирается при положительных реальных классах
        left, right = self._split_on_classes(self.indices, q)
        significance, indices_with_error = self._calc_significance(self.weights, self.indices, q, left, right)
        # строим простое дерево и добавляем его в свой лес
        self.forest.append(Tree(significance, left, right, q))
        self.weights = self._upd_weights(self.weights, significance, len(self.indices), indices_with_error)
        # обновляем случайные записи для следующего шага
        self.indices = self._chooce_new_indices(self.weights, len(self.indices))

    # предсказываем по массиву фич вероятные классы
    def predict(self, X):
        return np.fromiter(map(lambda x: self.classify(x), X), int)

    def classify(self, x):
        classes_score = np.zeros(2)
        # для классификации смотрим на каждое дерево, суммируем score (некоторые деревья влияют сильнее), смотрим максимальный
        for tree in self.forest:
            clazz, score = tree.classify(x)
            classes_score[1 if clazz == 1 else 0] += score
        return 1 if np.argmax(classes_score) == 1 else -1

    def _chooce_new_indices(self, weights, n):
        indices = np.zeros(n, dtype=int)
        for i in range(n):
            random_value = np.random.uniform(low=0.0, high=1.0)
            for idx in range(n):
                random_value -= weights[idx]
                if random_value <= 0:
                    indices[i] = idx
                    break
        return indices

    def _upd_weights(self, weights, significance, n, indices_with_error):
        for i in range(n):
            v = significance if i in indices_with_error else -significance
            weights[i] = weights[i] * math.exp(v)

        x = sum(weights)
        for i in range(n):
            weights[i] = weights[i] / x
        return weights

    def _init_weights(self):
        a = np.empty(len(self.values))
        a.fill(1 / len(self.values))
        return a

    def _calc_significance(self, weights, indices, q, left, right):
        total_error = 0
        indices_with_error = set()
        # пробегаем по всем записям, определяем для них реальный класс (правый-левый), если не угадали, штраф
        for idx, i in enumerate(indices):
            clazz = self.values[i]
            actual = left if self._accept(self.features[i], q.feature_idx, q.value) else right
            if actual != clazz:
                total_error += weights[idx]
                indices_with_error.add(idx)
        if total_error == 0:
            return np.inf, indices_with_error
        return math.log((1 - total_error) / total_error) / 2, indices_with_error

    def _get_question(self, indices):
        min_gini = INT_MAX

        # перебор всех фич от 0 до n
        for feature_idx in range(len(self.features[0])):
            # случайные строки данных с выбранной фичой (номер строки, значение у нее данной фичи)
            ids = list(map(lambda i: (i, self.features[i][feature_idx]), indices))
            ids = sorted(ids, key=lambda pair: pair[1])
            # в одномерный массив, берем только индексы (у соседей близкие значения фичи)
            ids = np.fromiter(map(lambda pair: pair[0], ids), int)

            left_counters = {}
            right_counters = {}
            # подсчитываем, сколько раз входит какое значение на выбранных данных
            for i in ids:
                curr_cls = self.values[i] - 1
                if not curr_cls in right_counters:
                    left_counters[curr_cls] = 0
                    right_counters[curr_cls] = 0
                right_counters[curr_cls] += 1

            idx = 0
            # перебираем выбранные данные
            while idx < len(ids):
                # вместо того, чтобы просто брать value, немного усредним ее с соседом
                curr_value = self._calc_value(idx, feature_idx, ids)
                # текущий номер записи
                curr_idx = ids[idx]
                while True:
                    curr_class = self.values[curr_idx] - 1
                    # перекладываем из правой кучи в левую
                    right_counters[curr_class] -= 1
                    left_counters[curr_class] += 1
                    # и переходим к следующей записи
                    idx += 1
                    if idx >= len(ids):
                        break
                    curr_idx = ids[idx]
                    # до тех пор, пока не превысим усредненное значение
                    if self.features[curr_idx][feature_idx] >= curr_value:
                        break

                # оценка данного вопроса с помощью вычисления gini (зависит от энтропии, меньше-лучше)
                curr_gini = self._gini(idx, left_counters, len(ids) - idx, right_counters)
                # обновляем min_gini
                if min_gini > curr_gini:
                    min_gini = curr_gini
                    min_gini_value = curr_value
                    min_gini_feature = feature_idx

        return Question(min_gini_feature, min_gini_value)

    def _entropy(self, group, group_quantity):
        if group_quantity == 0:
            return 0

        entropy = 0
        for cls in range(self.class_quantity):
            if cls in group:
                p = group[cls] / group_quantity
                if (p != 0):
                    entropy += p * math.log(p)
        return -entropy

    def _gini(self, left_quantity, left, right_quantity, right):
        entropy_left = self._entropy(left, left_quantity)
        entropy_right = self._entropy(right, right_quantity)
        n = left_quantity + right_quantity
        return left_quantity * entropy_left / n + right_quantity * entropy_right / n

    # по данному вопросу разбиваем отобранные записи, по каждой принимаеся решение, смотрим, чего больше
    def _split_on_classes(self, indices, q):
        left, right = np.zeros(2), np.zeros(2)
        for i in indices:
            # положительный ли класс на самом деле
            clazz = 1 if self.values[i] == 1 else 0
            # TP, TN, FP, FN
            # какой ответ на вопрос
            if self._accept(self.features[i], q.feature_idx, q.value):
                left[clazz] += 1
            else:
                right[clazz] += 1

        # если больше значение в ячейке 1 (положительный на самом деле класс)
        def _is_class(yy):
            return 1 if np.argmax(yy) == 1 else -1

        return _is_class(left), _is_class(right)

    def _accept(self, row, feature, value):
        return row[feature] < value

    # полусумма значений из i и i+1 строк по ids(отсортированному) (если нет соседа, то просто большое число) - позволит немного усреднить
    def _calc_value(self, i, feature_index, ids):
        return INT_MAX if i + 1 == len(ids) else (self.features[ids[i]][feature_index] + self.features[ids[i + 1]][feature_index]) / 2
