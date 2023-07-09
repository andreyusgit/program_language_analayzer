import joblib
import pandas as pd
from colorama import Back, Style
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk import word_tokenize
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV


class SupportMethods:
    def tokenize_sentence(self, sentence: str):
        """
        Функция производит токенизацию текста, отправленного в него
        :param sentence: текст для токенизации
        """
        tokens = word_tokenize(sentence)
        return tokens

    def get_data(self):
        """
        Функция возвращает два словаря, один для тренировки, а другой для тестирования модели
        """
        train_data = pd.read_csv('converted_train.csv')
        train_data["code"].fillna("", inplace=True)
        test_data = pd.read_csv('converted_test.csv')
        test_data["code"].fillna("", inplace=True)
        return train_data, test_data


class ExperimentModel:

    def __init__(self):
        self.model_pipeline = None
        self.support = SupportMethods()
        self.train_df, self.test_df = SupportMethods.get_data(self.support)

    def learn_model(self, show_quantity_of_data: bool = False):
        """
        Метод обучает экспериментальною модель методом логистической регрессии
        :param show_quantity_of_data: нужно ли выводить статистику по полученным данным для обучения
        """
        if show_quantity_of_data:
            print("Количество примеров кода для каждого языка в тренировочной выборке:")
            print(self.train_df["language"].value_counts())

        self.model_pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(tokenizer=lambda x: SupportMethods.tokenize_sentence(self.support, x),
                                           token_pattern=None)),
            ("model", LogisticRegression(random_state=0))
        ]
        )
        self.model_pipeline.fit(self.train_df['code'], self.train_df['language'])

    def result_of_learning(self, show_testing: bool = True):
        """
        Метод показывает результаты обученной модели на тестовых данных
        :param show_testing: нужно ли выводить подробные результаты тестирования
        """
        print('\nПОЛУЧЕННЫЕ РЕЗУЛЬТАТЫ:\n')

        y_true = self.test_df["language"]
        y_pred = self.model_pipeline.predict(self.test_df["code"])

        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)

        print("Precision = " + str(precision * 100) + "%")
        print("Recall = " + str(recall * 100) + "%")
        print("Accuracy = " + str(accuracy * 100) + "%")

        if show_testing:
            print("\nВывод тестовых результатов:\n")
            for item in range(len(self.test_df['language'])):
                ans = self.test_df['language'][item]
                res = self.model_pipeline.predict([self.test_df['code'][item]])[0]
                color = Back.GREEN if ans == res else Back.RED
                print(color + f"Ожидается: {self.test_df['language'][item]}; "
                              f"Получено: {self.model_pipeline.predict([self.test_df['code'][item]])[0]}"
                      + Style.RESET_ALL)

    def try_on_code(self, path_to_file: str):
        """
        Метод для ручной проверки работоспособности классификатора
        :param path_to_file: путь до файла с кодом (файл должен лежать в проекте)
        """
        print('\n\nРЕЗУЛЬТАТ ПРОИЗВОЛЬНОГО КОДА:')
        try:
            with open(path_to_file, 'r') as file:
                text = file.read()
        except FileNotFoundError:
            print(f"Файл '{path_to_file}' не найден.")
            return
        except IOError as e:
            print(f"Ошибка ввода-вывода: {e}")
            return
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return

        print("Experimental Logistic Regression answer: " + self.model_pipeline.predict([text])[0])


class FinalModel:
    def __init__(self):
        self.support = SupportMethods()
        self.train_df, self.test_df = SupportMethods.get_data(self.support)
        self.model_LR_pipeline = None
        self.model_SVC_pipeline = None
        self.model_SGD_pipeline = None
        self._update_models("all", update_params=False)

    def _update_models(self, model_type: str, update_params: bool = True, new_params: list = None):
        """
        Внутренний метод обновляющий лучшие параметры для только что обученных моделей
        :param model_type: модель для которой нужно обновить параметры. Принимает параметры:
        LR - Logistic Regression
        SVC - Support Vector Machines
        SGD - Gradient Descent
        all - all models
        :param new_params: список оптимальных гиперпараметров
        """
        if model_type == "LR" or model_type == "all":
            if update_params:
                joblib.dump(new_params.pop(0), 'best_params/best_params_LR.pkl')

            self.model_LR_pipeline = Pipeline([
                ("vectorizer", TfidfVectorizer(tokenizer=lambda x: SupportMethods.tokenize_sentence(self.support, x),
                                               token_pattern=None)),
                ("model", LogisticRegression(random_state=0, max_iter=1000,
                                             **joblib.load('best_params/best_params_LR.pkl')))
            ]
            )
            self.model_LR_pipeline.fit(self.train_df['code'], self.train_df['language'])

        if model_type == "SVC" or model_type == "all":
            if update_params:
                joblib.dump(new_params.pop(0), 'best_params/best_params_SVC.pkl')

            self.model_SVC_pipeline = Pipeline([
                ("vectorizer", TfidfVectorizer(tokenizer=lambda x: SupportMethods.tokenize_sentence(self.support, x),
                                               token_pattern=None)),
                ("model", SVC(random_state=0, **joblib.load('best_params/best_params_SVC.pkl')))
            ]
            )
            self.model_SVC_pipeline.fit(self.train_df['code'], self.train_df['language'])

        if model_type == "SGD" or model_type == "all":
            if update_params:
                joblib.dump(new_params.pop(0), 'best_params/best_params_SGD.pkl')

            self.model_SGD_pipeline = Pipeline([
                ("vectorizer", TfidfVectorizer(tokenizer=lambda x: SupportMethods.tokenize_sentence(self.support, x),
                                               token_pattern=None)),
                ("model", SGDClassifier(random_state=0, eta0=0.1, **joblib.load('best_params/best_params_SGD.pkl')))
            ]
            )
            self.model_SGD_pipeline.fit(self.train_df['code'], self.train_df['language'])

    def model_stats(self, model_type: str = "all"):
        """
        Метод выводящий метрики актуальных моделей
        :param model_type: модель для которой нужно обновить параметры. Принимает параметры:
        LR - Logistic Regression
        SVC - Support Vector Machines
        SGD - Gradient Descent
        all - all models
        """
        if model_type != ("LR" and "SVC" and "SGD" and "all"):
            print("Переданный тип модели не поддерживается")
            return

        print('\n\n\nМЕТРИКИ ОБУЧЕННЫХ МОДЕЛЕЙ:')

        if model_type == "LR" or model_type == "all":
            print('\nЛОГИСТИЧЕСКАЯ РЕГРЕСИЯ\n')

            y_true = self.test_df["language"]
            y_pred = self.model_LR_pipeline.predict(self.test_df["code"])

            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            accuracy = accuracy_score(y_true, y_pred)

            print("Precision = " + str(int(precision * 100)) + "%")
            print("Recall = " + str(int(recall * 100)) + "%")
            print("Accuracy = " + str(int(accuracy * 100)) + "%")

        if model_type == "SVC" or model_type == "all":
            print('\nМЕТОД ОПОРНЫХ ВЕКТОРОВ\n')

            y_true = self.test_df["language"]
            y_pred = self.model_SVC_pipeline.predict(self.test_df["code"])

            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            accuracy = accuracy_score(y_true, y_pred)

            print("Precision = " + str(int(precision * 100)) + "%")
            print("Recall = " + str(int(recall * 100)) + "%")
            print("Accuracy = " + str(int(accuracy * 100)) + "%")

        if model_type == "SGD" or model_type == "all":
            print('\nМЕТОД ГРАДИЕНТНОГО СПУСКА\n')

            y_true = self.test_df["language"]
            y_pred = self.model_SGD_pipeline.predict(self.test_df["code"])

            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            accuracy = accuracy_score(y_true, y_pred)

            print("Precision = " + str(int(precision * 100)) + "%")
            print("Recall = " + str(int(recall * 100)) + "%")
            print("Accuracy = " + str(int(accuracy * 100)) + "%")

    def learn_model(self, show_quantity_of_data: bool = False, model_type: str = 'all', param_grid: object = None):
        """
        Метод обучает модель/модели по датасету полученному на этапе инициализации класса и сохраняет
        оптимальные параметры в отдельные файлы для каждой модели, далее модели создаются по параметрам из этих файлов
        :param show_quantity_of_data: нужно ли выводить статистику по полученным данным для обучения
        :param model_type: модель, которую нужно переобучить. Принимает параметры:
        LR - Logistic Regression
        SVC - Support Vector Machines
        SGD - Gradient Descent
        all - all models
        :param param_grid: список гиперпараметров для обучения моделей, применим только при обучении моделей
        по отдельности (НЕ ПРИМЕНИМ с model_type="all")
        """
        if show_quantity_of_data:
            print("Количество примеров кода для каждого языка в тренировочной выборке:")
            print(self.train_df["language"].value_counts())
        models = [LogisticRegression(random_state=0, max_iter=1000),
                  SVC(random_state=0),
                  SGDClassifier(random_state=0, eta0=0.1)]
        param_grids = [
            {'C': [0.1, 1, 10], 'solver': ['liblinear', 'sag', 'saga']},
            {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']},
            {'alpha': [0.0001, 0.001, 0.01], 'learning_rate': ['constant', 'optimal', 'invscaling'],
             'penalty': ['l1', 'l2', 'elasticnet']}]
        best_params = []
        if model_type == "LR":
            del models[2]
            del models[1]
            del param_grids[2]
            del param_grids[1]

        elif model_type == "SVC":
            del models[2]
            del models[0]
            del param_grids[2]
            del param_grids[0]

        elif model_type == "SGD":
            del models[1]
            del models[0]
            del param_grids[1]
            del param_grids[0]

        elif model_type == "all":
            pass
        else:
            print("Неверный тип передаваемой модели")
            return

        if param_grid is not None and model_type == ("LR" or "SVC" or "SGD"):
            param_grids = param_grid
        elif param_grid is not None:
            print("Некорректная передача гиперпараметров, см. описание метода")
            return

        print('\n\n\nОБУЧЕНИЕ МОДЕЛЕЙ\n\n\n')
        index = 0
        for model in models:
            try:
                grid_pipeline = Pipeline([
                    (
                        "vectorizer",
                        TfidfVectorizer(tokenizer=lambda x: SupportMethods.tokenize_sentence(self.support, x),
                                        token_pattern=None)),
                    ("model",
                     GridSearchCV(
                         model,
                         param_grids[index],
                         cv=5,
                         verbose=4
                     )
                     )
                ])
                index += 1
                grid_pipeline.fit(self.train_df['code'], self.train_df['language'])
                best_params.append(grid_pipeline["model"].best_params_)
            except ValueError as e:
                print(f"Ошибка значения: {e}")
                return
            except Exception as e:
                print(f"Произошла ошибка: {e}")
                return
        self._update_models(model_type, new_params=best_params)

    def try_on_code(self, path_to_file: str, model_type: str = 'all'):
        """
        Метод для ручной проверки работоспособности классификатора
        :param path_to_file: путь до файла с кодом (файл должен лежать в проекте)
        :param model_type: модель на которой проверять классификатор. Принимает параметры:
        LR - Logistic Regression
        SVC - Support Vector Machines
        SGD - Gradient Descent
        all - all models
        """
        print('\n\nРЕЗУЛЬТАТ ПРОИЗВОЛЬНОГО КОДА:\n')
        try:
            with open(path_to_file, 'r') as file:
                text = file.read()
        except FileNotFoundError:
            print(f"Файл '{path_to_file}' не найден.")
            return
        except IOError as e:
            print(f"Ошибка ввода-вывода: {e}")
            return
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return

        if model_type != ("LR" and "SVC" and "SGD" and "all"):
            print("Переданный тип модели не поддерживается")
            return

        if model_type == "LR" or model_type == "all":
            print("Logistic Regression answer: " + self.model_LR_pipeline.predict([text])[0])

        if model_type == "SVC" or model_type == "all":
            print("Support Vector Machines answer: " + self.model_SVC_pipeline.predict([text])[0])

        if model_type == "SGD" or model_type == "all":
            print("Gradient Descent answer: " + self.model_SGD_pipeline.predict([text])[0])

    def test_all_models(self):
        """
        Метод тестирует все модели на тестовом датасете
        """
        print("\nВывод тестовых результатов:\n")
        models = [self.model_LR_pipeline, self.model_SVC_pipeline, self.model_SGD_pipeline]
        models_names = ["Logistic Regression", "Support Vector Machines", "Gradient Descent"]
        index = 0
        for model in models:
            print(f'\nTests of {models_names[index]}:\n')
            index += 1
            for item in range(len(self.test_df['language'])):
                ans = self.test_df['language'][item]
                res = self.model_LR_pipeline.predict([self.test_df['code'][item]])[0]
                color = Back.GREEN if ans == res else Back.RED
                print(color + f"Ожидается: {self.test_df['language'][item]}; "
                              f"Получено: {model.predict([self.test_df['code'][item]])[0]}" +
                      Style.RESET_ALL)


# тесты для экспериментальной модели
def learn_exp_model():
    exp = ExperimentModel()
    exp.learn_model()
    exp.result_of_learning()


def exp_try_on_code():
    exp = ExperimentModel()
    exp.learn_model()
    exp.try_on_code("test_code.txt")


# тесты для финальной модели
def learn_all_models():
    final = FinalModel()
    final.learn_model()
    final.model_stats()


def bad_train_test():
    final = FinalModel()
    final.learn_model(model_type="LR")
    final.model_stats("LR")
    print("\n\nРезультат хорошо обученной модели:\n\n")
    final.try_on_code("test_code.txt", model_type="LR")
    user_param_grid = [
        {'C': [0.001, 0.005, 0.01], 'solver': ['sag', 'saga']}]
    final.learn_model(model_type="LR", param_grid=user_param_grid)
    final.model_stats("LR")
    print("\n\nРезультат плохо обученной модели:\n\n")
    final.try_on_code("test_code.txt", model_type="LR")


def stats_of_final_model():
    final = FinalModel()
    final.model_stats()


def tests_of_final_model():
    final = FinalModel()
    final.test_all_models()


if __name__ == '__main__':
    stats_of_final_model()
    tests_of_final_model()
