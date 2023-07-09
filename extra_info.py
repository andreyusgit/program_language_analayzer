import csv
import re


class RemoveExtraInfo:
    def del_cooments(self, file_name_start: str, file_name_end: str):
        """
        Метод удаляет комментарии и пользовательский вывод из датасета
        :param file_name_start: имя файла вместе с расширением (.csv), в котором нужно удалить лишнюю информацию
        :param file_name_end: имя файла вместе с расширением (.csv), в который сохранится 'чистый' датасет
        """
        # Открытие файла CSV для чтения
        with open(file_name_start, 'r', errors='ignore') as csv_file:
            csv_reader = csv.reader(csv_file)

            # Создание нового файла CSV для записи
            with open(file_name_end, 'w', newline='') as output_file:
                csv_writer = csv.writer(output_file)

                # Обход строк входного файла CSV
                for row in csv_reader:
                    language = row[0]
                    code = row[1]

                    # Удаление комментариев и текста пользователя из кода в зависимости от языка программирования
                    cleaned_code = RemoveExtraInfo._remove_comments_and_quotes(self, code, language)

                    # Запись очищенного кода в новый файл CSV
                    csv_writer.writerow([language, cleaned_code])

    def _remove_comments_and_quotes(self, code, language):
        """
        Функция удаляет комментарии и пользовательский вывод из датасета
        :param code: код для очистки
        :param language: ЯП на котором написан этот код
        """
        start_code = code.splitlines()
        cleaned_code = []
        if language == 'Python':
            # Удаление комментариев и текста пользователя Python с помощью регулярного выражения
            for item in start_code:
                cleaned_code.append(re.sub(r'#.*|\'\'\'.*?\'\'\'|\"\"\".*?\"\"\"', '', item))
        elif language == 'Haskell':
            # Удаление комментариев и текста пользователя Haskell с помощью регулярного выражения
            for item in start_code:
                cleaned_code.append(re.sub(r'--.*', '', item))
        elif language == 'C#' or 'C++' or 'Rust' or 'Java' or 'PHP' or 'JavaScript' or 'C' or 'D':
            # Удаление комментариев и текста пользователя C#, C++, Rust, Java, PHP, JavaScript, C
            for item in start_code:
                cleaned_code.append(re.sub(r'//.*|/\*.*?\*/|".*?"', '', item))
        else:
            # Неизвестный язык программирования, возвращаем исходный код без изменений
            cleaned_code.append(start_code)
        return "\n".join(filter(None, cleaned_code))
