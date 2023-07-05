import csv
import os
import chardet
from extra_info import RemoveExtraInfo


class ConverterToCsv:

    def __init__(self, path_to_folder: str, res_file_name: str):
        self.folder_path = path_to_folder
        self.file_name = res_file_name
        self.abs_path = os.path.dirname(os.path.abspath(__file__))

    def convert_to_csv(self, remove_extra_info: bool = True, del_full_file: bool = True):
        """
        Метод конвертирует датасет из папок в таблицу .csv
        :param remove_extra_info: нужно ли удалять комментарии и пользовательский вывод из файла
        :param del_full_file: нужно ли удалять исходный файл после очистки 'лишней' информации
        """
        files = []
        column_names = ['language', 'code']
        for folder_name, subfolders, filenames in os.walk(self.abs_path + self.folder_path):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                files.append((folder_name, file_path))
        csv_file = open(f'converted_{self.file_name}_full.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file, escapechar='\\')
        csv_writer.writerow(column_names)

        for folder_name, file_path in files:
            parent_folder = os.path.basename(os.path.dirname(file_path))
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                detected_encoding = chardet.detect(raw_data)['encoding']

            with open(file_path, 'r', encoding=detected_encoding) as file:
                file_contents = file.read()
                if parent_folder != "test" and parent_folder != "train":
                    csv_writer.writerow([parent_folder, file_contents])

        csv_file.close()

        if remove_extra_info:
            remover = RemoveExtraInfo()
            remover.del_cooments(f'converted_{self.file_name}_full.csv',
                                 f'converted_{self.file_name}.csv')
            if del_full_file:
                os.remove(f'{self.abs_path}/converted_{self.file_name}_full.csv')


if __name__ == '__main__':
    folder_path = '/data/test'
    converter = ConverterToCsv(folder_path, "test")
    converter.convert_to_csv()





