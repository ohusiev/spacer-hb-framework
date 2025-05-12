import sys, os, yaml
import pandas as pd


class Rooftops_Data():
    ''' Зчитування даних площ і нахилів дахів '''
    
    def __init__(self, data_struct_file='pv_energy.yml', rayon='', show=False):
        # зчитуємо структуру зберігання даних по районах
        with open(data_struct_file, 'r', encoding = "utf-8") as f:
            self.data_struct = yaml.safe_load(f)
        # список районів
        self.rayon_list = list(self.data_struct.keys())
        # нормалізуємо назву району
        self.rayon = rayon.lower()
        # перевіряємо чи є район у списку
        if self.rayon not in self.rayon_list:
            print(f'Задайте аргумент `rayon` класу `Rooftops_Data` зі списку {self.rayon_list}')
            sys.exit(1)
        # будуємо шляхи до файлів даних
        self.work_dir = self.data_struct[self.rayon]['work_dir']
        self.rooftop_file = os.path.join(self.work_dir, 
                                         self.data_struct[self.rayon]['rooftop_file'])
        self.segment_file = os.path.join(self.work_dir, 
                                         self.data_struct[self.rayon]['segment_file'])

        f_name, f_ext = os.path.splitext(self.rooftop_file)
        self.res_file = f_name + '_pv' + f_ext
        self.res_file_month = f_name + '_pv_month' + f_ext

        # перевіряємо наявність файлів
        if not os.path.isfile(self.rooftop_file):
            print(f'Не знайдено файл: {self.rooftop_file}')
            sys.exit(1)
        if not os.path.isfile(self.segment_file):
            print(f'Не знайдено файл: {self.segment_file}')
            sys.exit(1)
        # зчитуємо дані
        self.rooftop = pd.read_excel(self.rooftop_file, sheet_name='Sheet1')
        self.segment = pd.read_excel(self.segment_file, sheet_name='Sheet1')
        # показати результат
        info_msg = [
            f'Район: {self.rayon.upper()}',
            'Дані площ і нахилів дахів зчитані:',
            f'-- структура зберігання даних у файлі `{data_struct_file}`',
            f'-- фолдер даних `{self.work_dir}`',
            f'-- файл даних дахів `{self.data_struct[self.rayon]["rooftop_file"]}`',
            f'-- файл даних сегментів дахів `{self.data_struct[self.rayon]["segment_file"]}`',
            f'-- зчитано дві таблиці `rooftop` ({len(self.rooftop)} рядків) і `segment` ({len(self.segment)} рядків)'
        ]
        if show:
            print(*info_msg, sep='\n')


def sun_energy_stats(cs):
    ''' Енергетичні показники сонячної радіації '''
    def get_stats(df, period_str):
        descr = df.describe()
        descr = descr.loc[['mean', 'max'], :]
        descr.reset_index(inplace=True, names='stats')
        descr['period'] = period_str
        descr.set_index(['period', 'stats'], inplace=True)
        return descr
    descr_annual = get_stats(cs, 'annual')
    longest_day = cs.loc[(cs.index.month == 6) & (cs.index.day == 22), :]
    descr_summer = get_stats(longest_day, 'summer, daily')
    shortest_day = cs.loc[(cs.index.month == 12) & (cs.index.day == 22), :]
    descr_winter = get_stats(shortest_day, 'winter, daily')
    cs_stats = pd.concat([descr_annual, descr_summer, descr_winter])
    return cs_stats
