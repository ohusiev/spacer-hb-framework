''' Скрипт 'pv_energy_month2.py' аналогічний до 'pv_energy.py', але є відмінність
    'pv_energy_month.py' проводить обчислення для кожного календарного місяця, 
    а не для року в цілому, як то робить 'pv_energy.py'.
    
    Author: Oleghbond
    Date: 2023-11-26
'''
#%%
import pandas as pd
import numpy as np
import os
import time

from pvlib.irradiance import get_total_irradiance

from irrad import get_irradiance
from light_day_percent import days_in_month
from pv_energy_util import Rooftops_Data
from filter import Filter
#%%
''' Параметри енергетичних установок '''

panel_pars = {'pv': {}, 'heat': {}}

# ККД сонячних колекторів
panel_pars['pv']['k_eff'] = 0.15 # 15%

# ККД теплових колекторів
# Основні характеристики та критерії оцінки сонячних колекторів Viessmann
# https://serviceportal.viessmann.ua/articles/osnovni-harakteristiki-ta-kriterii-ocinki-sonacnih-kolektoriv-viessmann)
panel_pars['heat']['k_eff'] = 0.7

# Розмір сонячної панелі: ширина=1,134 м, висота=2,279 м, номінальна потужність 545 Вт
# Сонячна панель JA Solar JAM72S30-545/MR на 545Вт, 41.8В в інтернет магазині електротехніки || AxiomPlus
# https://axiomplus.com.ua/ua/solnechnyie-paneli/product-139848/
panel_pars['pv']['s_panel'] = 0.99 * 1.95 # 1.93 m2
nominal_power = 0.3 # kWp
# limit of the kWp of panels on the roof considering regulatory disadvantages
panel_pars['pv']['kWp_limit'] = 0# kWp # 0 - no limit

# calculate limit number of panels on the roof considering regulatory disadvantages
panel_pars['pv']['n_panel_limit'] = panel_pars['pv']['kWp_limit'] / nominal_power
# calculate limit area for panels on the roof considering regulatory disadvantages
panel_pars['pv']['s_panel_limit'] = panel_pars['pv']['n_panel_limit'] * panel_pars['pv']['s_panel']

# Розмір сонячного колектора: ширина=2,005 м, висота=2,020 м
# СОЛНЕЧНЫЙ ВОДОНАГРЕВАТЕЛЬ БЕЗНАПОРНЫЙ ВАКУУМНЫЙ КОЛЛЕКТОР AC‐VG(L)
# https://solar-tech.com.ua/image/data/Instructions/solar-collectors/Altek/C-VG(L).pdf
panel_pars['heat']['s_panel'] = 2.005 * 2.020 # 4.05 m2

# Щильність заповнення даху панелями
k_panel_density = pd.Series({'plain': 0.75, 'slopy': 0.65})

''' Вибір обрахунку сонячних панелей або сонячних колекторів '''

solar_energy_type = 'pv' # вибрати з ['pv', 'heat']

k_eff = panel_pars[solar_energy_type]['k_eff']
s_panel = panel_pars[solar_energy_type]['s_panel']

print('Тип енергетичних установок:', solar_energy_type.upper())

''' Параметри обчислення сонячної радіації '''

in_pars = {
    'lat': 43.257928,# 43.269135, 43.257928, -2.897791
    'lon': -2.897791, #23.962810, 
    'tz': 'Europe/Helsinki', 
    'altitude': 79, 
    'name': 'Otxarkoaga', 
    'start': '2023-01-01', 
    'end': '2024-01-01', 
    'freq': '30min',
    'turbidity': True}

''' Вибір району міста '''

rayon = 'otxarkoaga'  # вибрати з ['сихів', 'сигнівка']

''' Дані площ і нахилів дахів двох ділянок  '''

rtd = Rooftops_Data(rayon=rayon, show=True)
rooftop = rtd.rooftop # таблиця дахів будинків
segment = rtd.segment # таблиця дахів сегментів


if panel_pars['pv']['kWp_limit'] > 0:
    filter = Filter(segment)
    start_time = time.time()
    print('Filtering segments considering regulatory disadvantages: ', panel_pars['pv']['kWp_limit'], 'kWp')
    segment = filter.calculate_filtered_area(segment, panel_pars, nominal_power=nominal_power, k_panel_density=k_panel_density)
    print(f'Filtering completed with time: {(time.time() - start_time)}')

# Додати до імені результуючого файлу суфікс 'solar_energy_type'
rtd.res_file_month = os.path.splitext(rtd.res_file_month)
rtd.res_file_month = ''.join([rtd.res_file_month[0], '_', solar_energy_type, 
                              rtd.res_file_month[1]])

''' Підготовка сегментів дахів '''

# викреслюємо рядки і стовчики виключень
segment = segment[segment['exception'] != 1] # видалити виключення == 1
rooftop = rooftop[rooftop['exception'] != 1]
rooftop = rooftop.drop(columns='exception')
rooftop.set_index('build_id', inplace=True)

# Видаляємо сегменти, що дивляться на північ і мають ненульовий нахил
segment_idx = ((segment.aspect > 85) & (segment.aspect < 275) & (segment.slope > 0)) | \
              (segment.slope == 0)
segment = segment.loc[segment_idx, :]

''' Обчислення сонячної радіації '''

irrad, times, solpos = get_irradiance(in_pars)
irrad, times, solpos = irrad.iloc[:-1,:], times[:-1], solpos.iloc[:-1,:]

def pv_monthly(irrad, solpos, segment, rooftop):
    ''' Обчислення середньомісячних енергетичних показників за сегментами і 
        дахами '''
    segment = segment.copy()
    rooftop = rooftop.copy()
    
    ''' Обчислення середньомісячної потужності окремих сегментів `ppv` у Вт 
        на всю площу сегмента '''
        
    ppv_list = []
    for _, r in segment.iterrows():
        if r.slope > 0:
            tilt = r.slope
            surface = r.s_area / np.cos(tilt/180*np.pi) # м²
            az = r.aspect # aspect - 0 - північ, збільшується за год. стрілкою
            n_panel = np.floor(k_panel_density.slopy * surface / s_panel)
            surface = n_panel * s_panel
        else:
            ''' Готуємо пласкі сегменти 'aspect' == 0 (для будинків 
            rooftop.plain_roof == 1). Всі панелі викладаємо горизонтально з 
            орієнтацією на південь, піднімаємо з нахилом на 42 град, таким 
            чином площа залишається незмінною '''
            tilt = 35
            surface = r.s_area # м²
            az = 180
            n_panel = np.floor(k_panel_density.plain * surface / s_panel)
            surface = n_panel * s_panel
        if n_panel > 0:
            # Calculate solar radiation on the panel
            radiation = get_total_irradiance(surface_tilt=tilt, surface_azimuth=az,
                                             solar_zenith=solpos['apparent_zenith'],
                                             solar_azimuth=solpos['azimuth'],
                                             dni=irrad.dni, ghi=irrad.ghi, dhi=irrad.dhi)
            energy_output = (radiation['poa_global'] * surface * k_eff).mean()
        else:
            energy_output = 0.0
        
        ppv_list.append({'ppv': energy_output, 's_area2': surface, 'n_panel': n_panel})
    
    ppv = pd.DataFrame(ppv_list, index=segment.index)
    # додаємо стовпчик `ppv` до таблиці `segment`
    segment = pd.concat([segment, ppv], axis=1)
    # видаляємо нульові площі, видаляємо стару колонку 's_area', перейменовуємо нову
    segment = segment.loc[segment.s_area2 > 0, :]
    segment.drop(columns='s_area', inplace=True)
    segment.rename(columns={'s_area2': 's_area'}, inplace=True)
    
    ''' Проводимо підсумки 'res': площі 's_roof' у м^2 і середньомісячної 
        потужності 'p_roof' у Вт (на всю площу даху) за будинками 'build_id' '''
    
    res = []
    for build_id, df in segment.groupby('build_id'):
        s_roof = df.s_area.sum()
        n_panel = df.n_panel.sum()
        p_roof = df.ppv.sum()
        res.append(pd.Series([build_id, s_roof, n_panel, p_roof], 
                             index=['build_id', 's_roof', 'n_panel', 'p_roof']))
    
    res = pd.DataFrame(res).set_index('build_id')
    
    # Приєднуємо стовчики таблиці 'res' до таблиці будинків 'rooftop'
    rooftop = pd.concat([rooftop, res], axis=1)
    #rooftop.reset_index(inplace=True)

    # Видалити зайві стовпчики
    """ 
    segm_cols_to_delete = ['s_cent_x', 's_cent_y', 's_cent_z', 's_poly_xy', 
                           's_xy_WGS84']
    segment.drop(segm_cols_to_delete, axis=1, inplace=True)
    segment.set_index('segm_id', inplace=True)
    
    try:
        roof_cols_to_delete = ['r_cent_x', 'r_cent_y', 'r_cent_z', 'r_poly_xy', 
                               'r_xy_WGS84']
        rooftop.drop(columns=roof_cols_to_delete, inplace=True)
    except:
        roof_cols_to_delete = ['r_cent_x', 'r_cent_y', 'r_cent_z']    
        rooftop.drop(columns=roof_cols_to_delete, inplace=True)
    """
    return segment, rooftop


''' Помісячні обрахунки по дахах '''

Roofs = pd.DataFrame()
for month in range(1,13):
    time_idx = times.month == month
    time_points = times[time_idx]
    _, roofs = pv_monthly(irrad.loc[time_idx, :], solpos.loc[time_idx, :], 
                          segment, rooftop)
    # Перетворюємо з Вт -> у кВт*год за міс
    roofs.p_roof = roofs.p_roof * days_in_month[month - 1] * 24.0 / 1000.0
    # Враховуємо щільність заповнення даху панелями
    roofs.p_roof = [r.p_roof * (k_panel_density.plain if r.plain_roof else k_panel_density.slopy) \
                    for i, r in roofs.iterrows()]
    roofs = roofs[['building', 's_roof', 'n_panel', 
                'r_area', 'p_roof']]#, 'r_x_WGS84', 'r_y_WGS84', 
                #'Link Google Maps']]
    roofs['month'] = month
    #roofs.rename(columns={'r_x_WGS84': 'lon', 'r_y_WGS84': 'lat', 
    #             'Link Google Maps': 'link'}, inplace=True)
    Roofs = pd.concat([Roofs, roofs])
    print('month - ', month)

# Перебудувати таблицю: будинки - рядками, місяці - колонками
Roofs.reset_index(inplace=True)
Roofs_pv = pd.pivot_table(Roofs, index='build_id', columns='month', 
                          values='p_roof', aggfunc='sum')
# підсумок за рік для окремого будинку
Roofs_pv['Total, kWh'] = Roofs_pv.sum(axis=1)
# підсумок за рік для окремого будинку
Roofs_s_n = pd.pivot_table(Roofs.loc[Roofs.month == 1,:], index='build_id', 
                          values=['s_roof', 'n_panel'], aggfunc='sum')

# Вибір і перейменування колонок

rooftop = rooftop[['census_id', 'building', 'plain_roof', 
                'r_area']]#, 'r_x_WGS84', 'r_y_WGS84', 
#                'Link Google Maps']]
""" 
rooftop = rooftop[['building', 'build_str', 'build_num', 'plain_roof', 
                   'r_area', 'r_x_WGS84', 'r_y_WGS84', 'Link Google Maps']]
"""
#rooftop.rename(columns={'r_x_WGS84': 'lon', 'r_y_WGS84': 'lat', 
#                        'Link Google Maps': 'link'}, inplace=True)

# Об'єднати вхідні дані й результати
Roofs_pv = pd.concat([rooftop, Roofs_s_n, Roofs_pv], axis=1)
#Roofs_pv = Roofs_pv.merge(rooftop["census_id"], how="left", left_index=True, right_index=True)
reorder_cols = [
    'census_id','building', 
    'plain_roof', 'r_area', 'n_panel', 's_roof',
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'Total, kWh']
"""
reorder_cols = [
    'building', 'build_str', 'build_num', 'lon', 'lat', 'link', 
    'plain_roof', 'r_area', 'n_panel', 's_roof',
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'Total, kWh']
"""
Roofs_pv = Roofs_pv.loc[:, reorder_cols]

# підсумок для всіх будинків для окремого місяця
cols = list(range(1,13)) + ['s_roof', 'n_panel', 'r_area', 'Total, kWh']
Roofs_pv_summary = Roofs_pv[cols].sum().to_frame().T.rename(index={0: 'Total'})
Roofs_pv = pd.concat([Roofs_pv, Roofs_pv_summary])
#calclate installed_kWp for each building
Roofs_pv['installed_kWp'] = Roofs_pv['n_panel'] * nominal_power
# add colun of census_id from the initial data
# add column of census_id from the initial data



# Зберегти результати
Roofs_pv.to_excel(rtd.res_file_month, 
                  index_label='build_id', 
                  sheet_name=rayon.capitalize())
print(f'\nРезультат збережено у файлі `{rtd.res_file_month}`')


# %%
