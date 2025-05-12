#%%
import pandas as pd

''' The script 'pv_energy_month2.py' is similar to 'pv_energy.py',
 but there is a difference that 'pv_energy_month.py' performs calculations for each calendar month,
not for the year as a whole, as 'pv_energy.py' does.

    
    Author: Oleghbond
    Date: 2023-11-26
'''

import pandas as pd
import numpy as np
import os

from pvlib.irradiance import get_total_irradiance

from irrad import get_irradiance
from light_day_percent import days_in_month
from pv_energy_util import Rooftops_Data

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
    'altitude': 40, 
    'name': 'Otzarkoaga', 
    'start': '2023-01-01', 
    'end': '2024-01-01', 
    'freq': '1H',
    'turbidity': True}

''' Вибір району міста '''

rayon = 'otxarkoaga'  # вибрати з ['сихів', 'сигнівка']

''' Дані площ і нахилів дахів двох ділянок м. Львова '''

rtd = Rooftops_Data(rayon=rayon, show=True)
rooftop = rtd.rooftop # таблиця дахів будинків
segment = rtd.segment # таблиця дахів сегментів

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

def pv_hourly(irrad, solpos, segment, rooftop):
    ''' Обчислення середньогодинних енергетичних показників за сегментами і 
        дахами '''
    segment = segment.copy()
    rooftop = rooftop.copy()
    
    ''' Обчислення середньогодинної потужності окремих сегментів `ppv` у Вт 
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
            energy_output = (radiation['poa_global'] * surface * k_eff)
        else:
            energy_output = pd.Series([0.0] * len(irrad), index=irrad.index)
        
        ppv_list.append(energy_output)
    
    ppv = pd.DataFrame(ppv_list, index=segment.index).T
    ppv.columns = segment.index
    
    return ppv


''' Годинні обрахунки по дахах '''

Roofs_hourly = pd.DataFrame()
for month in range(1, 13):
    time_idx = times.month == month
    time_points = times[time_idx]
    ppv = pv_hourly(irrad.loc[time_idx, :], solpos.loc[time_idx, :], segment, rooftop)
    ppv['time'] = time_points
    ppv = ppv.melt(id_vars=['time'], var_name='segm_id', value_name='p_roof')
    ppv = ppv.merge(segment[['build_id']], left_on='segm_id', right_index=True)
    ppv = ppv.merge(rooftop[['census_id']], left_on='build_id', right_index=True)
    Roofs_hourly = pd.concat([Roofs_hourly, ppv])
    print('month - ', month)

'''
# Group by census_id and time
Roofs_hourly = Roofs_hourly.groupby(['census_id', 'time'])['p_roof'].sum().reset_index()

# Зберегти результати
Roofs_hourly.to_csv('pv_generation_hourly.csv', index=False)
print(f'\nРезультат збережено у файлі `pv_generation_hourly.csv`')

'''
## Save each census_id to a separate column
# Group by census_id and time
#Roofs_hourly['time'] = Roofs_hourly['time'].dt.strftime('%d/%m/%Y %H:%M')
Roofs_hourly = Roofs_hourly.groupby(['time', 'census_id'])['p_roof'].sum().unstack(fill_value=0)
#Roofs_hourly.index = pd.to_datetime(Roofs_hourly.index, utc=True).tz_convert(None) #+ pd.Timedelta(hours=0)
#Roofs_hourly.index = pd.to_datetime(Roofs_hourly.index).dt.tz_convert('Etc/GMT-2').tz_localize(None)
Roofs_hourly = (Roofs_hourly*0.75) / 1000
# Hourly generation from pv_power_hourly2.py exides the values obtained by obond file in pv_power_month2.py by 1.33 (33%) if output aggregated on month level

# Зберегти результати
directory = 'data/Otxarkoaga/pv_generation_hourly.csv'
Roofs_hourly.to_csv(directory)
print(f'\nРезультат збережено у файлі {directory}')
#Roofs_hourly.index = pd.to_datetime(Roofs_hourly.index, utc=True).tz_convert(None) + pd.Timedelta(hours=0)

# %%
