''' LIGHT_DAY_PERCENT.PY - Розрахунок середньої (за календарний місяць) частки 
    світлого дня, коли світить пряме сонце (щось протилежне до коефіцієнту 
    хмарності) для м. Львова. Повертає датафрейм (таблицю) 'sunshine_duration' 
    з такими стовпчиками:
        'Warsaw', 'Kyiv', 'Bratislava' - загальна кількість сонячних годин у 
            місяці для відповідних міст Європи, які розташовані приблизно на
            широті Львова західніше і східніше; 
        'Lviv' - загальна кількість сонячних годин у місяці для Львова, отримана 
            усерендненням за кожний місяць по трьох містах зазначених вище;
        'day_length' - загальна тривалість денного часу (від сходу до заходу) у 
            годинах за місяць для заданої широти;
        'Lviv_perc' - середня (за календарний місяць) частка світлого дня для 
            м. Львова;
        'month' - індекс ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                          'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Також використана таблиця 'coord':
                Warsaw     Kyiv  Bratislava      Lviv
        coord                                        
        lat    52.2370  48.3794     48.1486  49.82517
        lon    21.0175  31.1656     17.1077  23.96281
'''

import numpy as np
import pandas as pd
import yaml

def daylength(dayOfYear, lat):
    """Computes the length of the day (the time between sunrise and
    sunset) given the day of the year and latitude of the location.
    Function uses the Brock model for the computations.
    For more information see, for example,
    Forsythe et al., "A model comparison for daylength as a
    function of latitude and day of year", Ecological Modelling,
    1995.

    https://gist.github.com/anttilipp/ed3ab35258c7636d87de6499475301ce

    Parameters
    ----------
    dayOfYear : int
        The day of the year. 1 corresponds to 1st of January
        and 365 to 31st December (on a non-leap year).
    lat : float
        Latitude of the location in degrees. Positive values
        for north and negative for south.

    Returns
    -------
    d : float
        Daylength in hours.
    """
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0+dayOfYear)/365.0))
    if -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) <= -1.0:
        return 24.0
    elif -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) >= 1.0:
        return 0.0
    else:
        hourAngle = np.rad2deg(np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))))
        return 2.0*hourAngle/15.0

days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def sum_day_len_month(lat: float) -> list:
    ''' Розрахунок сумарної за календарний місяць тривалості дня у годинах 
        для заданої широти місцевості 'lat'. 
    '''
    dl_df = pd.DataFrame()
    for m, dim in enumerate(days_in_month):
        month = pd.DataFrame({'day_of_month': range(1, dim + 1), 'month': m + 1})
        dl_df = pd.concat([dl_df, month])
        
    dl_df.reset_index(drop=True, inplace=True)
    dl_df['day_of_year'] = dl_df.index + 1
    dl_df['day_length'] = [daylength(doy + 1, lat) for doy in range(365)]
    
    day_len_month = [sum(df.day_length) for m, df in dl_df.groupby('month')]
    
    return day_len_month

# List of cities by sunshine duration - Wikipedia
# https://en.wikipedia.org/wiki/List_of_cities_by_sunshine_duration#daylight_duration
# hour
sunshine_duration = '''
month: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Warsaw: [44.6, 66.5, 139.4, 210.1, 272.4, 288.8, 295.4, 280.2, 193.1, 122.6, 50.6, 33.6]
Kyiv: [31, 57, 124, 180, 279, 270, 310, 248, 210, 155, 60, 31]
Bratislava: [65, 82, 152, 204, 264, 270, 276, 270, 207, 143, 60, 47]
Bilbao: [86,97, 128, 128,160,173,188,179,157,123,93,78]
'''
coord = '''
coord: [lat, lon]
Warsaw: [52.237, 21.0175]
Kyiv: [48.3794, 31.1656]
Bratislava: [48.1486, 17.1077]
Lviv: [49.82517, 23.96281] # Львів, Сигнівка
Bilbao: [43.257928, -2.897791]
'''

coord = yaml.safe_load(coord)
coord = pd.DataFrame(coord).set_index('coord')

sunshine_duration = yaml.safe_load(sunshine_duration)
sunshine_duration = pd.DataFrame(sunshine_duration).set_index('month')

sunshine_duration['Bilbao'] = sunshine_duration.mean(axis=1)
sunshine_duration['day_length'] = sum_day_len_month(coord.Bilbao.lat)

sunshine_duration['Bilbao_perc'] = sunshine_duration.Bilbao / sunshine_duration.day_length

