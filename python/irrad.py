import pandas as pd
from pvlib import solarposition, atmosphere, clearsky, irradiance, location
import matplotlib.pyplot as plt

def get_irradiance(in_pars:[pd.Series, dict]):
    assert isinstance(in_pars, dict) or isinstance(in_pars, pd.Series), \
        "Тип вхідного аргументу: або 'pd.Series', або 'dict'"
    if isinstance(in_pars, dict):
        in_pars = pd.Series(in_pars)

    times = pd.date_range(start=in_pars.start, end=in_pars.end, 
                          freq=in_pars.freq, tz=in_pars.tz)
    solpos = solarposition.get_solarposition(times, in_pars.lat, in_pars.lon)
    
    if not in_pars.turbidity:
        loc = location.Location(in_pars.lat, in_pars.lon, tz=in_pars.tz)
        irrad = loc.get_clearsky(times)  
    else:
        apparent_zenith = solpos['apparent_zenith']
        airmass = atmosphere.get_relative_airmass(apparent_zenith)
        pressure = atmosphere.alt2pres(in_pars.altitude)
        airmass = atmosphere.get_absolute_airmass(airmass, pressure)
        linke_turbidity = clearsky.lookup_linke_turbidity(times, in_pars.lat, in_pars.lon)
        dni_extra = irradiance.get_extra_radiation(times)
        irrad = clearsky.ineichen(apparent_zenith, airmass, linke_turbidity, 
                                  in_pars.altitude, dni_extra)
    
    return irrad, times, solpos

if __name__ == '__main__':

    in_pars = {
        'lat': 49.825171, 
        'lon': 23.962810, 
        'tz': 'Europe/Helsinki', 
        'altitude': 289, 
        'name': 'Львів', 
        'start': '2022-06-01', 
        'end': '2022-06-02', 
        'freq': '1min',
        'turbidity': False}

    irrad, times, solpos = get_irradiance(in_pars)

    plt.figure();
    ax = irrad.plot()
    ax.set_ylabel('Irradiance $W/m^2$');
    ax.set_title(f'Ineichen Clear Sky Model, {in_pars["name"]}');
    ax.legend(loc=2);