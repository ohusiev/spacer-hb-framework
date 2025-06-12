

#%%
#%%
import pandas as pd
import numpy as np
import os
import time

from pvlib.irradiance import get_total_irradiance

from .irrad import get_irradiance
from .light_day_percent import days_in_month
from .pv_energy_util import Rooftops_Data
from .filter import Filter

class PVHourlyCalculator:
    """
    Class for calculating monthly PV energy for rooftops.
    """

    def __init__(self, params_file, sheet_name='panel_pars', solar_energy_type='pv'):
        """
        Initialize parameters and read data from Excel.
        """
        # Read parameters from Excel
        self.params_df = pd.read_excel(params_file, sheet_name=sheet_name, index_col=0)
        self.district = self.params_df.loc['name', 'pv']  # District name
        self.solar_energy_type = solar_energy_type  # 'pv' or 'heat'

        # Assign parameters from DataFrame
        self.panel_pars = {'pv': {}, 'heat': {}}
        self.panel_pars['pv']['k_eff'] = self.params_df.loc[ 'k_eff','pv']
        self.panel_pars['heat']['k_eff'] = self.params_df.loc['k_eff','heat']
        self.panel_pars['pv']['s_panel'] = self.params_df.loc['s_panel','pv']
        self.panel_pars['heat']['s_panel'] = self.params_df.loc['s_panel', 'heat']
        self.panel_pars['pv']['kWp_limit'] = self.params_df.loc['kWp_limit', 'pv'] if not pd.isnull(self.params_df.loc['kWp_limit','pv']) else 0

        self.nominal_power = 0.3  # kWp

        # Calculate limit number of panels on the roof considering regulatory disadvantages
        self.panel_pars['pv']['n_panel_limit'] = self.panel_pars['pv']['kWp_limit'] / self.nominal_power if self.panel_pars['pv']['kWp_limit'] else 0
        # Calculate limit area for panels on the roof considering regulatory disadvantages
        self.panel_pars['pv']['s_panel_limit'] = self.panel_pars['pv']['n_panel_limit'] * self.panel_pars['pv']['s_panel']

        # Panel density on the roof
        self.k_panel_density = pd.Series({'plain': self.params_df.loc['k_panel_density_plain', 'pv'], 'slopy': self.params_df.loc['k_panel_density_slopy', 'pv']})

        # Select calculation type: PV panels or solar collectors
        self.k_eff = self.panel_pars[self.solar_energy_type]['k_eff']
        self.s_panel = self.panel_pars[self.solar_energy_type]['s_panel']

        print('Type of energy installation:', self.solar_energy_type.upper())

        # Solar radiation calculation parameters
        self.in_pars = {
            'lat': self.params_df.loc['lat', 'pv'],
            'lon': self.params_df.loc['lon', 'pv'],
            'altitude': self.params_df.loc['altitude', 'pv'],
            'name': self.params_df.loc['name', 'pv'],
            'tz': 'Europe/Helsinki',
            'start': '2023-01-01',
            'end': '2024-01-01',
            'freq': '30min',
            'turbidity': True
        }

        # Rooftop and segment data for two districts
        self.rtd = Rooftops_Data(district=self.district, show=True)
        self.rooftop = self.rtd.rooftop  # rooftops table
        self.segment = self.rtd.segment  # segments table

        if self.panel_pars['pv']['kWp_limit'] > 0:
            filter = Filter(self.segment)
            start_time = time.time()
            print('Filtering segments considering regulatory disadvantages: ', self.panel_pars['pv']['kWp_limit'], 'kWp')
            self.segment = filter.calculate_filtered_area(self.segment, self.panel_pars, nominal_power=self.nominal_power, k_panel_density=self.k_panel_density)
            print(f'Filtering completed with time: {(time.time() - start_time)}')

        # Add suffix to result file name for 'solar_energy_type'
        self.rtd.res_file_month = os.path.splitext(self.rtd.res_file_month)
        self.rtd.res_file_month = ''.join([self.rtd.res_file_month[0], '_', self.solar_energy_type, self.rtd.res_file_month[1]])

        # Prepare rooftop and segment data
        self.prepare_segments()

        # Calculate solar irradiance
        self.irrad, self.times, self.solpos = get_irradiance(self.in_pars)
        self.irrad, self.times, self.solpos = self.irrad.iloc[:-1, :], self.times[:-1], self.solpos.iloc[:-1, :]

    def prepare_segments(self):
        """
        Prepare rooftop and segment data: remove exceptions and filter segments.
        """
        # Remove rows and columns with exceptions
        self.segment = self.segment[self.segment['exception'] != 1]  # remove exceptions == 1
        self.rooftop = self.rooftop[self.rooftop['exception'] != 1]
        self.rooftop = self.rooftop.drop(columns='exception')
        self.rooftop.set_index('build_id', inplace=True)

        # Remove segments facing north with non-zero slope
        segment_idx = ((self.segment.aspect > 85) & (self.segment.aspect < 275) & (self.segment.slope > 0)) | \
                      (self.segment.slope == 0)
        self.segment = self.segment.loc[segment_idx, :]

    def pv_hourly(self, irrad, solpos, segment, rooftop):
        """
        Calculate average hourly energy indicators by segments and rooftops.
        Returns a DataFrame with hourly PV power per segment.
        """
        segment = segment.copy()
        rooftop = rooftop.copy()

        ppv_list = []
        for _, r in segment.iterrows():
            if r.slope > 0:
                tilt = r.slope
                surface = r.s_area / np.cos(np.deg2rad(tilt))  # m²
                az = r.aspect  # aspect - 0 - north, increases clockwise
                n_panel = np.floor(self.k_panel_density.slopy * surface / self.s_panel)
                surface = n_panel * self.s_panel
            else:
                # Prepare flat segments 'aspect' == 0 (for buildings with rooftop.plain_roof == 1).
                # All panels are placed horizontally facing south, tilted at 35 degrees,
                # so the area remains unchanged.
                tilt = 35
                surface = r.s_area  # m²
                az = 180
                n_panel = np.floor(self.k_panel_density.plain * surface / self.s_panel)
                surface = n_panel * self.s_panel

            if n_panel > 0:
                # Calculate solar radiation on the panel
                radiation = get_total_irradiance(
                    surface_tilt=tilt,
                    surface_azimuth=az,
                    solar_zenith=solpos['apparent_zenith'],
                    solar_azimuth=solpos['azimuth'],
                    dni=irrad.dni,
                    ghi=irrad.ghi,
                    dhi=irrad.dhi
                )
                energy_output = (radiation['poa_global'] * surface * self.k_eff)
            else:
                energy_output = pd.Series([0.0] * len(irrad), index=irrad.index)

            ppv_list.append(energy_output)

        ppv = pd.DataFrame(ppv_list, index=segment.index).T
        ppv.columns = segment.index

        return ppv

    def calculate_hourly_roofs(self):
        """
        Perform hourly calculations for rooftops and save results.
        """
        Roofs_hourly = pd.DataFrame()
        for month in range(1, 13):
            time_idx = self.times.month == month
            time_points = self.times[time_idx]
            ppv = self.pv_hourly(self.irrad.loc[time_idx, :], self.solpos.loc[time_idx, :], self.segment, self.rooftop)
            ppv['time'] = time_points
            ppv = ppv.melt(id_vars=['time'], var_name='segm_id', value_name='p_roof')
            ppv = ppv.merge(self.segment[['build_id']], left_on='segm_id', right_index=True)
            ppv = ppv.merge(self.rooftop[['census_id']], left_on='build_id', right_index=True)
            Roofs_hourly = pd.concat([Roofs_hourly, ppv])
            print('month - ', month)

        # Group by census_id and time
        Roofs_hourly = Roofs_hourly.groupby(['census_id', 'time'])['p_roof'].sum().reset_index()

        # Save results
        Roofs_hourly.to_csv('pv_generation_hourly.csv', index=False)
        print(f'\nResult saved to file `pv_generation_hourly.csv`')

        # Save each census_id to a separate column
        Roofs_hourly = Roofs_hourly.groupby(['time', 'census_id'])['p_roof'].sum().unstack(fill_value=0)

        # Transform to kW and apply 75% safety factor
        Roofs_hourly = (Roofs_hourly * 0.75) / 1000
        print('\nPower values converted to kW and multiplied by 0.75 (safety factor).')

        # Save results
        directory = 'data/04_energy_consumption_profiles/pv_generation_hourly.csv'
        Roofs_hourly.to_csv(directory)
        print(f'\nResult saved to file {directory}')

#%%
# Example usage:
# calculator = PVMonthCalculator('input_data_for_unit_testing_file.xlsx')
# calculator.calculate_hourly_roofs()
if __name__ == "__main__":
    # Example usage
    params_file = 'input_data_for_unit_testing_file.xlsx'  # Replace with your actual file path
    calculator = PVHourlyCalculator(params_file)
    calculator.calculate_hourly_roofs()

# %%
