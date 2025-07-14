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

from .irrad import get_irradiance
from .light_day_percent import days_in_month
from .pv_energy_util import Rooftops_Data
from .filter import Filter

class PVMonthCalculator:
    """
    Class for calculating monthly PV energy for rooftops.
    """

    def __init__(self, params_file, sheet_name='panel_pars', solar_energy_type='pv', nominal_power_kwp=0.3):
        """
        Initialize parameters and read data from Excel.
        """
        # Read parameters from Excel
        self.params_df = pd.read_excel(params_file, sheet_name=sheet_name, index_col=0)
        self.district = self.params_df.loc['name', solar_energy_type]  # District name
        self.solar_energy_type = solar_energy_type  # 'pv' or 'heat'

        # Assign parameters from DataFrame
        self.panel_pars = {'pv': {}, 'heat': {}}
        self.panel_pars['pv']['k_eff'] = self.params_df.loc[ 'k_eff','pv']
        self.panel_pars['heat']['k_eff'] = self.params_df.loc['k_eff','heat']
        self.panel_pars['pv']['s_panel'] = self.params_df.loc['s_panel','pv']
        self.panel_pars['heat']['s_panel'] = self.params_df.loc['s_panel', 'heat']
        self.panel_pars['pv']['kWp_limit'] = self.params_df.loc['kWp_limit', 'pv'] if not pd.isnull(self.params_df.loc['kWp_limit','pv']) else 0

        self.nominal_power = nominal_power_kwp  # kWp

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

    def pv_monthly(self, irrad, solpos, segment, rooftop):
        """
        Calculate average monthly energy indicators for segments and rooftops.
        """
        segment = segment.copy()
        rooftop = rooftop.copy()

        # Calculate average monthly power for individual segments `ppv` in W for the entire segment area
        ppv_list = []
        for _, r in segment.iterrows():
            if r.slope > 0:
                tilt = r.slope
                surface = r.s_area / np.cos(tilt / 180 * np.pi)  # m²
                az = r.aspect  # aspect - 0 - north, increases clockwise
                n_panel = np.floor(self.k_panel_density.slopy * surface / self.s_panel)
                surface = n_panel * self.s_panel
            else:
                # Prepare flat segments 'aspect' == 0 (for buildings with rooftop.plain_roof == 1).
                # All panels are laid horizontally facing south, raised at a tilt of 35 degrees,
                # so the area remains unchanged.
                tilt = 35
                surface = r.s_area  # m²
                az = 180
                n_panel = np.floor(self.k_panel_density.plain * surface / self.s_panel)
                surface = n_panel * self.s_panel
            if n_panel > 0:
                # Calculate solar radiation on the panel
                radiation = get_total_irradiance(surface_tilt=tilt, surface_azimuth=az,
                                                 solar_zenith=solpos['apparent_zenith'],
                                                 solar_azimuth=solpos['azimuth'],
                                                 dni=irrad.dni, ghi=irrad.ghi, dhi=irrad.dhi)
                energy_output = (radiation['poa_global'] * surface * self.k_eff).mean()
            else:
                energy_output = 0.0

            ppv_list.append({'ppv': energy_output, 's_area2': surface, 'n_panel': n_panel})

        ppv = pd.DataFrame(ppv_list, index=segment.index)
        # Add `ppv` column to the `segment` table
        segment = pd.concat([segment, ppv], axis=1)
        # Remove zero areas, drop old 's_area' column, rename new one
        segment = segment.loc[segment.s_area2 > 0, :]
        segment.drop(columns='s_area', inplace=True)
        segment.rename(columns={'s_area2': 's_area'}, inplace=True)

        # Summarize 'res': areas 's_roof' in m^2 and average monthly power 'p_roof' in W (for the entire roof area) by buildings 'build_id'
        res = []
        for build_id, df in segment.groupby('build_id'):
            s_roof = df.s_area.sum()
            n_panel = df.n_panel.sum()
            p_roof = df.ppv.sum()
            res.append(pd.Series([build_id, s_roof, n_panel, p_roof],
                                 index=['build_id', 's_roof', 'n_panel', 'p_roof']))

        res = pd.DataFrame(res).set_index('build_id')

        # Join columns of 'res' table to the rooftops table
        rooftop = pd.concat([rooftop, res], axis=1)

        # Remove unnecessary columns (commented out for now)
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

    def calculate(self):
        """
        Perform monthly calculations for rooftops and save results.
        """
        Roofs = pd.DataFrame()
        for month in range(1, 13):
            time_idx = self.times.month == month
            time_points = self.times[time_idx]
            _, roofs = self.pv_monthly(self.irrad.loc[time_idx, :], self.solpos.loc[time_idx, :],
                                       self.segment, self.rooftop)
            # Convert from W -> kWh per month
            roofs.p_roof = roofs.p_roof * days_in_month[month - 1] * 24.0 / 1000.0
            # Consider panel density on the roof
            roofs.p_roof = [r.p_roof * (self.k_panel_density.plain if r.plain_roof else self.k_panel_density.slopy)
                            for i, r in roofs.iterrows()]
            roofs = roofs[['building', 's_roof', 'n_panel',
                           'r_area', 'p_roof']]
            roofs['month'] = month
            Roofs = pd.concat([Roofs, roofs])
            print('month - ', month)

        # Reshape table: buildings as rows, months as columns
        Roofs.reset_index(inplace=True)
        Roofs_pv = pd.pivot_table(Roofs, index='build_id', columns='month',
                                  values='p_roof', aggfunc='sum')
        # Annual total for each building
        Roofs_pv['Total, kWh'] = Roofs_pv.sum(axis=1)
        # Annual summary for each building
        Roofs_s_n = pd.pivot_table(Roofs.loc[Roofs.month == 1, :], index='build_id',
                                   values=['s_roof', 'n_panel'], aggfunc='sum')

        # Select and rename columns
        rooftop = self.rooftop[['census_id', 'building', 'plain_roof', 'r_area']]

        # Combine input data and results
        Roofs_pv = pd.concat([rooftop, Roofs_s_n, Roofs_pv], axis=1)
        reorder_cols = [
            'census_id', 'building',
            'plain_roof', 'r_area', 'n_panel', 's_roof',
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'Total, kWh']
        Roofs_pv = Roofs_pv.loc[:, reorder_cols]

        # Summary for all buildings for each month
        cols = list(range(1, 13)) + ['s_roof', 'n_panel', 'r_area', 'Total, kWh']
        Roofs_pv_summary = Roofs_pv[cols].sum().to_frame().T.rename(index={0: 'Total'})
        Roofs_pv = pd.concat([Roofs_pv, Roofs_pv_summary])
        # Calculate installed_kWp for each building
        Roofs_pv['installed_kWp'] = Roofs_pv['n_panel'] * self.nominal_power

        # Save results
        Roofs_pv.to_excel(self.rtd.res_file_month,
                          index_label='build_id',
                          sheet_name=self.district.capitalize())
        print(f'\nResult saved to file `{self.rtd.res_file_month}`')


# Example usage:
# calculator = PVMonthCalculator('input_data_for_unit_testing_file.xlsx')
# calculator.calculate()

# %%
