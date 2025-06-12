import sys, os, yaml
import pandas as pd

#%%
class Rooftops_Data():
    ''' Reading data on roof areas and slopes '''
    
    def __init__(self, data_struct_file='pv_energy.yml', district='', show=False):
        # read the data structure for districts
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, data_struct_file)

        with open(file_path, 'r', encoding="utf-8") as f:
            self.data_struct = yaml.safe_load(f)
        # list of districts
        self.district_list = list(self.data_struct.keys())
        # normalize district name
        self.district = district.lower()
        # check if the district is in the list
        if self.district not in self.district_list:
            print(f'Set the `district` argument of the `Rooftops_Data` class from the list {self.district_list}')
            sys.exit(1)
        # build paths to data files
        self.work_dir = self.data_struct[self.district]['work_dir']
        self.rooftop_file = os.path.join(self.work_dir, 
                                         self.data_struct[self.district]['rooftop_file'])
        self.segment_file = os.path.join(self.work_dir, 
                                         self.data_struct[self.district]['segment_file'])

        f_name, f_ext = os.path.splitext(self.rooftop_file)
        self.res_file = f_name + '_pv' + f_ext
        self.res_file_month = f_name + '_pv_month' + f_ext

        # check if files exist
        if not os.path.isfile(self.rooftop_file):
            print(f'File not found: {self.rooftop_file}')
            sys.exit(1)
        if not os.path.isfile(self.segment_file):
            print(f'File not found: {self.segment_file}')
            sys.exit(1)
        # read data
        self.rooftop = pd.read_excel(self.rooftop_file, sheet_name='Sheet1')
        self.segment = pd.read_excel(self.segment_file, sheet_name='Sheet1')
        # show result
        info_msg = [
            f'District: {self.district.upper()}',
            'Roof area and slope data loaded:',
            f'-- data structure in file `{data_struct_file}`',
            f'-- data folder `{self.work_dir}`',
            f'-- rooftop data file `{self.data_struct[self.district]["rooftop_file"]}`',
            f'-- segment data file `{self.data_struct[self.district]["segment_file"]}`',
            f'-- two tables loaded: `rooftop` ({len(self.rooftop)} rows) and `segment` ({len(self.segment)} rows)'
        ]
        if show:
            print(*info_msg, sep='\n')


def sun_energy_stats(cs):
    ''' Solar radiation energy statistics '''
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

# %%
