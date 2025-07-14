#%%
import pandas as pd
import geopandas as gpd
import os
import numpy as np

class SelfConsumptionEstimator:
    def __init__(self, df_pv, work_dir, district, pv_pct=0.25):
        # INPUT FILES:
        self.district = district.lower()
        self.work_dir = work_dir #self.data_struct[self.district]['work_dir']
        self.pv_pct = pv_pct
        self.no_pv_pct = 1 - pv_pct

        # Load the dataframes of aggregated consumption profiles
        self.aggregated_profiles_file_path = 'data/04_energy_consumption_profiles/'
        if __name__ == "__main__":
            self.aggregated_profiles_file_path = os.path.join(work_dir, self.aggregated_profiles_file_path)
            #self.work_dir = os.path.join(temp_path, self.work_dir)


        self.df_aggregated_profiles = pd.read_csv(
            os.path.join(self.aggregated_profiles_file_path, f'dwell_share_{self.pv_pct}/04_2_aggregated_1h_profiles_with_pv_dwell_share_{self.pv_pct}.csv'),
            parse_dates=['Time']
        )
        self.df_aggregated_profiles.set_index('Time', inplace=True)

        self.df_aggregated_profiles_no_pv = pd.read_csv(
            os.path.join(self.aggregated_profiles_file_path, f'dwell_share_{self.pv_pct}/04_2_aggregated_1h_profiles_no_pv_dwell_share_{self.no_pv_pct}.csv'),
            parse_dates=['Time']
        )
        self.df_aggregated_profiles_no_pv.set_index('Time', inplace=True)

        # Load the pv generation aggregated profile
        self.pv_df = pd.read_excel(os.path.join(self.work_dir, "data","02_footprint_r_area_wb_rooftop_analysis_pv_month_pv.xlsx")) if __name__ =="__main__" else df_pv
        self.pv_df_hourly = pd.read_csv(os.path.join(work_dir,"data", "04_energy_consumption_profiles", "pv_generation_hourly.csv") if __name__ =="__main__" else 'data/04_energy_consumption_profiles/pv_generation_hourly.csv' )

        # Reassigning the time range for the pv generation data to be aligned with the aggregated profiles
        time_range = pd.date_range(start='2021-01-01 00:00:00', end='2021-12-31 23:00:00', freq='H')
        self.pv_df_hourly['Time'] = time_range
        self.pv_df_hourly.set_index('Time', inplace=True)

        # Ensure output directory exists
        self.output_dir = f"data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}/self_cons_estimation_files"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def time_alignment(self):
        # TIME ALIGNMENT
        # Standardize the time format for all DataFrames
        self.df_aggregated_profiles['Time'] = self.df_aggregated_profiles['Time'].str.replace('2021', '2023')
        self.df_aggregated_profiles['Time'] = pd.to_datetime(self.df_aggregated_profiles['Time'])
        self.df_aggregated_profiles.set_index('Time', inplace=True)

        self.df_aggregated_profiles_no_pv['Time'] = self.df_aggregated_profiles_no_pv['Time'].str.replace('2021', '2023')
        self.df_aggregated_profiles_no_pv['Time'] = pd.to_datetime(self.df_aggregated_profiles_no_pv['Time'])
        self.df_aggregated_profiles_no_pv.set_index('Time', inplace=True)

    @staticmethod
    def aggregate_by_month_and_season(df):
        # Function to aggregate energy consumption profiles by month and season
        df['month'] = df.index.month

        def get_season(date):
            month = date.month
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            elif month in [9, 10, 11]:
                return 'Autumn'

        df['season'] = df.index.map(get_season)
        monthly_aggregation = df.groupby('month').sum()
        seasonal_aggregation = df.groupby('season').sum().drop(columns=['month'])
        return monthly_aggregation, seasonal_aggregation

    @staticmethod
    def calc_dir_self_consumption(df_generation, df_pv_consumption, df_no_pv_consumption, pv_pct):
        # Function to calculate self-consumption percentage

        # Calculate the hourly self-consumed energy by taking the minimum of generation and consumption
        #df_generation = (df_generation)
        df_self_consumed_hourly = np.minimum(df_generation, df_pv_consumption)
        df_self_cons_pct_hourly = (df_self_consumed_hourly / df_generation)

        # Aggregate the hourly self-consumption percentage to monthly by averaging each month
        df_self_cons_pct_monthly = (df_self_consumed_hourly.resample('M').sum() / df_generation.resample('M').sum()).round(4)
        df_self_cons_pct_monthly = df_self_cons_pct_monthly.T
        df_self_cons_pct_monthly.columns = [f'self_m{month}' for month in df_self_cons_pct_monthly.columns.month]

        # Calculate residual generation and load within the energy community (EC)
        df_residual_generation = df_generation - df_self_consumed_hourly
        df_residual_consumption = df_pv_consumption - df_self_consumed_hourly
        df_residual_consumption[df_residual_consumption < 0] = 0

        df_self_cons_pct_hourly.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{pv_pct}/self_cons_estimation_files/04_1_self_cons_pct_hourly.csv')
        df_residual_generation.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{pv_pct}/self_cons_estimation_files/04_2_pv_residual_generation.csv')
        df_self_consumed_hourly.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{pv_pct}/self_cons_estimation_files/04_1_self_consumed_hourly.csv')
        df_residual_consumption.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{pv_pct}/self_cons_estimation_files/04_2_pv_residual_load.csv')

        # Calculate the coverage of consumption without PV
        no_pv_ec_cov_consumption = df_residual_generation.where(df_residual_generation <= df_no_pv_consumption, df_no_pv_consumption)
        no_pv_ec_resid_generation = df_residual_generation - no_pv_ec_cov_consumption
        no_pv_ec_resid_generation.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{pv_pct}/self_cons_estimation_files/04_3_no_pv_ec_resid_generation.csv')

        no_pv_resid_consumption = df_no_pv_consumption - no_pv_ec_cov_consumption
        no_pv_resid_consumption.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{pv_pct}/self_cons_estimation_files/04_4_no_pv_resid_consumption.csv')

        # Calculate the coverage of consumption with PV
        pv_ec_cov_consumption = no_pv_ec_resid_generation.where(no_pv_ec_resid_generation <= df_residual_consumption, df_residual_consumption)

        # Calculate the hourly coverage percentage without PV
        df_cov_pct_no_pv_hourly = (no_pv_ec_cov_consumption / df_generation)
        df_cov_pct_no_pv_hourly = df_cov_pct_no_pv_hourly.fillna(0)
        df_cov_pct_no_pv_hourly.index = pd.to_datetime(df_cov_pct_no_pv_hourly.index, format='%d/%m/%Y %H:%M')

        # Aggregate the hourly coverage percentage without PV to monthly by averaging each month
        df_cov_pct_no_pv_monthly = no_pv_ec_cov_consumption.resample('M').sum() / df_generation.resample('M').sum()
        df_cov_pct_no_pv_monthly = df_cov_pct_no_pv_monthly.T
        df_cov_pct_no_pv_monthly.columns = [f'no_pv_cov_m{month}' for month in df_cov_pct_no_pv_monthly.columns.month]

        # Calculate the hourly coverage percentage with PV
        df_cov_pct_pv_hourly = (pv_ec_cov_consumption / df_generation)
        df_cov_pct_pv_hourly = df_cov_pct_pv_hourly.fillna(0)
        df_cov_pct_pv_hourly.index = pd.to_datetime(df_cov_pct_pv_hourly.index, format='%d/%m/%Y %H:%M')

        # Aggregate the hourly coverage percentage with PV to monthly by averaging each month
        df_cov_pct_pv_monthly = pv_ec_cov_consumption.resample('M').sum() / df_generation.resample('M').sum()
        df_cov_pct_pv_monthly = df_cov_pct_pv_monthly.T
        df_cov_pct_pv_monthly.columns = [f'with_pv_cov_m{month}' for month in df_cov_pct_pv_monthly.columns.month]

        return df_self_cons_pct_monthly, df_cov_pct_no_pv_monthly, df_cov_pct_pv_monthly

    def get_matching_columns_and_index(self):
        matching_columns = self.pv_df_hourly.columns.intersection(self.df_aggregated_profiles.columns)
        matching_index = self.pv_df_hourly.index.intersection(self.df_aggregated_profiles.index)
        return matching_columns, matching_index

    def run_self_consumption(self):
        matching_columns, matching_index = self.get_matching_columns_and_index()
        df_self_cons_pct_monthly, df_cov_pct_no_pv_monthly, df_cov_pct_pv_monthly = self.calc_dir_self_consumption(
            self.pv_df_hourly.loc[matching_index, matching_columns],
            self.df_aggregated_profiles.loc[matching_index, matching_columns],
            self.df_aggregated_profiles_no_pv.loc[matching_index, matching_columns],
            self.pv_pct
        )
        # Save the results
        df_self_cons_pct_monthly.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}/04_self_cons_pct_month_{self.pv_pct}.csv', index=True, index_label='census_id')
        df_self_cons_pct_monthly.to_csv(f'data/04_energy_consumption_profiles/self_cons_estim/04_self_cons_pct_month_{self.pv_pct}.csv', index=True, index_label='census_id')
        df_cov_pct_no_pv_monthly.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}/04_cov_pct_no_pv_month_{self.no_pv_pct}.csv', index=True, index_label='census_id')
        df_cov_pct_pv_monthly.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}/04_cov_pct_pv_month_{self.pv_pct}.csv', index=True, index_label='census_id')
        return df_self_cons_pct_monthly, df_cov_pct_no_pv_monthly, df_cov_pct_pv_monthly

    def get_aggregated_profiles(self, df_aggregated_profiles, matching_columns, col_suffix=None):
        # Function to aggregate energy consumption profiles by month and season
        monthly_agg, seasonal_agg = self.aggregate_by_month_and_season(df_aggregated_profiles)
        monthly_agg = monthly_agg.loc[:, matching_columns]
        monthly_agg = monthly_agg.T
        monthly_agg.columns = monthly_agg.columns.astype(int)
        monthly_agg.columns = [f'{col_suffix}{month}' for month in monthly_agg.columns]
        monthly_agg['Total, kWh'] = monthly_agg.sum(axis=1).round(4)
        return monthly_agg, seasonal_agg

    def save_aggregated_profiles(self):
        matching_columns, _ = self.get_matching_columns_and_index()
        monthly_agg, seasonal_agg = self.get_aggregated_profiles(self.df_aggregated_profiles, matching_columns, col_suffix='cons_m')
        monthly_agg.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}/04_aggreg_cons_prof_with_pv_by_census_id_monthly_{self.pv_pct}.csv', index=True, index_label='census_id')
        monthly_agg.to_csv(f'data/04_energy_consumption_profiles/cons/04_aggreg_cons_prof_with_pv_by_census_id_monthly_{self.pv_pct}.csv', index=True, index_label='census_id')

        monthly_agg_no_pv, seasonal_agg_no_pv = self.get_aggregated_profiles(self.df_aggregated_profiles_no_pv, matching_columns, col_suffix='no_pv_cons_m')
        monthly_agg_no_pv.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}/04_aggreg_cons_prof_no_pv_by_census_id_monthly_{self.no_pv_pct}.csv', index=True, index_label='census_id')
        return monthly_agg, monthly_agg_no_pv

    def save_pv_profiles(self):
        # Get the pv profiles by census_id column
        pv_df = self.pv_df[self.pv_df['census_id'].notna()]
        pv_df['census_id'] = pv_df['census_id'].astype(str)
        pv_census_aggreg_df = pv_df.groupby('census_id').sum([1,2,3,4,5,6,7,8,9,10,11,12, 'Total, kWh']).drop(columns=['plain_roof']).round(4)
        pv_census_aggreg_df.columns = [f'gen_m{month}' if month in [1,2,3,4,5,6,7,8,9,10,11,12] else month for month in pv_census_aggreg_df.columns]
        pv_census_aggreg_df.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}/04_aggregated_pv_gen_by_census_id_monthly.csv', index=True)
        return pv_census_aggreg_df

    def save_net_balance(self, monthly_agg, pv_census_aggreg_df):
        # Get intersection of both index and columns
        matching_index = pv_census_aggreg_df.index.intersection(monthly_agg.index)
        matching_columns = pv_census_aggreg_df.columns.intersection(monthly_agg.columns)
        # Subset both DataFrames to only the matching rows and columns
        generation_matching = pv_census_aggreg_df.loc[matching_index, matching_columns]
        consumption_matching = monthly_agg.loc[matching_index, matching_columns]
        # Perform element-wise division
        df_self_consumption = (generation_matching / consumption_matching).round(4)
        df_self_consumption.to_csv(f'data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}/04_net_balance_gen_to_cons_by_census_id_monthly.csv', index=True)
        return df_self_consumption

# Example usage:
if __name__ == "__main__":
    estimator = SelfConsumptionEstimator(
        work_dir = os.getcwd().split('python')[0],
        district="otxarkoaga",
        pv_pct=0.25
    )
    #estimator.time_alignment()
    estimator.run_self_consumption()
    monthly_agg, monthly_agg_no_pv = estimator.save_aggregated_profiles()
    pv_census_aggreg_df = estimator.save_pv_profiles()
    estimator.save_net_balance(monthly_agg, pv_census_aggreg_df)

# %%
