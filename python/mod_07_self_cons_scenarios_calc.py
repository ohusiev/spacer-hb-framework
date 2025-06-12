#%%
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from util_func import UtilFunctions
from util_func import PlotFunctions
from util_func import PVAnalysis

class SelfConsumptionAnalysis:
    def __init__(self, root):
        # INPUT FILES:
        self.util = UtilFunctions()
        self.root = root or os.getcwd()
        # PV data
        self.pv_file = os.path.join(root, "data", "01_footprint_s_area_wb_rooftop_analysis_pv_month_pv.xlsx")
        self.df_pv_gen = pd.read_excel(self.pv_file, sheet_name='Otxarkoaga')
        self.df_pv_gen=self.df_pv_gen[self.df_pv_gen["building"]=="V"].copy().reset_index(drop=True)
        print( self.df_pv_gen['r_area'].sum(), "m2 of PV area in Otxarkoaga")
        self.df_self_cons_pct = pd.DataFrame()
        self.stat_data = pd.read_excel(os.path.join(self.root, "data\\04_energy_consumption_profiles", "00_data_census_id_ener_consum_profiling.xlsx"), sheet_name='04_dwelling_profiles_census', index_col=0)
        # Facades data
        self.df_facades = gpd.read_file(os.path.join(self.root, "data/05_buildings_with_energy_and_co2_values+HDemProj.geojson"))
        self.standart_month_cons_cols = {"cons_m1": 1, "cons_m2": 2, "cons_m3": 3, "cons_m4": 4, "cons_m5": 5, "cons_m6": 6, "cons_m7": 7, "cons_m8": 8, "cons_m9": 9, "cons_m10": 10, "cons_m11": 11, "cons_m12": 12}
        self.standart_month_pv_gen_cols = {"self_m1": 1, "self_m2": 2, "self_m3": 3, "self_m4": 4, "self_m5": 5, "self_m6": 6, "self_m7": 7, "self_m8": 8, "self_m9": 9, "self_m10": 10, "self_m11": 11, "self_m12": 12}
        self.prepare_facades_data()
        self.files = os.listdir(os.path.join(self.root,"data\\04_energy_consumption_profiles\\self_cons_estim"))
        self.df_self_cons_pct, self.dwelling_accounted_pct = self.util.load_data(os.path.join(self.root,"data\\04_energy_consumption_profiles\\self_cons_estim"), self.files)
        self.df_pv_gen_census = self.df_pv_gen.groupby('census_id').sum()
        self.df_self_cons_calc = pd.DataFrame()
        self.df_self_cons_calc_per_dwelling = pd.DataFrame()
        self.df_results = pd.DataFrame()
        self.df_heatmap = pd.DataFrame()
        self.df_result_self_sufficiency = pd.DataFrame()

    def prepare_facades_data(self):
        # Prepare facades data
        df_pv_temp = self.df_pv_gen.groupby('build_id').sum().reset_index()
        self.df_facades['build_id'] = self.df_facades['build_id'].astype(str)
        df_pv_temp['build_id'] = df_pv_temp['build_id'].astype(str)
        self.df_facades = pd.merge(self.df_facades, df_pv_temp[["build_id", "r_area", "installed_kWp", "n_panel", "Total, kWh"]], on="build_id", how='left')
        #self.df_facades[["r_area", "n_panel", "Total, kWh"]] = self.df_facades[["r_area", "n_panel", "Total, kWh"]].fillna(0)

    def calculate_self_consumption(self):
        # CALCULATION OF SELF CONSUMPTION PERCENTAGE
        for i in self.dwelling_accounted_pct:
            temp_filter = self.df_self_cons_pct[self.df_self_cons_pct['%dwelling_accounted'] == i].copy()
            for key, value in self.standart_month_pv_gen_cols.items():
                temp_filter.loc[:, key] = temp_filter.loc[:, key].mul(self.df_pv_gen_census.loc[:, value], axis=0)
            self.df_self_cons_calc = pd.concat([self.df_self_cons_calc, temp_filter], axis=0)

        # Write to excel and create csv separately
        with pd.ExcelWriter("data/07_pv_self_cons_econom.xlsx", mode='w', engine='openpyxl') as writer:
            self.df_self_cons_pct.to_excel(writer, sheet_name='self_cons_pct')
            self.df_self_cons_calc.to_excel(writer, sheet_name='self_cons_calc')

        self.df_self_cons_pct.to_csv("data/07_pv_self_cons_pct.csv")
        self.df_self_cons_calc.to_csv("data/07_pv_self_cons_calc.csv")

    def calculate_per_dwelling(self):
        # CALCULATE AND PLOT AVERAGE SELF-CONSUMPTION PER Dwelling Percentage
        for i in self.dwelling_accounted_pct:
            df_temp = self.df_self_cons_calc.loc[self.df_self_cons_calc['%dwelling_accounted'] == i, self.df_self_cons_calc.columns[0:12]].div(self.stat_data['Total number of dwellings']*float(i), axis=0)
            df_temp['%dwelling_accounted'] = i
            df_temp = df_temp.dropna()
            self.df_self_cons_calc_per_dwelling = pd.concat([self.df_self_cons_calc_per_dwelling, df_temp], axis=0)

        PlotFunctions.plot_self_consumption_trends(
            self.df_self_cons_calc_per_dwelling,
            self.df_self_cons_pct,
            xlabel='Months',
            ylabel='Total Self-Consumed Electricity (kWh)',
            header='Monthly Average PV Electricity Self-Consumption per dwelling Percentage by Scenario'
        )

    def filter_by_census_id(self, CENSUS_ID):
        # Fileter per census_id
        df_self_cons_calc_per_dwelling_filter = self.df_self_cons_calc_per_dwelling[self.df_self_cons_calc_per_dwelling.index == CENSUS_ID]
        df_self_cons_pct_filer = self.df_self_cons_pct[self.df_self_cons_pct.index == CENSUS_ID]
        return df_self_cons_calc_per_dwelling_filter, df_self_cons_pct_filer

    def plot_self_consumption_trends(self, df_calc_self_cons_per_dwelling, df_self_cons_pct_, xlabel, ylabel, header, y_min=None, y_max=None):
        # Plotting function with adjustable y_min and y_max
        scenarios = df_calc_self_cons_per_dwelling['%dwelling_accounted'].unique()
        monthly_columns = [col for col in df_calc_self_cons_per_dwelling.columns if 'self_m' in col]

        plt.figure(figsize=(12, 6))
        for scenario in scenarios:
            scenario_data = df_calc_self_cons_per_dwelling[df_calc_self_cons_per_dwelling['%dwelling_accounted'] == scenario]
            monthly_totals = scenario_data[monthly_columns].mean()
            plt.plot(monthly_columns, monthly_totals, marker='o', label=f"Scenario {float(scenario)*100:.0f}%")

        plt.title(header)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(y_min, y_max)
        plt.xticks(rotation=45)
        plt.legend(title='% of Dwellings Accounted')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        for scenario in scenarios:
            scenario_data = df_self_cons_pct_[df_self_cons_pct_['%dwelling_accounted'] == scenario]
            monthly_averages = scenario_data[monthly_columns].mean()
            plt.plot(monthly_columns, monthly_averages, marker='o', label=f"Scenario {float(scenario)*100:.0f}%")

        plt.title('Monthly Self-Consumption Percentage by Scenario')
        plt.xlabel('Months')
        plt.ylabel('Average Self-Consumption Percentage')
        plt.xticks(rotation=45)
        plt.legend(title='% of Dwellings Accounted')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def load_consumption_data(self):
        # Upload consumption data from folder
        files = os.listdir(os.path.join(self.root,"data\\04_energy_consumption_profiles\\cons"))
        self.df_consumption, self.dwelling_accounted_pct = self.util.load_data(os.path.join(self.root,"data\\04_energy_consumption_profiles\\cons"), files)

    def filter_per_dwelling_percentage(self):
        # Filter per dwelling percentage
        for PCT_DWELLING in self.dwelling_accounted_pct:
            df_consumption_filter = self.df_consumption.loc[self.df_consumption["%dwelling_accounted"] == str(PCT_DWELLING)]
            df_pv_sef_consm_filter = self.df_self_cons_calc.loc[self.df_self_cons_calc["%dwelling_accounted"] == str(PCT_DWELLING)]

            df_consumption_filter = df_consumption_filter.rename(columns=self.standart_month_cons_cols)
            df_pv_sef_consm_filter = df_pv_sef_consm_filter.rename(columns=self.standart_month_pv_gen_cols)

            df_joined = pd.DataFrame()
            pv_generation = self.df_pv_gen_census.loc[:, 1:12].sum()
            consumption = df_consumption_filter.loc[:, 1:12].sum()
            self_cons = df_pv_sef_consm_filter.loc[:, 1:12].sum()

            df_joined['pv_generation'] = pv_generation
            df_joined['aggreg_consumption'] = consumption
            df_joined['PV_self_consumed'] = self_cons

            df_joined['Consumed_from_grid'] = df_joined['aggreg_consumption'] - df_joined['PV_self_consumed']
            df_joined['Surplus_to_Grid'] = df_joined['pv_generation'] - df_joined['PV_self_consumed']
            df_joined['Surplus_to_Grid'] = df_joined['Surplus_to_Grid'].clip(lower=0)
            df_plot = df_joined.transpose()
            title = f'Distribution of annual generation, consumption and self-consumption for {float(PCT_DWELLING)*100}% of dwellings'
            PlotFunctions.plot_combined_load_data_from_df_adj(df_plot, title, legend_show=True, y_min=None, y_max=1400000)
            df_joined['Filter'] = PCT_DWELLING
            df_joined["Month"] = df_joined.index
            self.df_results = pd.concat([self.df_results, df_joined], axis=0)

        # Save to excel
        self.df_results.to_excel("data/07_distribution of_self_cons_cons__pv_gen.xlsx")

    def create_heatmap(self):
        # Create heatmap data
        self.df_heatmap = self.df_self_cons_pct.copy()
        self.df_heatmap['Total'] = self.df_heatmap[list(self.standart_month_pv_gen_cols.keys())].mean(axis=1) * 100
        self.df_heatmap = self.df_heatmap.pivot_table(index=self.df_heatmap.index, columns='%dwelling_accounted', values='Total')
        self.df_heatmap = self.df_heatmap.rename(columns={'1': '100%', '0.75': '75%', '0.5': '50%', '0.25': '25%'})
        self.heatmap(self.df_heatmap, cbarname='Self-Consumption Percentage (%)')

    def heatmap(self, df_heatmap, title="Scenario Matrix Heatmap: aggregated PV self-consumptionn", cbarname='Benefit Levels (%)', cmap='YlGnBu'):
        # Plot Heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_heatmap, annot=True, cmap=cmap, cbar_kws={'label': cbarname}, fmt='.0f', vmin=0, vmax=100)
        plt.title(title)
        plt.xlabel('Scenarios')
        plt.ylabel('Census ID')
        plt.show()

    def calculate_self_sufficiency(self):
        # Calculate self-sufficiency
        for PC_DWELLING in self.dwelling_accounted_pct:
            df_temp1 = self.df_self_cons_calc.loc[self.df_self_cons_calc['%dwelling_accounted'] == PC_DWELLING]
            df_temp2 = self.df_consumption.loc[self.df_consumption['%dwelling_accounted'] == PC_DWELLING]
            df_temp1 = df_temp1.rename(columns=self.standart_month_pv_gen_cols)
            df_temp2 = df_temp2.rename(columns=self.standart_month_cons_cols)
            df_temp1 = df_temp1.loc[:, self.standart_month_pv_gen_cols.values()].div(df_temp2.loc[:, self.standart_month_cons_cols.values()], axis=0)
            df_temp1['%dwelling_accounted'] = PC_DWELLING
            self.df_result_self_sufficiency = pd.concat([self.df_result_self_sufficiency, df_temp1], axis=0)

        self.df_result_self_sufficiency['Total'] = self.df_result_self_sufficiency[list(self.standart_month_pv_gen_cols.values())].mean(axis=1) * 100
        self.df_result_self_sufficiency = self.df_result_self_sufficiency.pivot_table(index=self.df_result_self_sufficiency.index, columns='%dwelling_accounted', values='Total').round(2)
        self.df_result_self_sufficiency = self.df_result_self_sufficiency.rename(columns={'1': '100%', '0.75': '75%', '0.5': '50%', '0.25': '25%'})
        self.heatmap(self.df_result_self_sufficiency, title="Scenario Matrix Heatmap: self-sufficiency", cbarname='Self-Sufficiency Percentage (%)', cmap='YlGnBu')
#%%
# Example usage:
if __name__ == "__main__":
    root = r"C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework"
    analysis = SelfConsumptionAnalysis(root)
    analysis.calculate_self_consumption()
    analysis.calculate_per_dwelling()
    CENSUS_ID = 4802003011
    df_self_cons_calc_per_dwelling_filter, df_self_cons_pct_filer = analysis.filter_by_census_id(CENSUS_ID)
    analysis.plot_self_consumption_trends(
        df_self_cons_calc_per_dwelling_filter,
        df_self_cons_pct_filer,
        xlabel='Months',
        ylabel='Total Self-Consumed Electricity (kWh)',
        header=f'Monthly Average PV Electricity Self-Consumption in {CENSUS_ID} section per dwelling Percentage by Scenario',
        y_min=0, y_max=200
    )
    analysis.load_consumption_data()
    analysis.filter_per_dwelling_percentage()
    analysis.create_heatmap()
    analysis.calculate_self_sufficiency()

# %%
