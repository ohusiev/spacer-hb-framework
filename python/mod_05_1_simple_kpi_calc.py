#%%
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from .util_func import UtilFunctions
from .util_func import PlotFunctions
from .util_func import PVAnalysis

class EconomicKPIAnalyzer:
    def __init__(self, root, pv_file_name, cost_file_name):
        # INPUT FILES:
        self.root = root
        self.pv_file_name = pv_file_name
        self.pv_path = os.path.join("data/", pv_file_name)
        self.cost_file_name = cost_file_name

        # Load Data
        self.df_pv_gen = pd.read_excel(self.pv_path, sheet_name='Otxarkoaga', dtype={'build_id': str, 'census_id': str})
        self.df_facades = gpd.read_file(os.path.join(root, "data/05_buildings_with_energy_and_co2_values+HDemProj.geojson"))
        self.df_costs = pd.read_excel(cost_file_name, sheet_name='cost_facade')
        self.df_h_dem_reduction_coeffs = pd.read_excel(cost_file_name, sheet_name='HDemReductionCoeffs', index_col='id')
        self.df_prices = pd.read_excel(cost_file_name, sheet_name='cost_electricity')
        self.df_monthly_self_cons_percentages = pd.read_excel(cost_file_name, sheet_name='sel-cons_%_fixed', index_col=0)
        self.monthly_self_cons_percentages = self.df_monthly_self_cons_percentages.set_index('Month')['self-cons'].to_dict()
        self.df_ref_census_data = pd.read_csv('economic_calc/reference_census_data_table.csv', dtype={'census_id': str})

        # Utilities
        self.util = UtilFunctions()
        self.util_pv = PVAnalysis()
        self.util_plot = PlotFunctions()

    def prepare_facades(self):
        # Merge PV data into facades
        print("Preparing facades data by merging with PV generation data...")
        self.df_facades['build_id'] = self.df_facades['build_id'].astype(str)
        self.df_facades = pd.merge(
            self.df_facades,
            self.df_pv_gen[["build_id",  "installed_kWp", "n_panel", "Total, kWh"]],
            on="build_id", how='left'
        )
        self.df_facades[["n_panel", "Total, kWh"]] = self.df_facades[["n_panel", "Total, kWh"]].fillna(0)
        print("Facades data prepared with PV generation information.")
        return self.df_pv_gen

    def plot_pv_generation(self, filter_col_name='building', filter_value='V'):
        # Plot PV generation by month total
        print("Filtering PV data, by column:", filter_col_name, "with value:", filter_value)
        self.df_facades = self.df_facades[self.df_facades[filter_col_name] == filter_value]
        print("Filter PV generation by type of building...")
        self.df_pv_gen=self.df_pv_gen[self.df_pv_gen[filter_col_name]==filter_value].copy().reset_index(drop=True)
        self.df_pv_gen.loc[len(self.df_pv_gen)]= self.df_pv_gen.sum(axis=0,numeric_only=True)
        self.df_pv_gen.loc[len(self.df_pv_gen)-1,'build_id']='Total'

        columns_to_plot = [i for i in range(1, 13)]
        self.util_plot.bar_plot_month_pv_columns(
            self.df_pv_gen[self.df_pv_gen['build_id'] == 'Total'],
            columns_to_plot,
            title="Monthly Data Distribution",
            xlabel="Months",
            ylabel="PV Electricity (kWh)",
            font_size=18
        )

    def plot_census_stackplot(self):
        # Reindex by census_id and plot stackplot
        columns_to_plot = [i for i in range(1, 13)]
        df_pv_gen_census = self.df_pv_gen.groupby('census_id').sum()
        df_pv_gen_census = df_pv_gen_census[columns_to_plot]
        df_pv_gen_census = df_pv_gen_census.round(0)
        self.util_plot.plot_census_stackplot(
            df_pv_gen_census,
            title="Estimated Monthly PV Generation by Census",
            xlabel="Months",
            ylabel="PV Electricity (kWh)",
            font_size=18
        )

    @staticmethod
    def adjust_heating_demand_vectorized(df, menthod1_col_name, menthod2_col_name, relative_diff_threshold=5, base_weight=0.5, scaling_factor=0.01):
        """
        Adjusts heating demand estimates from two methods in a vectorized manner.
        Applies dynamic weighting if one estimate is 'relative_diff_threshold'-times more than value of the other;
        otherwise, returns the menthod2_col_name by default.
        """
        method1 = df[menthod1_col_name]
        method2 = df[menthod2_col_name]
        ratio = method1 / method2
        condition = (ratio > relative_diff_threshold) | (ratio < 1 / relative_diff_threshold)
        difference = np.abs(method1 - method2)
        adjustment_coefficient = 1 + (scaling_factor * difference)
        weight1 = np.where(method1 > method2, base_weight * adjustment_coefficient, base_weight / adjustment_coefficient)
        weight2 = np.where(method1 > method2, (1 - base_weight) / adjustment_coefficient, (1 - base_weight) * adjustment_coefficient)
        total_weight = weight1 + weight2
        weight1 /= total_weight
        weight2 /= total_weight
        adjusted_value = (weight1 * method1) + (weight2 * method2)
        adjusted_value = np.where(condition, adjusted_value, method2)
        print(f'By default, the adjusted value is the value of `{menthod2_col_name}` when the condition is not met')
        return pd.Series(adjusted_value, index=df.index).round(2)

    @staticmethod
    def calculate_heating_demand_reduction(df_h_dem_reduction_coeffs, region, scenario, combination):
        h_dem_reduction_coeff = df_h_dem_reduction_coeffs.loc[
            (df_h_dem_reduction_coeffs["Region"] == region) &
            (df_h_dem_reduction_coeffs["Type"] == 'HDemReduction') &
            (df_h_dem_reduction_coeffs["Component"] == 'envelope') &
            (df_h_dem_reduction_coeffs["Name"].isin(combination)),
            [scenario]
        ].sum().values[0].round(3)
        if pd.isna(h_dem_reduction_coeff):
            h_dem_reduction_coeff = 0
            print(f"No heating demand reduction coefficient found for Region: {region}, Scenario: {scenario}, Combination: {combination}")
        return h_dem_reduction_coeff

    def calculate_costs(self, SCENARIO):
        # Select the scenario Name for the costs
        df_costs = self.df_costs
        df_facades = self.df_facades

        c_facade_m2 = df_costs.loc[(df_costs["Type"] == 'costs') & (df_costs["Component"] == 'facade') & (df_costs["Name"] != 'Maint'), [SCENARIO]].sum().values[0]
        maint_facade_m2 = df_costs.loc[(df_costs["Type"] == 'costs') & (df_costs["Component"] == 'facade') & (df_costs["Name"] == 'Maint'), [SCENARIO]].sum().values[0]
        c_windows_m2 = df_costs.loc[(df_costs["Type"] == 'costs') & (df_costs["Component"] == 'windows'), [SCENARIO]].sum().values[0]
        maint_windows_m2 = df_costs.loc[(df_costs["Type"] == 'costs') & (df_costs["Component"] == 'windows') & (df_costs["Name"] == 'Maint'), [SCENARIO]].sum().values[0]
        c_roof_m2 = df_costs.loc[(df_costs["Type"] == 'costs') & (df_costs["Component"] == 'roof'), [SCENARIO]].sum().values[0]
        maint_roof_m2 = df_costs.loc[(df_costs["Type"] == 'costs') & (df_costs["Component"] == 'roof') & (df_costs["Name"] == 'Maint'), [SCENARIO]].sum().values[0]

        df_facades[f"{SCENARIO}_Facd_I_C"] = df_facades["Total_fa_area"] * c_facade_m2
        df_facades[f"{SCENARIO}_Facd_M_C"] = df_facades["Total_fa_area"] * maint_facade_m2
        df_facades[f"{SCENARIO}_Wind_I_C"] = df_facades["Tot_window_area"] * c_windows_m2
        df_facades[f"{SCENARIO}_Wind_M_C"] = df_facades["Tot_window_area"] * maint_windows_m2
        df_facades[f"{SCENARIO}_Roof_I_C"] = df_facades["r_area"] * c_roof_m2
        df_facades[f"{SCENARIO}_Roof_M_C"] = df_facades["r_area"] * maint_roof_m2

        self.df_facades = df_facades

    def calculate_heating_demand(self, REGION, SCENARIO, COMBINATION_ID, ESTIM_AVG_DWE_SIZE, intervention_combinations, heating_energy_price_euro_per_kWh):
        df_facades = self.df_facades
        df_h_dem_reduction_coeffs = self.df_h_dem_reduction_coeffs
        combination = intervention_combinations[COMBINATION_ID]

        # HEATING DEMAND REDUCTION CALCULATIONS
        h_dem_reduction_coeff = self.calculate_heating_demand_reduction(df_h_dem_reduction_coeffs, REGION, SCENARIO, combination)

        # HEATING DEMAND SAVINGS CALCULATIONS
        df_facades['HDemProj'] = df_facades['HDemProj'].fillna(0)
        df_facades['HDem_iNSPiRE'] = df_facades['HDem_iNSPiRE'].fillna(0)
        print(f"Heating demand reduction coefficient for {REGION}, {SCENARIO}, {combination}: {h_dem_reduction_coeff}")
        df_facades['HDemProj'] = self.adjust_heating_demand_vectorized(df_facades, 'HDem_iNSPiRE', 'HDemProj')

        df_facades[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_kWh/m2'] = df_facades['HDemProj'] * (1 - h_dem_reduction_coeff)

        # Estimation based on the original data for Otxarkoaga
        data = {
            "census_id": [4802003003, 4802003005, 4802003006, 4802003007, 4802003009, 4802003010, 4802003011, 4802003015],
            "Num dwellings": [501, 830, 404, 464, 536, 350, 664, 464],
            "Total Population": [1037, 1624, 846, 914, 1220, 737, 1395, 1088],
            "Avg dwelling size": [79, 59, 59, 59, 73, 59, 57, 80],
            "Buildings": [29, 58, 25, 28, 36, 22, 50, 19]
        }
        #df_reference = pd.DataFrame(data)
        #df_reference['census_id'] = df_reference['census_id'].astype(str)

        if 'Avg dwelling size' not in df_facades.columns:
            df_facades = df_facades.merge(self.df_ref_census_data[['census_id', 'Num dwellings', 'Avg dwelling size']], on='census_id', how='left')
        if 'grossFloorArea' not in df_facades.columns:
            df_facades['grossFloorArea'] = df_facades['f_area'] * df_facades['n_floorsEstim']
        if 'n_dwellOriginal' not in df_facades.columns:
            df_facades['n_dwellOriginal'] = (df_facades['grossFloorArea'] / df_facades['Avg dwelling size']).round(0)
        if 'n_dwellEstim' not in df_facades.columns:
            AVG_DWE_SIZE = ESTIM_AVG_DWE_SIZE
            df_facades['n_dwellEstim'] = (df_facades['grossFloorArea'] / AVG_DWE_SIZE).round(0)

        df_facades[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_Savings_kWh'] = (df_facades['HDemProj'] - df_facades[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_kWh/m2']) * df_facades['grossFloorArea']
        df_facades[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_Savings_Euro'] = df_facades[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_Savings_kWh'] * heating_energy_price_euro_per_kWh

        # Round All values to 3 decimal points
        df_facades.loc[:, df_facades.columns[-8:]] = df_facades.loc[:, df_facades.columns[-8:]].round(3)
        self.df_facades = df_facades

    def save_facades(self, filename):
        self.df_facades.to_file(os.path.join(self.root, filename), driver='GeoJSON')

    def calculate_pv_costs(self, PV_M_C=0.025):
        df_pv_gen = self.df_pv_gen
        df_pv_gen['PV_I_C'] = df_pv_gen['installed_kWp'].apply(lambda x: self.util_pv.pv_size_to_cost_equation(x))
        df_pv_gen['PV_M_C'] = df_pv_gen['PV_I_C'] * PV_M_C
        df_pv_gen.loc[:, ['PV_I_C', 'PV_M_C']] = df_pv_gen.loc[:, ['PV_I_C', 'PV_M_C']].round(3)
        self.df_pv_gen = df_pv_gen

    def save_pv_to_excel(self):
        save_path = os.path.join(self.root,"data", self.pv_file_name)
        self.util.add_sheet_to_excel(self.df_pv_gen, save_path, sheet_name='pv_cost_upd', index=False)

    def join_pv_costs_to_facades(self):
        self.df_facades = pd.merge(
            self.df_facades,
            self.df_pv_gen[["build_id", "PV_I_C", "PV_M_C"]],
            on="build_id", how='left'
        )

    def save_facades_with_pv(self, filename):
        self.df_facades.to_file(os.path.join(self.root, filename), driver='GeoJSON')

    def calculate_pv_metrics(self, pv_degradation_rate, energy_price_growth_rate):
        df_pv_results = self.util_pv.calculate_pv_metrics_adv_df(
            self.df_pv_gen,
            self.df_prices,
            self.monthly_self_cons_percentages,
            degradation_rate=pv_degradation_rate,
            price_increase_rate=energy_price_growth_rate
        )
        df_pv_results.drop(columns=['census_id'], inplace=True)
        #cols_to_merge = [col for col in df_pv_results.columns if col not in self.df_facades.columns or col == 'build_id']
        #self.df_economic_analysis = self.df_facades.merge(df_pv_results[cols_to_merge], on=['build_id'], how='left')
        self.df_economic_analysis = self.df_facades.merge(df_pv_results, on=['build_id'], how='left')
        print(self.df_economic_analysis[['build_id', 'census_id', 'PV_self_cons_kWh', 'PV_to_grid_kWh', 'PV_self_cons_Euro', 'PV_to_grid_Euro', ]])

    def save_economic_analysis(self, geojson_filename, excel_filename):
        self.df_economic_analysis.to_file(os.path.join(self.root, geojson_filename), driver='GeoJSON')
        df_no_geom = self.df_economic_analysis.drop(columns='geometry')
        df_no_geom.to_excel(os.path.join(self.root, excel_filename), index=False)

# Example usage:
if __name__ == "__main__":
    root = r"C:\Users\Oleksandr-MSI\Documentos\GitHub\spacer-hb-framework"
    pv_file_name = "01_footprint_s_area_wb_rooftop_analysis_pv_month_pv.xlsx"
    cost_file_name = "241211_econom_data.xlsx"

    analyzer = EconomicKPIAnalyzer(root, pv_file_name, cost_file_name)
    analyzer.prepare_facades()
    analyzer.plot_pv_generation()
    analyzer.plot_census_stackplot()

    SCENARIO = "S2"
    REGION = 'Pais_Vasco'
    COMBINATION_ID = 2
    intervention_combinations = {
        1: ["facade", "roof"],
        2: ["facade", "windows", "roof"],
    }
    heating_energy_price_euro_per_kWh = 0.243
    energy_price_growth_rate = 0
    pv_degradation_rate = 0

    analyzer.calculate_costs(SCENARIO)
    analyzer.calculate_heating_demand(REGION, SCENARIO, COMBINATION_ID, intervention_combinations, heating_energy_price_euro_per_kWh)
    analyzer.save_facades("05_buildings_with_energy_and_co2_values+HDemProj_facade_costs_with_HDem_corr.geojson")
    analyzer.calculate_pv_costs()
    analyzer.save_pv_to_excel()
    analyzer.join_pv_costs_to_facades()
    analyzer.save_facades_with_pv("05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV.geojson")
    analyzer.calculate_pv_metrics(pv_degradation_rate, energy_price_growth_rate)
    analyzer.save_economic_analysis(
        "05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic.geojson",
        "05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic_otxarkpaga_orginal_price_incr+pv_degrad.xlsx"
    )

# %%
