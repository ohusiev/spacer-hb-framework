#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .util_func import PVAnalysis
from .util_func import UtilFunctions
from matplotlib.font_manager import FontProperties

class EconomicAnalysis:
    def __init__(self, root):
        self.root = root
        self.util_pv = PVAnalysis()
        self.util_func = UtilFunctions()
        self.combined_df = pd.DataFrame()
        self.df = None
        self.agg_df = None
        self.dwelling_col = "n_dwellOriginal"
        self.nrpe_factor = 2.007 # electricity factor for NRPE 
        self.co2_elect_factor = 0.357 # electricity factor for CO2 emissions

    def load_data(self):
        self.df = pd.read_excel(self.root + r"\data\05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic_otxarkpaga_orginal_price_incr+pv_degrad.xlsx")

    def filter_residential_with_pv(self):
        # select only residential buildings and buildings with PV
        self.df = self.df[(self.df["building"] == "V") & (self.df["Total, kWh"] > 0)]

    def calculate_economic_indicators(self, r=0.05, n=20, n_int=30, energy_price_growth_rate=0, heating_energy_price_euro_per_kWh=0.243, pv_degradation_rate=0, CAL_REFERENCE_CASE=False, SCENARIO="S2", COMBINATION_ID=2):
        # Define parameters
        intervention_combinations = {
            1: ["facade", "roof"], # low_intervention_min or high_intervention_min
            2: ["facade", "windows", "roof"],#low_intervention_max or high_intervention_max
        }
        self.dwelling_col = "n_dwellOriginal" # 'n_dwellEstim' or 'n_dwellOriginal'
        df = self.df

        # Calculate EUAC for Envelope and PV
        if not CAL_REFERENCE_CASE:
            # Calculate initial investment and annual maintenance for Envelope and PV
            if "windows" in intervention_combinations[COMBINATION_ID]:
                df['Envelope_Initial_I'] = df[[f'{SCENARIO}_Facd_I_C',f'{SCENARIO}_Wind_I_C',  f'{SCENARIO}_Roof_I_C']].sum(axis=1)
                df['Envelope_Annual_M'] = df[[f'{SCENARIO}_Facd_M_C', f'{SCENARIO}_Wind_M_C', f'{SCENARIO}_Roof_M_C']].sum(axis=1)
            else:
                df['Envelope_Initial_I'] = df[[f'{SCENARIO}_Facd_I_C',  f'{SCENARIO}_Roof_I_C' ]].sum(axis=1)
                df['Envelope_Annual_M'] = df[[f'{SCENARIO}_Facd_M_C', f'{SCENARIO}_Roof_M_C']].sum(axis=1)

            # Calculate annual savings for Envelope and PV separately
            df['Envelope_Annual_Savings'] = df[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_Savings_Euro']
            df['heating_energy_price_euro_per_kWh'] = heating_energy_price_euro_per_kWh 
            df['heating_energy_price_euro_per_kWh'] = self.util_pv.average_price_increase(df['heating_energy_price_euro_per_kWh'],energy_price_growth_rate, lifespan=30)
            df['Envelope_Energy_Euro'] = df[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_kWh/m2']* df['grossFloorArea']*df['heating_energy_price_euro_per_kWh']

            df['PV_Annual_Savings'] = df['PV_self_cons_Euro'] + df['PV_to_grid_Euro']
            
            #Calculate Annual Cost (For EUAC)
            df["Envelope_Annual_Cost"]= df['Envelope_Annual_M'] + df['Envelope_Energy_Euro']
            df["PV_Annual_Cost"] = df['PV_M_C'] - df['PV_Annual_Savings']

            # CALCULATE Present Value for Envelope and PV separately
            df['Envelope_PV'] = self.util_pv.calculate_present_value(df, 'Envelope_Initial_I',  'Envelope_Annual_Cost', r, n_int)
            df['PV_PV'] = self.util_pv.calculate_present_value(df, 'PV_I_C',  'PV_Annual_Cost', r, n) # PV_I_C is the initial investment for PV system, PV_PV is the present value of the PV system

            # CALCUALTE EUAC FOR SCENARIO
            df['Envelope_EUAC'] = self.util_pv.calculate_euac(df, r, n_int, 'Envelope_PV')
            df['PV_EUAC'] = self.util_pv.calculate_euac(df, r, n, 'PV_PV')
            df['EUAC'] = df['Envelope_EUAC'] + df['PV_EUAC']

            df['Envelope_Energy_kWh'] = df[f'Comb{COMBINATION_ID}{SCENARIO}_HDem_kWh/m2']* df['grossFloorArea']
        else:
            df['Envelope_Energy_kWh'] = df['HDemProj']* df['grossFloorArea']
            df['heating_energy_price_euro_per_kWh'] = heating_energy_price_euro_per_kWh 
            df['heating_energy_price_euro_per_kWh'] = self.util_pv.average_price_increase(df['heating_energy_price_euro_per_kWh'],energy_price_growth_rate, lifespan=30)
            
            df['Envelope_Energy_Euro'] = df['Envelope_Energy_kWh']*df['heating_energy_price_euro_per_kWh']
            df['PV_Annual_Savings'] = df['PV_self_cons_Euro'] + df['PV_to_grid_Euro']

            # Calculate Annual Cost (For EUAC)
            df["Envelope_Annual_Cost"]= df['Envelope_Energy_Euro']
            df["PV_Annual_Cost"] = df['PV_M_C'] - df['PV_Annual_Savings']

            # CALCULATE Present Value for Envelope and PV separately
            df['Envelope_Initial_I'] = 0
            df['Envelope_PV'] = self.util_pv.calculate_present_value(df, 'Envelope_Initial_I',  'Envelope_Annual_Cost', r, n_int)
            df['PV_PV'] = self.util_pv.calculate_present_value(df, 'PV_I_C',  'PV_Annual_Cost', r, n)

            # CALCUALTE EUAC FOR SCENARIO
            df['Envelope_EUAC'] = self.util_pv.calculate_euac(df, r, n_int, 'Envelope_PV')
            df['PV_EUAC'] = self.util_pv.calculate_euac(df, r, n, 'PV_PV')
            df['EUAC'] = df['Envelope_EUAC'] + df['PV_EUAC']

        # Aggregate by census_id if needed
        if not CAL_REFERENCE_CASE:
            self.agg_df = df.groupby('census_id').agg({
                'Envelope_EUAC': 'sum',
                'PV_EUAC': 'sum',
                'Envelope_Energy_kWh': 'sum',
                "Envelope_Annual_Cost" : 'sum',
                "PV_Annual_Cost": 'sum',
                'Total, kWh': 'sum',
                'EUAC': 'sum',
                'n_dwellEstim': 'sum',
                "n_dwellOriginal": 'sum',
                "Envelope_Initial_I": 'sum',
                "PV_I_C": 'sum',
                "Envelope_Annual_M": 'sum',
                "PV_M_C": 'sum',
                "Envelope_Annual_Savings": 'sum',
                "PV_Annual_Savings": 'sum',
                'grossFloorArea': 'sum'
            }).reset_index()
        else:
            self.agg_df = df.groupby('census_id').agg({
                'Envelope_EUAC': 'sum',
                'PV_EUAC': 'sum',
                'Envelope_Energy_kWh': 'sum',
                "Envelope_Annual_Cost" : 'sum',
                "PV_Annual_Cost": 'sum',
                'Total, kWh': 'sum',
                'EUAC': 'sum',
                'n_dwellEstim': 'sum',
                "n_dwellOriginal": 'sum',
                'grossFloorArea': 'sum'
            }).reset_index()

        # EUAC per dwelling
        agg_df = self.agg_df
        agg_df["EUAC_per_dwelling"] = agg_df["EUAC"]/agg_df[self.dwelling_col]
        agg_df['Envelope_EUAC_per_dwelling'] = agg_df['Envelope_EUAC']/agg_df[self.dwelling_col]
        agg_df["Envelope_Energy_kWh_per_dwelling"] = agg_df["Envelope_Energy_kWh"]/agg_df[self.dwelling_col]
        agg_df['PV_kWh_per_dwelling'] = agg_df['Total, kWh']/agg_df[self.dwelling_col]
        agg_df['PV_EUAC_per_dwelling'] = agg_df['PV_EUAC']/agg_df[self.dwelling_col]
        
        # NRPE and CO2 emissions
        agg_df['NRPE_Envelope_kWh_per_dwelling'] = agg_df['Envelope_Energy_kWh_per_dwelling'] * self.nrpe_factor
        agg_df['NRPE_kWh_per_dwelling'] = (agg_df['Envelope_Energy_kWh_per_dwelling'] - agg_df['PV_kWh_per_dwelling']) * self.nrpe_factor

        agg_df['CO2_Envelope_per_dwelling'] = agg_df['Envelope_Energy_kWh_per_dwelling'] * self.co2_elect_factor
        agg_df['CO2_per_dwelling'] = (agg_df['Envelope_Energy_kWh_per_dwelling'] - agg_df['PV_kWh_per_dwelling']) * self.co2_elect_factor

        # EUAC per m2
        agg_df['EUAC_per_m2'] = agg_df['EUAC']/agg_df["grossFloorArea"]
        agg_df['Envelope_EUAC_per_m2'] = agg_df['Envelope_EUAC']/agg_df["grossFloorArea"]
        if not CAL_REFERENCE_CASE:
            agg_df['PV_EUAC_per_m2'] = agg_df['PV_EUAC']/agg_df["grossFloorArea"]

        # NPRE and CO2 emissions per m2
        agg_df['NRPE_Envelope_kWh_per_m2'] = agg_df['Envelope_Energy_kWh']/agg_df["grossFloorArea"] * self.nrpe_factor
        agg_df['NRPE_kWh_per_m2'] = (agg_df['Envelope_Energy_kWh'] - agg_df['Total, kWh'])/agg_df["grossFloorArea"] * self.nrpe_factor

        agg_df['CO2_Envelope_per_m2'] = agg_df['Envelope_Energy_kWh']/agg_df["grossFloorArea"] * self.co2_elect_factor
        agg_df['CO2_per_m2'] = (agg_df['Envelope_Energy_kWh'] - agg_df['Total, kWh'])/agg_df["grossFloorArea"] * self.co2_elect_factor

        self.agg_df = agg_df

    def append_long_dataframe(self, df, COMBINATION_ID, SCENARIO):
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        # Add the extra columns
        df_copy['Filter'] = f'Comb{COMBINATION_ID}{SCENARIO}'
        return df_copy

    def save_to_excel(self, filename_base):
        self.util_func.add_sheet_to_excel(self.combined_df, self.root + fr"\data\{filename_base}_ADD_SHEET.xlsx", "Sheet0")
        self.combined_df.to_excel(self.root + fr"\data\{filename_base}.xlsx", index=False)

    @staticmethod
    def plotNPV(df, column_mapping, filename, title="EUAC Composition by Scenario with rooftop PV"):
        """
        Plots the NPV composition using data from a specific DataFrame.
        Parameters:
        - df: pandas DataFrame containing the data to plot.
        - column_mapping: dict with keys ["names", "NPV", "CAPEX", "fix_costs", "var_costs", "revenue"] 
          mapping to column names in the DataFrame.
        - filename: str, name of the file to save the plot.
        """
        labels = df[column_mapping["names"]].tolist()
        x = np.arange(len(labels))
        width = 0.35

        # Extract values and scale
        NPV = df[column_mapping["EUAC"]] / 1e6
        inv = -df[column_mapping["CAPEX"]] / 1e6
        fix = -df[column_mapping["fix_costs"]] / 1e6
        rev = df[column_mapping["revenue"]] / 1e6

        sns.set_theme(style='darkgrid')
        fig, ax = plt.subplots()
        ax.grid()

        # Bar plots
        ax.bar(x - width/2, inv, width, label="CAPEX", color="#bc5090")
        ax.bar(x - width/2, fix, width, bottom=inv, label="Fix Costs", color="#58508d")
        ax.bar(x - width/2, rev, width, bottom=0, label="Revenue", color="#20a9a6")
        ax.bar(x + width/2, NPV, width, label="EUAC", color="#ffa600")

        # Labels and formatting
        ax.set_ylabel("EUAC Composition [Mio €]")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(labels)
        ax.grid(linestyle='-', linewidth='0.5', color='white')
        # Annotate NPV bars
        for i in range(len(x)):
            ax.text(x[i] + width/2, NPV.iloc[i] - 0.1, f"{NPV.iloc[i]:.2f}", 
                    color='black', ha="center", size="x-small")

        # Legend
        fontP = FontProperties()
        fontP.set_size('xx-small')
        ax.legend(prop=fontP)

        # Adjust layout and save
        fig.subplots_adjust(bottom=0.30)
        fig.savefig(filename + ".png", dpi=300)
        plt.show()
        plt.close(fig)

    def run_analysis(self, r=0.05, n=20, n_int=30, energy_price_growth_rate=0, heating_energy_price_euro_per_kWh=0.243, pv_degradation_rate=0, CAL_REFERENCE_CASE=False, SCENARIO="S2", COMBINATION_ID=2):
        self.load_data()
        self.filter_residential_with_pv()
        self.calculate_economic_indicators(r, n, n_int, energy_price_growth_rate, heating_energy_price_euro_per_kWh, pv_degradation_rate, CAL_REFERENCE_CASE, SCENARIO, COMBINATION_ID)
        if CAL_REFERENCE_CASE:
            COMBINATION_ID = 0
            SCENARIO = "ref_case"
        temp_df = self.append_long_dataframe(self.agg_df, COMBINATION_ID, SCENARIO)
        self.combined_df = pd.concat([self.combined_df, temp_df], ignore_index=True)
        if CAL_REFERENCE_CASE:
            print(self.combined_df)

    def plot_neighbourhood(self, filename="output_filename", euac_col="EUAC"):
        combined_df_neighbourhood = self.combined_df.groupby('Filter').agg({
            'EUAC': 'sum',
            "Envelope_EUAC": 'sum',
            self.dwelling_col: 'sum',
            "Envelope_Initial_I": 'sum',
            "PV_I_C": 'sum',
            "Envelope_Annual_M": 'sum',
            "PV_M_C": 'sum',
            "Envelope_Annual_Savings": 'sum',
            "PV_Annual_Savings": 'sum'
        }).reset_index()

        if euac_col == "EUAC":
            combined_df_neighbourhood['CAPEX'] = combined_df_neighbourhood['Envelope_Initial_I'] + combined_df_neighbourhood['PV_I_C']
            combined_df_neighbourhood['fix_share'] = combined_df_neighbourhood['Envelope_Annual_M'] + combined_df_neighbourhood['PV_M_C']
            combined_df_neighbourhood['revenue'] = combined_df_neighbourhood['Envelope_Annual_Savings'] + combined_df_neighbourhood['PV_Annual_Savings']
            title = "EUAC Composition by Scenario with rooftop PV"
        if euac_col == "Envelope_EUAC":
            combined_df_neighbourhood['CAPEX'] = combined_df_neighbourhood['Envelope_Initial_I']
            combined_df_neighbourhood['fix_share'] = combined_df_neighbourhood['Envelope_Annual_M']
            combined_df_neighbourhood['revenue'] = combined_df_neighbourhood['Envelope_Annual_Savings']
            title = "Envelope EUAC Composition by Scenario without rooftop PV"

        column_mapping = {
            "names": "Filter",
            "EUAC": euac_col, # "Envelope_EUAC", "EUAC"
            "CAPEX": "CAPEX",
            "fix_costs": "fix_share",
            "var_costs": "fix_share", # Not used in plot
            "revenue": "revenue"
        }
        self.plotNPV(combined_df_neighbourhood, column_mapping, filename, title)

# Example usage
if __name__ == "__main__":
    root = r"H:\My Drive\01_PHD_CODE\chapter_5\spain\bilbao_otxarkoaga"
    analysis = EconomicAnalysis(root)
    analysis.run_analysis()
    analysis.save_to_excel("05_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic+EUAC_ORIGIN")
    analysis.plot_neighbourhood("output_filename")

# %%
