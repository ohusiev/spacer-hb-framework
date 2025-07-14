#%%
import pandas as pd
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import util_func
import importlib
import matplotlib.pyplot as plt

class EnercomEstimator:
    def __init__(self, root,df_pv_gen=None, pv_pct=None, ec_price_coef=None ):
        # Initialize parameters
        self.root = root
        print(f"Root directory: {self.root}")
        self.pv_pct = pv_pct
        self.no_pv_pct = 1 - pv_pct
        self.ec_price_coef = ec_price_coef

        # DataFrames for sensitivity analysis
        self.df_sensitivity_total = pd.DataFrame()
        self.df_sensitivity_vulner_zones_monthly = pd.DataFrame()

        # Load utility functions
        #module_name = "util_func"
        #self.util_func = importlib.import_module(module_name)

        # Load data
        #self.load_data()

    #def load_data(self):
        # Load PV generation data
        #pv_file_name = "02_footprint_r_area_wb_rooftop_analysis_pv_month_pv.xlsx"
        #pv_path = os.path.join(self.root,"data", pv_file_name)
        #self.df_pv_gen = pd.read_excel(pv_path, sheet_name='Otxarkoaga', dtype={'census_id': str})
        self.df_pv_gen=df_pv_gen#[self.df_pv_gen["building"]=="V"]
        self.df_pv_gen = self.df_pv_gen.groupby('census_id').sum().reset_index()
        print (f"PV generation data loaded from and grouped by census_id.")

        # Load other dataframes
        self.df_cons_with_pv = pd.read_csv(
            os.path.join(self.root, f'data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}/04_aggreg_cons_prof_with_pv_by_census_id_monthly_{self.pv_pct}.csv'),
            dtype={'census_id': str})
        self.df_cons_no_pv = pd.read_csv(
            os.path.join(self.root, f'data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}/04_aggreg_cons_prof_no_pv_by_census_id_monthly_{self.no_pv_pct}.csv'),
            dtype={'census_id': str})
        self.df_self_cons_pct_pv = pd.read_csv(
            os.path.join(self.root, f'data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}/04_self_cons_pct_month_{self.pv_pct}.csv'),
            dtype={'census_id': str})
        self.df_cov_pct_no_pv = pd.read_csv(
            os.path.join(self.root, f'data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}/04_cov_pct_no_pv_month_{self.no_pv_pct}.csv'),
            dtype={'census_id': str})
        self.df_cov_pct_with_pv = pd.read_csv(
            os.path.join(self.root, f'data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}/04_cov_pct_pv_month_{self.pv_pct}.csv'),
            dtype={'census_id': str})

        self.df_prices = pd.read_excel(os.path.join(self.root,"00_input_data.xlsx"), sheet_name='cost_electricity', index_col=0)

        # Download statistical data
        self.stat_data = pd.read_excel(
            os.path.join(self.root, "00_mod_04_input_data_census_id_ener_consum_profiling.xlsx"),
            sheet_name='04_dwelling_profiles_census', dtype={'census_id': str}, index_col=0)
        self.stat_data.index = self.stat_data.index.astype(str)


        
    # non optimized version.  A critical difference: if any census ID is missing in one of the DataFrames, the second version will silently drop that row.
    """
    def prepare_energy_data(self):
        # Step 1: Aggregate PV generation per census_id by month
        self.df_pv_gen.rename(columns={i: f'gen_m{i}' for i in range(1, 13)}, inplace=True)
        monthly_gen_cols = [f'gen_m{i}' for i in range(1, 13)]
        pv_gen = self.df_pv_gen.groupby('census_id')[monthly_gen_cols].sum().reset_index()

        # Merge PV generation, consumption, and self-consumption data on census_id. Uses explicit sequential merges — more readable in small pipelines but less scalable.


        energy_data = pd.merge(pv_gen, self.df_cons_with_pv, on='census_id') #Uses default merge (how='inner'), which may drop rows if any DataFrame is missing a matching census_id.
        energy_data = pd.merge(energy_data, self.df_self_cons_pct_pv, on='census_id')
        energy_data = pd.merge(energy_data, self.df_cov_pct_no_pv, on='census_id')
        energy_data = pd.merge(energy_data, self.df_cov_pct_with_pv, on='census_id')
        energy_data = pd.merge(energy_data, self.df_cons_no_pv, on='census_id')
        self.energy_data = energy_data
    """
    def prepare_energy_data(self):
        # Rename PV generation columns only if not already renamed
        gen_cols = {i: f'gen_m{i}' for i in range(1, 13)}
        if not all(f'gen_m{i}' in self.df_pv_gen.columns for i in range(1, 13)):
            self.df_pv_gen.rename(columns=gen_cols, inplace=True)

        monthly_gen_cols = [f'gen_m{i}' for i in range(1, 13)]
        pv_gen = self.df_pv_gen.groupby('census_id', as_index=False)[monthly_gen_cols].sum()

        # List of (dataframe, on) to merge in order. Uses a loop to merge from a list — more flexible and easier to maintain.
        merge_list = [
            (self.df_cons_with_pv, 'census_id'),
            (self.df_self_cons_pct_pv, 'census_id'),
            (self.df_cov_pct_no_pv, 'census_id'),
            (self.df_cov_pct_with_pv, 'census_id'),
            (self.df_cons_no_pv, 'census_id')
        ]
        
        energy_data = pv_gen
        for df, key in merge_list:
            energy_data = pd.merge(energy_data, df, on=key, how='left') #Uses how='left', which ensures all census_ids from pv_gen are retained.

        self.energy_data = energy_data

    def calculate_costs(self):
        # Step 3: Calculate self-consumption (direct consumption) and residuals
        for i in range(1, 13):
            p_electr = self.df_prices.loc[i, 'Electricity Buy']
            p_feed = self.df_prices.loc[i, 'Electricity Sell']
            p_electr_EC_buy = self.df_prices.loc[i, 'Electricity Buy'] * (1 - self.ec_price_coef)
            p_electr_EC_sell = self.df_prices.loc[i, 'Electricity Sell'] * (1 + self.ec_price_coef)
            gen_col = f'gen_m{i}'
            cons_col = f'cons_m{i}'
            cons_no_pv_col = f'no_pv_cons_m{i}'
            self_col = f'self_m{i}'
            no_pv_pct_cov_col = f'no_pv_cov_m{i}'
            pv_pct_cov_col = f'with_pv_cov_m{i}'

            # Direct self-consumption
            self.energy_data[f'gen_used_dir_m{i}'] = self.energy_data[gen_col] * self.energy_data[self_col]
            self.energy_data[f'load_cov_dir_m{i}'] = self.energy_data[f'gen_used_dir_m{i}']

            # Residual load and surplus PV generation after direct consumption
            self.energy_data[f'gen_surpl_dir_m{i}'] = self.energy_data[gen_col] - self.energy_data[f'load_cov_dir_m{i}']
            self.energy_data[f'load_resid_dir_m{i}'] = self.energy_data[cons_col] - self.energy_data[f'load_cov_dir_m{i}']

            # Load covered within EC for non-PV households
            self.energy_data[f'load_cov_EC_BwoPV_m{i}'] = self.energy_data[gen_col] * self.energy_data[no_pv_pct_cov_col]
            self.energy_data[f'load_resid_EC_BwoPV_m{i}'] = self.energy_data[cons_no_pv_col] - self.energy_data[f'load_cov_EC_BwoPV_m{i}']

            # Load covered within EC for PV households
            self.energy_data[f'load_cov_EC_BwPV_m{i}'] = self.energy_data[gen_col] * self.energy_data[pv_pct_cov_col]
            self.energy_data[f'load_resid_EC_BwPV_m{i}'] = self.energy_data[f'load_resid_dir_m{i}'] - self.energy_data[f'load_cov_EC_BwPV_m{i}']

            # Surplus generation after local consumption within the EC
            self.energy_data[f'gen_surpl_EC_m{i}'] = self.energy_data[f'gen_surpl_dir_m{i}'] - self.energy_data[f'load_cov_EC_BwPV_m{i}'] - self.energy_data[f'load_cov_EC_BwoPV_m{i}']

            # Set possible negative values to zero
            for col in [
                f'gen_surpl_dir_m{i}', f'load_resid_dir_m{i}', f'load_cov_EC_BwoPV_m{i}',
                f'load_resid_EC_BwoPV_m{i}', f'load_cov_EC_BwPV_m{i}', f'load_resid_EC_BwPV_m{i}', f'gen_surpl_EC_m{i}'
            ]:
                self.energy_data[col] = self.energy_data[col].clip(lower=0)

            # Conventional costs without EC
            self.energy_data[f'C_BwPV_m{i}'] = self.energy_data[f'load_resid_dir_m{i}'] * p_electr - self.energy_data[f'gen_surpl_dir_m{i}'] * p_feed
            self.energy_data[f'C_BwoPV_m{i}'] = self.energy_data[cons_no_pv_col] * p_electr
            if self.pv_pct == 1:
                self.energy_data[f'C_BwoPV_m{i}'] = self.energy_data[cons_col] * p_electr

            # Costs with EC for PV households
            self.energy_data[f'C_EC_BwPV_m{i}'] = (
                self.energy_data[f'load_resid_EC_BwPV_m{i}'] * p_electr
                + self.energy_data[f'load_cov_EC_BwPV_m{i}'] * p_electr_EC_buy
                - (self.energy_data[f'load_cov_EC_BwPV_m{i}'] + self.energy_data[f'load_cov_EC_BwoPV_m{i}']) * p_electr_EC_sell
                - self.energy_data[f'gen_surpl_EC_m{i}'] * p_feed
            )

            # Costs with EC for non-PV households
            self.energy_data[f'C_EC_BwoPV_m{i}'] = (
                self.energy_data[f'load_resid_EC_BwoPV_m{i}'] * p_electr
                + self.energy_data[f'load_cov_EC_BwoPV_m{i}'] * p_electr_EC_buy
            )
            if self.pv_pct == 1:
                self.energy_data[f'C_EC_BwPV_m{i}'] = (
                    self.energy_data[f'load_resid_EC_BwPV_m{i}'] * p_electr
                    + self.energy_data[f'load_cov_EC_BwPV_m{i}'] * p_electr_EC_buy
                    - (self.energy_data[f'load_cov_EC_BwPV_m{i}'] + self.energy_data[f'load_cov_EC_BwoPV_m{i}']) * p_electr_EC_sell
                    - self.energy_data[f'gen_surpl_EC_m{i}'] * p_feed
                )

        self.energy_data = self.energy_data.set_index('census_id')

    def export_to_excel(self):
        # Create a directory for results if it doesn't exist
        results_dir = os.path.join(self.root, "data", '06_enercom_estimator')
        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        # Create Excel writer
        with pd.ExcelWriter(f'{results_dir}/EC_costs_results_{self.pv_pct}_price_diff_{self.ec_price_coef}.xlsx') as writer:
            # Sheet 1: gen_consumption
            gen_consumption_cols = [f'gen_m{i}' for i in range(1, 13)] + [f'cons_m{i}' for i in range(1, 13)] + [f'self_m{i}' for i in range(1, 13)] + [f'gen_used_dir_m{i}' for i in range(1, 13)] + [f'load_cov_dir_m{i}' for i in range(1, 13)] + [f'gen_surpl_dir_m{i}' for i in range(1, 13)] + [f'load_resid_dir_m{i}' for i in range(1, 13)]
            self.energy_data.reset_index()[['census_id'] + gen_consumption_cols].to_excel(writer, sheet_name='gen_consumption', index=False)

            # Sheet 2: loadsinEC
            loads_inEC_cols = [f'load_cov_EC_BwoPV_m{i}' for i in range(1, 13)] + [f'load_resid_EC_BwoPV_m{i}' for i in range(1, 13)] + [f'load_cov_EC_BwPV_m{i}' for i in range(1, 13)] + [f'load_resid_EC_BwPV_m{i}' for i in range(1, 13)] + [f'gen_surpl_EC_m{i}' for i in range(1, 13)]
            self.energy_data.reset_index()[['census_id'] + loads_inEC_cols].to_excel(writer, sheet_name='loads_within_EC', index=False)

            # Sheet 3: costs_without_EC
            costs_without_EC_cols = [f'C_BwPV_m{i}' for i in range(1, 13)] + [f'C_BwoPV_m{i}' for i in range(1, 13)]
            self.energy_data.reset_index()[['census_id'] + costs_without_EC_cols].to_excel(writer, sheet_name='costs_without_EC', index=False)

            # Sheet 4: cost_with_EC
            cost_with_EC_cols = [f'C_EC_BwPV_m{i}' for i in range(1, 13)] + [f'C_EC_BwoPV_m{i}' for i in range(1, 13)]
            self.energy_data.reset_index()[['census_id'] + cost_with_EC_cols].to_excel(writer, sheet_name='costs_with_EC', index=False)

    def analyze(self, plot = True, title="Estimated Costs Comparison"):
        # Plotting
        df_total_cost = pd.DataFrame(index=self.energy_data.index)
        df_total_cost['Avg Costs, dwellings with PV'] = self.energy_data[[f'C_BwPV_m{i}' for i in range(1, 13)]].sum(axis=1)
        df_total_cost['Avg Costs, dwellings without PV'] = self.energy_data[[f'C_BwoPV_m{i}' for i in range(1, 13)]].sum(axis=1)
        df_total_cost['Avg Costs in EC, dwellings with PV'] = self.energy_data[[f'C_EC_BwPV_m{i}' for i in range(1, 13)]].sum(axis=1)
        df_total_cost['Avg Costs in EC, dwellings without PV'] = self.energy_data[[f'C_EC_BwoPV_m{i}' for i in range(1, 13)]].sum(axis=1)
        
        if plot:
            # Plot using util_func
            fig = util_func.EconomicAnalysisGraphs.plot_ec_costs(
                df_total_cost,
                C_ec_b_PV='Avg Costs in EC, dwellings with PV',
                C_ec_b_nPV='Avg Costs in EC, dwellings without PV',
                C_cov_b_PV='Avg Costs, dwellings with PV',
                C_cov_b_nPV='Avg Costs, dwellings without PV',
                title=title
            )
            # Save the figure if returned
            if not os.path.exists('data/06_enercom_estimator/img'):
                os.makedirs('data/06_enercom_estimator/img')
                plt.tight_layout()
                fig.savefig(f'data/06_enercom_estimator/img/EC_costs_dwell_share_{self.pv_pct}_price_diff_{self.ec_price_coef}.png', bbox_inches='tight')
            else:
                plt.tight_layout()
                fig.savefig(f'data/06_enercom_estimator/img/EC_costs_dwell_share_{self.pv_pct}_price_diff_{self.ec_price_coef}.png', bbox_inches='tight')

        # Normalize by number of dwellings
        df_total_cost['Avg Costs, dwellings with PV'] = df_total_cost['Avg Costs, dwellings with PV'].div(
            self.stat_data['Total number of dwellings'] * float(self.pv_pct), axis=0)
        if self.pv_pct == 1:
            df_total_cost['Avg Costs, dwellings without PV'] = (
                df_total_cost['Avg Costs, dwellings without PV'].div(self.stat_data['Total number of dwellings'] * float(self.pv_pct), axis=0)
            )
        else:
            df_total_cost['Avg Costs, dwellings without PV'] = (
                df_total_cost['Avg Costs, dwellings without PV'].div(self.stat_data['Total number of dwellings'] * float(1 - self.pv_pct), axis=0)
            )
        df_total_cost['Avg Costs in EC, dwellings with PV'] = (
            df_total_cost['Avg Costs in EC, dwellings with PV'].div(self.stat_data['Total number of dwellings'] * float(self.pv_pct), axis=0)
        )
        if self.pv_pct == 1:
            df_total_cost['Avg Costs in EC, dwellings without PV'] = (
                df_total_cost['Avg Costs in EC, dwellings without PV'].div(self.stat_data['Total number of dwellings'] * float(self.pv_pct), axis=0)
            )
        else:
            df_total_cost['Avg Costs in EC, dwellings without PV'] = (
                df_total_cost['Avg Costs in EC, dwellings without PV'].div(self.stat_data['Total number of dwellings'] * float(1 - self.pv_pct), axis=0)
            )

        # Calculate savings
        if self.pv_pct == 1:
            df_total_cost['Savings, Dwell with PV, %'] = df_total_cost.apply(
                lambda row: (
                    (1 - (row['Avg Costs in EC, dwellings with PV'] / row['Avg Costs, dwellings without PV'])) * 100
                    if (row['Avg Costs, dwellings with PV'] * row['Avg Costs in EC, dwellings with PV'] < 0)
                    else ((row['Avg Costs, dwellings without PV'] - row['Avg Costs in EC, dwellings with PV']) / row['Avg Costs, dwellings without PV']) * 100
                ) if row['Avg Costs, dwellings with PV'] != 0 else 0,
                axis=1
            )
            df_total_cost['Savings, Dwell without PV, %'] = 0
        else:
            df_total_cost['Savings, Dwell with PV, %'] = df_total_cost.apply(
                lambda row: (
                    (1 - (row['Avg Costs in EC, dwellings with PV'] / row['Avg Costs, dwellings with PV'])) * 100
                    if (row['Avg Costs, dwellings with PV'] * row['Avg Costs in EC, dwellings with PV'] < 0)
                    else ((row['Avg Costs, dwellings with PV'] - row['Avg Costs in EC, dwellings with PV']) / row['Avg Costs, dwellings with PV']) * 100
                ) if row['Avg Costs, dwellings with PV'] != 0 else 0,
                axis=1
            )
            df_total_cost['Savings, Dwell without PV, %'] = df_total_cost.apply(
                lambda row: (
                    (1 - (row['Avg Costs in EC, dwellings without PV'] / row['Avg Costs, dwellings without PV'])) * 100
                    if (row['Avg Costs, dwellings without PV'] * row['Avg Costs in EC, dwellings without PV'] < 0)
                    else ((row['Avg Costs, dwellings without PV'] - row['Avg Costs in EC, dwellings without PV']) / row['Avg Costs, dwellings without PV']) * 100
                ) if row['Avg Costs, dwellings without PV'] != 0 else 0,
                axis=1
            )

        #plot savings
        #self.plot_saving(df_total_cost)

        # Append DF
        df_total_cost['Dwellings share with PV'] = self.pv_pct
        df_total_cost['Price diff'] = self.ec_price_coef
        self.df_sensitivity_total = self.df_sensitivity_total.append(df_total_cost)
        
        return df_total_cost

    def plot_saving(self,df_total_cost):
        # Plot savings
        plot = df_total_cost[['Savings, Dwell with PV, %', 'Savings, Dwell without PV, %']].plot(
            kind='bar', ylabel="Savings in %", figsize=(10, 6),
            title=f"Savings in % for {self.pv_pct * 100}% of dwellings with PV, price_diff:{self.ec_price_coef * 100}%",
            color=['blue', 'orange'])
        plot.set_xticklabels(df_total_cost.index, rotation=45)

        fig = util_func.EconomicAnalysisGraphs.plot_ec_costs(
            df_total_cost,
            C_ec_b_PV='Avg Costs in EC, dwellings with PV',
            C_ec_b_nPV='Avg Costs in EC, dwellings without PV',
            C_cov_b_PV='Avg Costs, dwellings with PV',
            C_cov_b_nPV='Avg Costs, dwellings without PV',
            title=f"Estimated Costs for `Scenario {int(self.pv_pct * 100)}%`, price_diff:{int(self.ec_price_coef * 100)}%"
        )
        # Save the figure if returned
        if not os.path.exists('data/06_enercom_estimator/img'):
            os.makedirs('data/06_enercom_estimator/img')
            plt.tight_layout()
            fig.savefig(f'data/06_enercom_estimator/img/EC_costs_per_dwelling_dwell_share_{self.pv_pct}_price_diff_{self.ec_price_coef}.png', bbox_inches='tight')
        else:
            plt.tight_layout()
            fig.savefig(f'data/06_enercom_estimator/img/EC_costs_per_dwelling_dwell_share_{self.pv_pct}_price_diff_{self.ec_price_coef}.png', bbox_inches='tight')


    def monthly_sensitivity_analysis(self):
        # Monthly Sensitivity Analysis
        df_temp = pd.DataFrame()
        df_temp = self.energy_data[[f'C_EC_BwoPV_m{i}' for i in range(1, 13)]]
        df_temp = pd.concat([df_temp, self.energy_data[[f'C_EC_BwPV_m{i}' for i in range(1, 13)]]], axis=1)
        df_temp = pd.concat([df_temp, self.energy_data[[f'C_BwoPV_m{i}' for i in range(1, 13)]]], axis=1)
        df_temp = pd.concat([df_temp, self.energy_data[[f'C_BwPV_m{i}' for i in range(1, 13)]]], axis=1)
        df_temp['Dwellings share with PV'] = self.pv_pct
        df_temp['Price diff'] = self.ec_price_coef
        self.df_sensitivity_vulner_zones_monthly = pd.concat([self.df_sensitivity_vulner_zones_monthly, df_temp])

    def save_sensitivity(self,df_sensitivity_append=None):
        if df_sensitivity_append is None:
            self.df_sensitivity_total.to_excel(f'data/06_enercom_estimator/sensitivity_analysis_{self.pv_pct}_price_diff_{self.ec_price_coef}.xlsx')
        else:
            df_sensitivity_append.to_excel(f'data/06_enercom_estimator/sensitivity_analysis.xlsx')

    @staticmethod
    def plot_ec_costs(df):
        # Test plotting function
        fig, ax = plt.subplots()
        df[['C_ec_b_PV', 'C_ec_b_nPV']].plot(kind='bar', ax=ax, width=0.7)
        for i, row in df.iterrows():
            ax.text(i - 0.15, row['C_ec_b_PV'] + 0.05, round(row['C_ec_b_PV'], 2), color='black', ha='center')
            ax.text(i + 0.15, row['C_ec_b_nPV'] + 0.05, round(row['C_ec_b_nPV'], 2), color='black', ha='center')
        for i, value in enumerate(df['C_cov_b_PV']):
            plt.plot([i - 0.3, i + 0.3], [value, value], color='blue')
        for i, value in enumerate(df['C_cov_b_nPV']):
            plt.plot([i - 0.3, i + 0.3], [value, value], color='red')
        ax.scatter(df.index, df['C_cov_b_PV'], color='red', marker='_', label='C_cov_b_nPV')
        ax.scatter(df.index, df['C_cov_b_nPV'], color='blue', marker='_', label='C_cov_b_PV')
        ax.set_xlabel('Use case')
        ax.set_ylabel('Costs EUR')
        ax.set_title('Comparing estimated costs')
        ax.set_xticklabels(df.index, rotation=0)
        ax.legend(loc='best', bbox_to_anchor=(1, 1)).set_visible(True)
        plt.show()

    @staticmethod
    def plot_savings_distribution(df, values = "Savings, Dwell with PV, %",colormap="blue"):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        price_diff_values = df["Price diff"].unique()
        price_diff_values.sort()
    # Iterate over each subplot and plot the corresponding bar plot
        for i, ax in enumerate(axes.flat):
            if i < len(price_diff_values):
                price_diff = price_diff_values[i]
                df_subset = df[df["Price diff"] == price_diff]
            
            # Creating a bar plot for Savings
                df_subset_pivot = df_subset.pivot(index="Dwellings share with PV", columns="census_id", values=values)
                df_subset_pivot.plot(kind="bar", ax=ax, colormap=colormap, alpha=0.7, width=0.8, fontsize=14)
            
            # Titles and labels
                ax.set_title(f"EC price sell: {int(price_diff*100)}% more grid feed-in, EC price buy: {int(price_diff*100)}% less grid electricity", fontsize=14)
                ax.set_xlabel("Scenario [i.e. Dwellings Share associated with collective (direct) PV self-consumption]" , fontsize=14)
                ax.set_ylabel("Savings (%)", fontsize=14)
                ax.grid(True)
                ax.legend(title="Census ID", loc="best")
                ax.set_xticklabels(labels=[int(s*100) for s in list(df_subset_pivot.index)], rotation=0, fontsize=14)
    # Adjust layout for better spacing
        plt.tight_layout()
    # Show the plot
        plt.show()
#%%
if __name__ == "__main__":
    # Example usage
    ec_price_coef_list = [0.1, 0.25, 0.5]
    pv_pct_list = [0.25, 0.5, 0.75, 1]
    root_dir = r"C:\Users\Oleksandr-MSI\Documentos\GitHub\spacer-hb-framework"
    df_sensitivity_append = pd.DataFrame()
    for ec_price_coef in ec_price_coef_list:
        for pv_pct in pv_pct_list:
            print(f"Processing for PV percentage: {pv_pct}, EC price coefficient: {ec_price_coef}")
            estimator = EnercomEstimator(pv_pct=pv_pct, root=root_dir, ec_price_coef=ec_price_coef)
            estimator.prepare_energy_data()
            estimator.calculate_costs()
            estimator.export_to_excel()
            estimator.analyze_and_plot()
            estimator.monthly_sensitivity_analysis()
            estimator.save_sensitivity()
            df_sensitivity_append = df_sensitivity_append.append(estimator.analyze_and_plot())
        
    estimator.save_sensitivity(df_sensitivity_append)
    #vulnerable_zones=["4802003006", "4802003007","4802003009","4802003010"]
    #df = df_sensitivity_append.loc[df_sensitivity_append.index.isin(vulnerable_zones)]
    #df = df.reset_index()
    df = df_sensitivity_append.reset_index()
    print ("Percentage of energy cost savings per scenario and share of associated with collective (direct) PV self-consumption in the census zone")
    estimator.plot_savings_distribution(df,values = "Savings, Dwell with PV, %", colormap='cividis')
    
    print ("Percentage of energy cost savings per scenario and share of dwellings not associated with collective (direct) PV self-consumption in the census zone")
    estimator.plot_savings_distribution(df,values = "Savings, Dwell without PV, %", colormap='copper')
    #%%
    df_sensitivity_append = pd.DataFrame()
    for ec_price_coef in ec_price_coef_list:
        for pv_pct in pv_pct_list:
            estimator = EnercomEstimator(pv_pct=pv_pct, root=root_dir, ec_price_coef=ec_price_coef)
            estimator.prepare_energy_data()
            estimator.calculate_costs()
            estimator.export_to_excel()
            estimator.analyze_and_plot()
            estimator.monthly_sensitivity_analysis()
            df_sensitivity_append = estimator.analyze_and_plot()#df_sensitivity_append.append(estimator.analyze_and_plot())
    
    estimator.save_sensitivity(df_sensitivity_append)
# %%
