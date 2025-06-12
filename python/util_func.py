import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties
import plotly.express as px


class UtilFunctions:
    def __init__(self):
        self.path = "economic_calc/self_cons_estim/"
        #self.files = os.listdir("economic_calc/cons/")
        
    @staticmethod
    def save_to_excel(df, filename="template.xlsx", sheet_name="template"):
        with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:  
            df.to_excel(writer, sheet_name=sheet_name)

    @staticmethod
    def add_sheet_to_excel(df, filename="template.xlsx", sheet_name="template2", index=False):
        with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists= "error") as writer:  
            df.to_excel(writer, sheet_name=sheet_name, index=index)
    
    def load_data(self, path, files):
        df_self_cons_pct = pd.DataFrame()
        dwelling_accounted_pct = []
        
        for file in files:
            if file.endswith('.csv'):
                df_temp = pd.read_csv(os.path.join(path, file), index_col=0)
                symbol = file.split('_')[-1].split('.csv')[0]
                df_temp["%dwelling_accounted"] = symbol
                dwelling_accounted_pct.append(symbol)
                print(file, " : ", symbol)
                df_self_cons_pct = pd.concat([df_self_cons_pct, df_temp], axis=0)
        
        return df_self_cons_pct, dwelling_accounted_pct

class PlotFunctions:
    @staticmethod
    def plot_self_consumption_trends(df_calc_self_cons_per_dwelling, df_self_cons_pct_, xlabel, ylabel, header):

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
    
    @staticmethod
    def plot_combined_load_data(data, categories, labels):
        """
        Plot load data as combined individual and stacked bar chart.

        Parameters:
            data (list of lists): 2D list where:
                                data[0] corresponds to baseline load,
                                data[1] corresponds to total load,
                                data[2:] corresponds to stacked components (direct consumption, EC, residual).
            categories (list): Names of the groups (x-axis labels).
            labels (list): Labels for the individual bars and stacked components.
        """
        # Number of categories (x-axis labels)
        n_categories = len(categories)

        # Bar width and x-axis positions
        bar_width = 0.3  # Width of each bar
        x_indices = np.arange(n_categories)

        # Colors for the bars
        colors = ['gray', 'blue', 'orange', 'green', 'red']

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the baseline load as a separate bar
        ax.bar(x_indices, data[0], width=bar_width, label=labels[0], color=colors[0])

        # Plot the total load as another separate bar
        ax.bar(x_indices + bar_width, data[1], width=bar_width, label=labels[1], color=colors[1])

        # Plot the stacked components
        bottom = np.zeros(n_categories)
        for i in range(2, len(data)):
            ax.bar(x_indices + 2 * bar_width, data[i], width=bar_width, bottom=bottom, label=labels[i], color=colors[i])
            bottom += np.array(data[i])  # Update the bottom for stacking

        # Add labels, legend, and grid
        ax.set_xlabel('Households', fontsize=12)
        ax.set_ylabel('Load in kWh', fontsize=12)
        ax.set_xticks(x_indices + 1.5 * bar_width)
        ax.set_xticklabels(categories, fontsize=10)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_combined_load_data_from_df(df, title):
        """
        Plot load data as a combined individual and stacked bar chart using a DataFrame.

        The DataFrame should have:
        - Columns as categories (e.g., 'HH1', 'HH2', 'HH3', ...)
        - Index as labels (e.g., ['baseline load', 'total load', 'direct consumption', 'EC consumption', 'residual load']).
        
        Parameters:
            df (DataFrame): DataFrame with rows as labels and columns as categories.
        """
        # Extract categories (column names) and labels (index)
        categories = df.columns.tolist()  # Categories are column names (e.g., HH1, HH2, HH3, ...)
        labels = df.index.tolist()  # Labels are the row names (e.g., baseline load, total load, etc.)

        # Number of categories (x-axis labels)
        n_categories = len(categories)

        # Bar width and x-axis positions
        bar_width = 0.3  # Width of each bar
        x_indices = np.arange(n_categories)

        # Colors for the bars (adjust if more labels are added)
        colors = ['gray', 'blue', 'orange', 'green', 'red']

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the baseline load as a separate bar
        ax.bar(x_indices, df.iloc[0], width=bar_width, label=labels[0], color=colors[0])

        # Plot the total load as another separate bar
        ax.bar(x_indices + bar_width, df.iloc[1], width=bar_width, label=labels[1], color=colors[1])

        # Plot the stacked components
        bottom = np.zeros(n_categories)
        for i in range(2, len(df)):
            ax.bar(x_indices + 2 * bar_width, df.iloc[i], width=bar_width, bottom=bottom, label=labels[i], color=colors[i])
            bottom += np.array(df.iloc[i])  # Update the bottom for stacking

        # Add labels, legend, and grid
        ax.set_xlabel('Month', fontsize=14)
        ax.set_ylabel('Load in kWh', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(x_indices + 1.5 * bar_width)
        ax.set_xticklabels(categories, fontsize=14)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
    @staticmethod
    def plot_combined_load_data_from_df_adj(df, title, legend_show=True, y_min=None, y_max=None):
        """
        Plot load data as a combined individual and stacked bar chart using a DataFrame.

        The DataFrame should have:
        - Columns as categories (e.g., 'HH1', 'HH2', 'HH3', ...)
        - Index as labels (e.g., ['baseline load', 'total load', 'direct consumption', 'EC consumption', 'residual load']).
        
        Parameters:
            df (DataFrame): DataFrame with rows as labels and columns as categories.
            title (str): Title of the plot.
            y_min (float, optional): Minimum value for the y-axis.
            y_max (float, optional): Maximum value for the y-axis.
        """
        # Extract categories (column names) and labels (index)
        categories = df.columns.tolist()  # Categories are column names (e.g., HH1, HH2, HH3, ...)
        labels = df.index.tolist()  # Labels are the row names (e.g., baseline load, total load, etc.)

        # Number of categories (x-axis labels)
        n_categories = len(categories)

        # Bar width and x-axis positions
        bar_width = 0.3  # Width of each bar
        x_indices = np.arange(n_categories)

        # Colors for the bars (adjust if more labels are added)
        colors = ['gray', 'blue', 'orange', 'green', 'red']

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the baseline load as a separate bar
        ax.bar(x_indices, df.iloc[0], width=bar_width, label=labels[0], color=colors[0])

        # Plot the total load as another separate bar
        ax.bar(x_indices + bar_width, df.iloc[1], width=bar_width, label=labels[1], color=colors[1])

        # Plot the stacked components
        bottom = np.zeros(n_categories)
        for i in range(2, len(df)):
            ax.bar(x_indices + 2 * bar_width, df.iloc[i], width=bar_width, bottom=bottom, label=labels[i], color=colors[i])
            bottom += np.array(df.iloc[i])  # Update the bottom for stacking

        # Add labels, legend, and grid
        ax.set_xlabel('Month', fontsize=14)
        ax.set_ylabel('Load in kWh', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(x_indices + 1.5 * bar_width)
        ax.set_xticklabels(categories, fontsize=14)
        ax.legend(fontsize=10, loc='upper left').set_visible(legend_show)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Set y-axis limits if provided
        if y_min is not None or y_max is not None:
            ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.show()
    '''
    # Sample data in DataFrame format
    data = {
        'HH1': [7500, 8000, 2000, 100, 5900],
        'HH2': [6800, 7000, 1500, 80, 5420],
        'HH3': [8200, 8500, 1800, 120, 6580],
        'HH4': [8000, 8200, 1700, 110, 6390],
        'HH5': [7700, 7900, 1600, 90, 6210],
        'HH6': [7600, 7800, 1500, 70, 6230],
        'HH7': [7400, 7600, 1400, 80, 6120],
        'HH8': [7500, 7700, 1450, 85, 6165],
        'HH9': [7800, 8000, 1550, 95, 6355],
        'HH10': [7200, 7500, 1300, 75, 6125]
    }

    # Labels for rows
    labels = ['baseline load', 'total load', 'load covered by direct consumption', 'load covered within EC', 'residual load']

    # Create the DataFrame
    df = pd.DataFrame(data, index=labels)

    # Plot the data
    plot_combined_load_data_from_df(df)
    '''
    @staticmethod
    def plot_stacked_load_data_2(data, categories, labels):
        """
        Plot load data as a combined stacked and single bar chart.

        Parameters:
            data (list of lists): 2D list where:
                                data[0] corresponds to total load,
                                data[1:] corresponds to stacked components (direct consumption, EC, residual).
            categories (list): Names of the groups (x-axis labels).
            labels (list): Labels for the stacked components and total load.
        """
        # Number of categories (x-axis labels)
        n_categories = len(categories)

        # Bar width and x-axis positions
        bar_width = 0.4  # Width of each bar
        x_indices = np.arange(n_categories)

        # Colors for the bars
        colors = ['blue', 'orange', 'green', 'red']

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the total load as a separate bar
        ax.bar(x_indices, data[0], width=bar_width, label=labels[0], color=colors[0])

        # Plot the stacked components
        bottom = np.zeros(n_categories)
        for i in range(1, len(data)):
            ax.bar(x_indices + bar_width, data[i], width=bar_width, bottom=bottom, label=labels[i], color=colors[i])
            bottom += np.array(data[i])  # Update the bottom for stacking

        # Add labels, legend, and grid
        ax.set_xlabel('Households', fontsize=12)
        ax.set_ylabel('Load in kWh', fontsize=12)
        ax.set_xticks(x_indices + bar_width / 2)
        ax.set_xticklabels(categories, fontsize=10)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
    """
        # Example Usage
        # Data: First list is total load, the rest are stacked components
        data = [
            [8000, 7000, 8500, 8200, 7900, 7800, 7600, 7700, 8000, 7500],  # total load
            [2000, 1500, 1800, 1700, 1600, 1500, 1400, 1450, 1550, 1300],  # load covered by direct consumption
            [100, 80, 120, 110, 90, 70, 80, 85, 95, 75],                   # load covered within EC
            [5900, 5420, 6580, 6390, 6210, 6230, 6120, 6165, 6355, 6125]   # residual load
        ]

        categories = ['HH1', 'HH2', 'HH3', 'HH4', 'HH5', 'HH6', 'HH7', 'HH8', 'HH9', 'HH10']
        labels = ['total load', 'load covered by direct consumption', 'load covered by consumption within EC', 'residual load']

        plot_stacked_load_data_2(data, categories, labels)
    """
    
    @staticmethod
    def bar_plot_month_pv_columns_interactive(dataframe, column_names, title="Data Distribution", xlabel="Columns", ylabel="Values", font_size=14):
        """
        Plots a bar chart for the specified columns in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            column_names (list): A list of column names to plot.
            title (str): The title of the chart.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            font_size (int): The font size for the title and labels.
        """
        # Prepare data for Plotly
        values = dataframe[column_names].iloc[0]
        fig = px.bar(
            x=column_names,
            y=values,
            labels={'x': xlabel, 'y': ylabel},
            title=title,
            text=values
        )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            font=dict(size=font_size),
            title_font=dict(size=font_size + 2),
            bargap=0.2,
            showlegend=False
        )
        fig.show()
        
    @staticmethod
    def bar_plot_month_pv_columns(dataframe, column_names, title="Data Distribution", xlabel="Columns", ylabel="Values", font_size=14):
        """
        Plots a bar chart for the specified columns in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            column_names (list): A list of column names to plot.
            title (str): The title of the chart.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            font_size (int): The font size for the title and labels.
        """
        font_size = font_size
        plt.figure(figsize=(12, 6))
        # Ensure column names and tick positions are correctly aligned
        ticks = range(len(column_names))
        plt.bar(ticks, dataframe[column_names].iloc[0])
        plt.title(title, fontsize=font_size)
        plt.xlabel(xlabel, fontsize=font_size)
        plt.ylabel(ylabel, fontsize=font_size)
        plt.xticks(ticks=ticks, labels=column_names, rotation=0, fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_month_pv_columns_line(dataframe, column_names, title="Data Distribution", xlabel="Columns", ylabel="Values", font_size=14):
        """
        Plots a bar chart and a line plot for the specified columns in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            column_names (list): A list of column names to plot.
            title (str): The title of the chart.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            font_size (int): The font size for the title and labels.
        """
        plt.figure(figsize=(16, 6))
        ticks = range(len(column_names))
        
        # Bar plot
        plt.bar(ticks, dataframe[column_names].iloc[0], alpha=0.6, label='Bar Plot')
        
        # Line plot
        plt.plot(ticks, dataframe[column_names].iloc[0], marker='o', color='r', label='Line Plot')
        
        plt.title(title, fontsize=font_size)
        plt.xlabel(xlabel, fontsize=font_size)
        plt.ylabel(ylabel, fontsize=font_size)
        plt.xticks(ticks=ticks, labels=column_names, rotation=0, fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.legend(fontsize=font_size)
        #plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_census_stackplot(dataframe, title="Stackplot", xlabel="X-axis", ylabel="Y-axis", font_size=14):
        """
        Plots a stackplot for the given DataFrame.

        Args:
            dataframe (pd.DataFrame): DataFrame where columns are x-axis values (i.e., Months, int type) and rows are groups for stacking (i.e., Census IDs, str type).
            title (str): The title of the plot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
        """
        dataframe.columns = dataframe.columns.astype(str)
        dataframe = dataframe.T
        # Extract x-axis values and y-axis groups from the DataFrame
        x = dataframe.index  # x-axis (e.g., Months)
        print(x)
        y = dataframe.values.T  # y-axis values for stacking (transpose for stackplot)
        print(y)

        # Plot the stackplot
        plt.figure(figsize=(12, 6))
        plt.stackplot(x, y, labels=dataframe.columns)
        plt.title(title, fontsize=font_size)
        plt.xlabel(xlabel, fontsize=font_size)
        plt.ylabel(ylabel, fontsize=font_size)
        plt.legend(loc="upper left", title="Census ID")
        plt.xticks(x, rotation=0, fontsize=font_size)
        plt.yticks(rotation=0, fontsize=font_size)
        #plt.tight_layout()
        plt.show()

class PVAnalysis:
    def __init__(self):
        pass

    #Calculation of eqatuions
    def pv_size_to_cost_equation(self, pv_size):
        pv_cost_per_kw= -133.4*np.log(pv_size) + 1095.7
        return pv_cost_per_kw * pv_size #€/kWp
    
    # Function to calculate PV_self_cons_Euro and PV_to_grid_Euro
    def calculate_pv_metrics(self,df_pv, df_prices, monthly_self_cons_percentages):
        monthly_columns = list(range(1, 13))  # Monthly columns
        results = []

        for index, row in df_pv.iterrows():
            pv_self_cons_kWh = 0
            pv_to_grid_kWh = 0
            pv_self_cons_euro = 0
            pv_to_grid_euro = 0
            
            for month in monthly_columns:
                monthly_generation = row[month]
                buy_price = df_prices.loc[month - 1, 'Electricity Buy']  # Match price by month
                sell_price = df_prices.loc[month - 1, 'Electricity Sell']
                
                # Get the self-consumption percentage for the current month
                month_name = df_prices.loc[month - 1, 'Month']
                self_cons_percent = monthly_self_cons_percentages[month_name]
                
                # Self-consumed and exported energy
                self_consumed_energy_kWh = monthly_generation * self_cons_percent
                exported_energy_kWh = monthly_generation * (1 - self_cons_percent)
                
                # Add to total kWh
                pv_self_cons_kWh += self_consumed_energy_kWh
                pv_to_grid_kWh += exported_energy_kWh
                
                # Monetary values
                pv_self_cons_euro += self_consumed_energy_kWh * buy_price
                pv_to_grid_euro += exported_energy_kWh * sell_price
            
            # Append results
            results.append({
                'build_id': row['build_id'],
                'census_id': row['census_id'],
                'PV_self_cons_kWh': round(pv_self_cons_kWh, 3),
                'PV_to_grid_kWh': round(pv_to_grid_kWh, 3),
                'PV_self_cons_Euro': round(pv_self_cons_euro, 3),
                'PV_to_grid_Euro': round(pv_to_grid_euro, 3)
            })
        
        return pd.DataFrame(results)
    
    def calculate_avg_production(self, df, initial_production_col, degradation_rate, lifespan):
        """
        Calculate the average annual production considering degradation for each row in the DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the input data.
        - initial_production_col (str): Column name for initial annual production (e.g., kWh or Euro).
        - degradation_rate (float): Annual degradation rate (e.g., 0.005 for 0.5%).
        - lifespan (int): Expected lifespan of the PV system (in years).

        Returns:
        - pd.DataFrame: DataFrame with an additional column for average annual production.
        """
        def calculate_row_average(row):
            initial_production = row[initial_production_col]
            total_production = sum(
                initial_production * (1 - degradation_rate) ** (t - 1) for t in range(1, lifespan + 1)
            )
            return total_production / lifespan

        df['PV_avg_annual_kWh'] = df.apply(calculate_row_average, axis=1)
        return df
    
    def calculate_pv_metrics_by_census(self, df_pv, df_prices, monthly_self_cons_percentages_by_census):
        """
        Calculate PV_self_cons_kWh, PV_to_grid_kWh, PV_self_cons_Euro, and PV_to_grid_Euro
        based on monthly self-consumption percentages per census_id.

        Parameters:
            df_pv (pd.DataFrame): DataFrame containing PV generation data.
            df_prices (pd.DataFrame): DataFrame containing electricity prices by month.
            monthly_self_cons_percentages_by_census (dict): Dictionary where keys are census_id
                and values are dictionaries of monthly self-consumption percentages.

        Returns:
            pd.DataFrame: DataFrame with calculated PV metrics.
        """
        monthly_columns = list(range(1, 13))  # Monthly columns (1-12 for January-December)
        results = []

        for index, row in df_pv.iterrows():
            census_id = row['census_id']
            if census_id not in monthly_self_cons_percentages_by_census:
                raise ValueError(f"No self-consumption percentages provided for census_id {census_id}")
            
            # Get the self-consumption percentages for this census_id
            self_cons_percentages = monthly_self_cons_percentages_by_census[census_id]

            pv_self_cons_kWh = 0
            pv_to_grid_kWh = 0
            pv_self_cons_euro = 0
            pv_to_grid_euro = 0
            
            for month in monthly_columns:
                monthly_generation = row[month]
                buy_price = df_prices.loc[month - 1, 'Electricity Buy']  # Match price by month
                sell_price = df_prices.loc[month - 1, 'Electricity Sell']
                
                # Get the self-consumption percentage for the current month
                month_name = df_prices.loc[month - 1, 'Month']
                if month_name not in self_cons_percentages:
                    raise ValueError(f"No self-consumption percentage provided for month {month_name} in census_id {census_id}")
                self_cons_percent = self_cons_percentages[month_name]
                
                # Self-consumed and exported energy
                self_consumed_energy_kWh = monthly_generation * self_cons_percent
                exported_energy_kWh = monthly_generation * (1 - self_cons_percent)
                
                # Add to total kWh
                pv_self_cons_kWh += self_consumed_energy_kWh
                pv_to_grid_kWh += exported_energy_kWh
                
                # Monetary values
                pv_self_cons_euro += self_consumed_energy_kWh * buy_price
                pv_to_grid_euro += exported_energy_kWh * sell_price
            
            # Append results
            results.append({
                'build_id': row['build_id'],
                'census_id': row['census_id'],
                'PV_self_cons_kWh': pv_self_cons_kWh,
                'PV_to_grid_kWh': pv_to_grid_kWh,
                'PV_self_cons_Euro': pv_self_cons_euro,
                'PV_to_grid_Euro': pv_to_grid_euro
            })
        
        return pd.DataFrame(results)

    def calculate_present_value(self, df, investment_column, annual_cost_column, interest_rate, project_lifetime):
        """
        Calculate the present value (P) for each row in the DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame containing investment and annual cost data.
        - investment_column (str): Name of the column with investment (I).
        - annual_cost_column (str): Name of the column with annual cost (A).
        - interest_rate (float): Interest rate (e.g., 0.025 for 2.5%).
        - project_lifetime (int): Lifetime of the project (n).

        Returns:
        - pd.Series: Present value (P) asfor each row.
        """
        annuity_factor = ((1 + interest_rate) ** project_lifetime - 1) / (interest_rate * (1 + interest_rate) ** project_lifetime)
        return df[investment_column] + df[annual_cost_column] * annuity_factor
    
    def calculate_euac(self, df, r, n, present_value_col):
        crf = r * (1 + r) ** n / ((1 + r) ** n - 1)  # Capital Recovery Factor
        #if present_value_col is None and annual_savings_col is None: # for Reference case calculation
        #    return df[present_value_col] * crf
        #else:
        return df[present_value_col] * crf 
 
    #%% Calculate PV metrics with simplified calculation of degradation and price increase
    def average_price_increase(self,column, price_increase, lifespan=20):
        """
        Calculate the average annual price over the lifespan for a DataFrame column.

        Parameters:
        - column (pd.Series): Column of initial prices in Euro.
        - price_increase (float): Annual price increase rate (e.g., 0.02 for 2%).
        - lifespan (int): Lifespan of the system in years.

        Returns:
        - pd.Series: Series with average price during lifespan considering price increase.
        """ 
        if not isinstance(column, pd.Series):
            raise ValueError("Input must be a pandas Series")
        #total cumulative price over a defined lifespan, assuming the price increases by a constant percentage (price_increase) each time step
        total_price_increase = column.apply(
            lambda initial_price: sum(
                initial_price * (1 + price_increase) ** (t - 1) for t in range(1, lifespan + 1)
            )
        )
        return total_price_increase / lifespan

    def calculate_pv_anual_average(self,column, degradation_rate=0.01, lifespan=20):
        """
        Calculate the average annual PV production for a DataFrame column considering degradation.

        Parameters:
        - column (pd.Series): Column of initial annual production in kWh.
        - degradation_rate (float): Annual PV degradation rate (e.g., 0.01 for 1%).
        - lifespan (int): Lifespan of the system in years.

        Returns:
        - pd.Series: Series with average annual PV production considering degradation.
        """
        #total cumulative production over a defined lifespan, assuming the production decreases by a constant percentage (degradation_rate) each time step
        total_production = column.apply(
            lambda initial_production: sum(
                initial_production * (1 - degradation_rate) ** (t - 1) for t in range(1, lifespan + 1)
            )
        )
        return total_production / lifespan

    def calculate_pv_metrics_adv_df(self,df_pv, df_prices, monthly_self_cons_percentages, degradation_rate=0, price_increase_rate=0):
        """
        Calculate PV metrics for each building in the dataset.

        Parameters:
        - df_pv (pd.DataFrame): PV data with monthly generation columns.
        - df_prices (pd.DataFrame): Electricity buy and sell prices by month.
        - monthly_self_cons_percentages (dict): Self-consumption percentages by month.
        - incl_degradation (bool): Whether to include degradation in calculations.
        - incl_price_increase (bool): Whether to include price increase in calculations.

        Returns:
        - pd.DataFrame: DataFrame with calculated PV metrics.
        """
        monthly_columns = list(range(1, 13))  # Monthly columns
        df_results = pd.DataFrame()
        df_temp = df_pv.copy()
        
        for month in monthly_columns:
            df_temp[month] = df_temp[month]
            if degradation_rate != 0:
                # Calculate monthly average generation with degradation for the lifespan
                df_temp[month] = self.calculate_pv_anual_average(df_pv[month], degradation_rate=degradation_rate, lifespan=20)
            
            buy_price = df_prices.loc[month - 1, 'Electricity Buy']  # Match price by month
            sell_price = df_prices.loc[month - 1, 'Electricity Sell']
            if price_increase_rate != 0:
                # Calculate average monthly price considering price increase for the lifespan
                buy_price = self.average_price_increase(pd.Series([buy_price]), price_increase=price_increase_rate).iloc[0]
                sell_price = self.average_price_increase(pd.Series([sell_price]), price_increase=price_increase_rate).iloc[0]
            
            # Get the self-consumption percentage for the current month
            month_name = df_prices.loc[month - 1, 'Month']
            self_cons_percent = monthly_self_cons_percentages[month_name]
            
            # Self-consumed and exported energy
            df_temp[f"self_consumed_energy_kWh_{month}"] = df_temp[month] * self_cons_percent
            df_temp[f"exported_energy_kWh_{month}"] = df_temp[month] * (1 - self_cons_percent)
            df_temp[f'pv_self_cons_euro_{month}'] = df_temp[f"self_consumed_energy_kWh_{month}"] * buy_price
            df_temp[f'pv_to_grid_euro_{month}'] = df_temp[f"exported_energy_kWh_{month}"] * sell_price
            
        # Add to total kWh
        df_temp['PV_self_cons_kWh'] = df_temp[[f"self_consumed_energy_kWh_{month}" for month in monthly_columns]].sum(axis=1).round(3)
        df_temp['PV_to_grid_kWh'] = df_temp[[f"exported_energy_kWh_{month}" for month in monthly_columns]].sum(axis=1).round(3)
        
        # Monetary values
        df_temp['PV_self_cons_Euro'] = df_temp[[f"pv_self_cons_euro_{month}" for month in monthly_columns]].sum(axis=1).round(2)
        df_temp['PV_to_grid_Euro'] = df_temp[[f"pv_to_grid_euro_{month}" for month in monthly_columns]].sum(axis=1).round(2)
        
        df_results = df_temp[['build_id', 'census_id', 'PV_self_cons_kWh', 'PV_to_grid_kWh', 'PV_self_cons_Euro', 'PV_to_grid_Euro']]

        #del df_temp
        return df_results #pd.DataFrame(df_results)

    # Same function with faster calculation
    def calculate_pv_metrics_adv_df_faster(self,df_pv, df_prices, monthly_self_cons_percentages, degradation_rate=0, price_increase_rate=0):
        """
        Calculate PV metrics for each building in the dataset.

        Parameters:
        - df_pv (pd.DataFrame): PV data with monthly generation columns.
        - df_prices (pd.DataFrame): Electricity buy and sell prices by month.
        - monthly_self_cons_percentages (dict): Self-consumption percentages by month.
        - incl_degradation (float): Whether to include degradation in calculations.
        - incl_price_increase (float): Whether to include price increase in calculations.

        Returns:
        - pd.DataFrame: DataFrame with calculated PV metrics.
        """
        monthly_columns = list(range(1, 13))  # Monthly columns
        df_results = pd.DataFrame()
        df_temp = df_pv.copy()
        df_temp["self_consumed_energy_kWh"] = 0
        df_temp["exported_energy_kWh"] = 0
        df_temp['pv_self_cons_euro'] = 0
        df_temp['pv_to_grid_euro'] = 0
        for month in monthly_columns:
            df_temp[month] = df_temp[month]
            if degradation_rate != 0:
                # Calculate monthly average generation with degradation for the lifespan
                df_temp[month] = self.calculate_pv_anual_average(df_pv[month], degradation_rate=degradation_rate, lifespan=20)
            
            buy_price = df_prices.loc[month - 1, 'Electricity Buy']  # Match price by month
            sell_price = df_prices.loc[month - 1, 'Electricity Sell']
            if price_increase_rate != 0:
                # Calculate average monthly price considering price increase for the lifespan
                buy_price = self.average_price_increase(pd.Series([buy_price]), price_increase=price_increase_rate).iloc[0]
                sell_price = self.average_price_increase(pd.Series([sell_price]), price_increase=price_increase_rate).iloc[0]
            
            # Get the self-consumption percentage for the current month
            month_name = df_prices.loc[month - 1, 'Month']
            self_cons_percent = monthly_self_cons_percentages[month_name]
            
            # Self-consumed and exported energy
            df_temp["self_consumed_energy_kWh_m-th"] =  df_temp[month] * self_cons_percent
            df_temp["self_consumed_energy_kWh"] += df_temp["self_consumed_energy_kWh_m-th"] 
            df_temp["exported_energy_kWh_m-th"] = df_temp[month] * (1 - self_cons_percent)
            df_temp["exported_energy_kWh"] += df_temp["exported_energy_kWh_m-th"]
            df_temp['pv_self_cons_euro'] += df_temp["self_consumed_energy_kWh_m-th"] * buy_price
            df_temp['pv_to_grid_euro'] += df_temp["exported_energy_kWh_m-th"] * sell_price
                
        # Add to total kWh
        df_temp['PV_self_cons_kWh'] = df_temp["self_consumed_energy_kWh"].round(3)
        df_temp['PV_to_grid_kWh'] = df_temp["exported_energy_kWh"].round(3)
        
        # Monetary values
        df_temp['PV_self_cons_Euro'] = df_temp["pv_self_cons_euro"].round(2)
        df_temp['PV_to_grid_Euro'] = df_temp["pv_to_grid_euro"].round(2)

        df_results = df_temp[['build_id', 'census_id', 'PV_self_cons_kWh', 'PV_to_grid_kWh', 'PV_self_cons_Euro', 'PV_to_grid_Euro']]

        return df_results 
    # Function to calculate PV_self_cons_Euro and PV_to_grid_Euro
    def calculate_pv_euro_variable_self_cons(self,df_pv, df_prices, monthly_self_cons_percentages):
        # Merge monthly prices with PV data
        monthly_columns = list(range(1, 13))  # Monthly columns
        results = []

        for index, row in df_pv.iterrows():
            pv_self_cons = 0
            pv_to_grid = 0
            
            for month in monthly_columns:
                monthly_generation = row[month]
                buy_price = df_prices.loc[month - 1, 'Electricity Buy']  # Match price by month
                sell_price = df_prices.loc[month - 1, 'Electricity Sell']
                
                # Get the self-consumption percentage for the current month
                month_name = df_prices.loc[month - 1, 'Month']
                self_cons_percent = monthly_self_cons_percentages[month_name]
                
                # Self-consumed and exported energy
                self_consumed_energy = monthly_generation * self_cons_percent
                exported_energy = monthly_generation * (1 - self_cons_percent)
                
                # Monetary values
                pv_self_cons += self_consumed_energy * buy_price
                pv_to_grid += exported_energy * sell_price
            
            # Append results
            results.append({'build_id': row['build_id'], 
                            'census_id': row['census_id'], 
                            'PV_self_cons_Euro': pv_self_cons, 
                            'PV_to_grid_Euro': pv_to_grid})
        
        return pd.DataFrame(results)
    
class HeatingAnalysis:
    def __init__(self):
        pass

    def calculate_heating_demand_reduction(self,df_h_dem_reduction_coeffs, region, scenario, combination):
        """
    Calculate the heating demand reduction coefficient for a given region, scenario, and combination of components.

    Parameters:
    df_h_dem_reduction_coeffs (pd.DataFrame): DataFrame containing heating demand reduction coefficients.
    region (str): The region for which the heating demand reduction is to be calculated.
    scenario (str): The scenario under which the heating demand reduction is to be calculated.
    combination (list): A list of component names to be considered in the calculation.

    Returns:
    float: The heating demand reduction coefficient rounded to two decimal places.

    The function filters the DataFrame based on the specified region, type ('HDemReduction'), component ('envelope'),
    and the names in the combination list. It then sums the values for the specified scenario and returns the result
    rounded to two decimal places.
    """
        h_dem_reduction_coeff = df_h_dem_reduction_coeffs.loc[
            (df_h_dem_reduction_coeffs["Region"] == region) &
            (df_h_dem_reduction_coeffs["Type"] == 'HDemReduction') &
            (df_h_dem_reduction_coeffs["Component"] == 'envelope') &
            (df_h_dem_reduction_coeffs["Name"].isin(combination)),
            [scenario]
        ].sum().values[0].round(2)
        return h_dem_reduction_coeff

class EconomicAnalysisGraphs:
    def __init__(self):
        pass
    
    @staticmethod
    def plot_ec_costs(df, C_ec_b_PV='C_ec_b_PV', C_ec_b_nPV= 'C_ec_b_nPV', C_cov_b_PV='C_cov_b_PV', C_cov_b_nPV='C_cov_b_PV'):
        fig, ax = plt.subplots()
        # Plot bar plots for Variable1 and Variable2
        df[[f'{C_ec_b_PV}', f'{C_ec_b_nPV}']].plot(kind='bar', ax=ax, width=0.7)
        # Plotting top sides of bars for the first set of values
        for i, value in enumerate(df[f'{C_cov_b_PV}']):
            plt.plot([i - 0.3, i + 0.3], [value, value], color='blue')
        # Plotting top sides of bars for the second set of values
        for i, value in enumerate(df[f'{C_cov_b_nPV}']):
            plt.plot([i - 0.3, i + 0.3], [value, value], color='red')
        # Plot scatter plots for Variable3 and Variable4
        ax.scatter(df.index, df[f'{C_cov_b_PV}'], color='red', marker='_', label=f'{C_cov_b_nPV}')
        ax.scatter(df.index, df[f'{C_cov_b_nPV}'], color='blue',marker='_', label=f'{C_cov_b_PV}')
        # Set labels and title
        ax.set_xlabel('Census_ID')
        ax.set_ylabel('Costs EUR')
        ax.set_title('Comparing estimated costs')
        ax.set_xticklabels(df.index, rotation=45)
        ax.legend(loc='best',bbox_to_anchor=(1, 1)).set_visible(True)
        plt.show()

    @staticmethod
    def plot_npv_bar_chart(agg_df):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=agg_df, x='census_id', y='NPV', palette='Blues_d')
        plt.title('Total Net Present Value (NPV) per Census Section')
        plt.xlabel('Census ID')
        plt.ylabel('NPV (€)')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_euac_bar_chart(agg_df):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=agg_df, x='census_id', y='EUAC', palette='Greens_d')
        plt.title('Total Equivalent Uniform Annual Cost (EUAC) per Census Section')
        plt.xlabel('Census ID')
        plt.ylabel('EUAC (€)')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_combined_npv_euac_bar_chart(agg_df):
        melted_agg_df = agg_df.melt(id_vars='census_id', value_vars=['NPV', 'EUAC'],
                                    var_name='Metric', value_name='Value')
        plt.figure(figsize=(12, 7))
        sns.barplot(data=melted_agg_df, x='census_id', y='Value', hue='Metric')
        plt.title('NPV and EUAC per Census Section')
        plt.xlabel('Census ID')
        plt.ylabel('Value (€)')
        plt.legend(title='Metric')
        plt.tight_layout()
        plt.show()

    def plotNPV(self,df, column_mapping, filename):
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
        NPV = df[column_mapping["NPV"]] / 1e6
        inv = -df[column_mapping["CAPEX"]] / 1e6
        fix = -df[column_mapping["fix_costs"]] / 1e6
        var = -df[column_mapping["var_costs"]] / 1e6
        rev = df[column_mapping["revenue"]] / 1e6

        sns.set_theme(style='darkgrid')
        fig, ax = plt.subplots()
        ax.grid()

        # Bar plots
        ax.bar(x - width/2, inv, width, label="CAPEX", color="#bc5090")
        ax.bar(x - width/2, fix, width, bottom=inv, label="Fix Costs", color="#58508d")
        ax.bar(x - width/2, var, width, bottom=(fix + inv), label="Variable Costs", color="#ff6361")
        ax.bar(x - width/2, rev, width, bottom=0, label="Revenue", color="#20a9a6")
        ax.bar(x + width/2, NPV, width, label="NPV", color="#ffa600")

        # Labels and formatting
        ax.set_ylabel("NPV Composition [Mio €]")
        ax.set_title("NPV Composition by Scenario")
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

    

    def geo_heatmap(self, df, column_1='NRPE_kWh_per_dwelling', column_2='EUAC_per_dwelling'):
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # Coumn_1 heatmap
        df.plot(column=column_1, cmap='Reds', legend=True, ax=ax[0], edgecolor='black', linewidth=0.5)
        ax[0].set_title(f"Heatmap of `{column_1}` by Census Section")
        # Adding labels for each element
        for x, y, label in zip(df.geometry.centroid.x, df.geometry.centroid.y, df[column_1]):
            ax[0].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))

        # Column_2 heatmap
        df.plot(column=column_2, cmap='Greens', legend=True, ax=ax[1], edgecolor='black', linewidth=0.5)
        ax[1].set_title(f"Heatmap of `{column_2}` by Census Section")
        # remove labes of y and x axis
        ax[0].set_yticklabels([])
        ax[0].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[1].set_xticklabels([])
        """# Add label for colorbar
        legend = ax[0].get_legend()
        if legend:
            legend.set_title(column_1)"""
        # Adding labels for each element
        for x, y, label in zip(df.geometry.centroid.x, df.geometry.centroid.y, df[column_2]):
            ax[1].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black',  bbox=dict(facecolor='white', alpha=0.5))#, backgroundcolor='white')

        plt.subplots_adjust(wspace=0.008)  # Adjusts the width spacing between subplots
        plt.show()

    def plot_savings_distribution(self, df, values = "Savings, Dwell with PV, %",colormap="blue"):
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
                ax.set_title(f"EC price sell: {int(price_diff*100)}% less grid feed-in, EC price buy: {int(price_diff*100)}% less grid electricity", fontsize=14)
                ax.set_xlabel("Dwellings Share with PV (%)" , fontsize=14)
                ax.set_ylabel("Savings (%)", fontsize=14)
                ax.grid(True)
                ax.legend(title="Census ID", loc="best")
                ax.set_xticklabels(labels=[int(s*100) for s in list(df_subset_pivot.index)], rotation=0, fontsize=14)
    # Adjust layout for better spacing
        plt.tight_layout()
    # Show the plot
        plt.show()
#%%

# %%
