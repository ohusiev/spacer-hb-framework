#%%
import pandas as pd
import os
import matplotlib.pyplot as plt

class EnergyProfileAggregator:
    def __init__(self, path, profile_mapping, pv_pct=0.25):
        self.PATH = path
        self.profile_mapping = profile_mapping
        self.pv_pct = pv_pct
        self.no_pv_pct = 1 - pv_pct
        self.files = [file for file in os.listdir(self.PATH) if file.endswith('.csv')]
        self.df_dwellings_data = pd.read_csv(
            "data/04_energy_consumption_profiles/04_1_energy_consumption_profiles_real_data.csv", index_col=0
        )
        self.data_no_pv = (self.df_dwellings_data[list(self.profile_mapping.keys())] * self.no_pv_pct).round(0)
        self.data = (self.df_dwellings_data[list(self.profile_mapping.keys())] * self.pv_pct).round(0)
        self.df_matrix = self.matrix_of_profiles()

    # Create a matrix of all the electric energy profiles
    def matrix_of_profiles(self):
        df_matrix = None
        for i, file_name in enumerate(self.files):
            if df_matrix is None:
                df_matrix = pd.read_csv(
                    f"{self.PATH}/{file_name}", sep=';', usecols=["Time", "Sum [kWh]"], parse_dates=['Time']
                )
                file_name_clean = file_name.strip(".csv")
                df_matrix = df_matrix.rename(columns={"Sum [kWh]": file_name_clean})
            else:
                df_temp = pd.read_csv(
                    f"{self.PATH}/{file_name}", sep=';', usecols=["Time", "Sum [kWh]"], parse_dates=['Time']
                )
                file_name_clean = file_name.strip(".csv")
                df_temp = df_temp.rename(columns={"Sum [kWh]": file_name_clean})
                df_matrix = pd.merge(df_matrix, df_temp[['Time', file_name_clean]], on='Time', how='left')
        return df_matrix

    # Plotting the profiles
    def plot_profiles(self):
        self.df_matrix.plot(kind='line', figsize=(10, 6), marker='o')
        plt.xlabel('Time')
        plt.ylabel('Sensor Reading')
        plt.title('Sensor Readings Over Time')
        plt.legend(title='Sensors')
        plt.show()

    # Iterate over each row in df1
    def aggregate_profiles(self, data):
        df_matrix = self.df_matrix
        profile_mapping = self.profile_mapping
        result_df = pd.DataFrame(index=df_matrix.index, columns=data.index)
        result_df['Time'] = df_matrix['Time']

        for row_id in data.index:
            # Get the row from df1 and filter for non-zero, non-NaN values
            row_values = data[list(profile_mapping.keys())].loc[row_id]
            non_zero_columns = row_values[(row_values != 0) & row_values.notna()].index

            # Initialize the sum for this row
            row_sum = pd.Series(0, index=df_matrix.index)

            # Multiply corresponding columns in df2 by the values in df1 and sum them
            for column in non_zero_columns:
                value = data.loc[row_id, column]
                row_sum += df_matrix[column] * value

            # Store the summed result in df3
            result_df[row_id] = row_sum

        return result_df

    def save_results(self):
        result_pv_df = self.aggregate_profiles(self.data)
        result_no_pv_df = self.aggregate_profiles(self.data_no_pv)

        #make a directory to save the results if it does not exist
        save_dir = f"data/04_energy_consumption_profiles/dwell_share_{self.pv_pct}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        result_pv_df.round(4).to_csv(
            f"{save_dir}/04_2_aggregated_1h_profiles_with_pv_dwell_share_{self.pv_pct}.csv", index=False
        )
        result_no_pv_df.round(4).to_csv(
            f"{save_dir}/04_2_aggregated_1h_profiles_no_pv_dwell_share_{self.no_pv_pct}.csv", index=False
        )
        return result_pv_df, result_no_pv_df

# Example usage:
profile_mapping = {
    #"1P_Occup": "ND1 Single Occupied Dwellings",
    "1P_Work": "CHR07 Single with work",
    "Stu_Work": "CHR13 Student with Work",
    "1P_Ret": "CHR30 Single, Retired Man/Woman", #redo profile, it has 1 min time step - DONE
    #"2P_Occup": "ND2 Two People Occupied Dwellings",
    "Couple_Work": "CHR01 Couple both at Work",
    "Couple_65+": "CHR16 Couple over 65 years",
    "1P_1Ch_Work": "CHR22 Single woman, 1 child, with work",
    #"3-5P_Occup": "ND3-5 Three to Five People Occupied Dwellings",
    "Fam_1Ch_Work": "CHR03 Family, 1 child, both at work",
    "Stu_Share": "CHR52 Student Flatsharing",
    "Fam_2Ch_Work": "CHR27 Family both at work, 2 children",
    "Fam_3Ch_Work": "CHR41 Family with 3 children, both at work",
    "Fam_1Ch_1Wrk1Hm": "CHR45 Family with 1 child, 1 at work, 1 at home",
    "Fam_3Ch_1Wrk1Hm": "CHR20 Family one at work, one work home, 3 children",
    "Fam_3Ch_HmWrk": "CHR59 Family, 3 children, parents without work/work at home",
    #"6-9P_Occup": "ND6-9 Six to Nine People Occupied Dwellings",
    "6-9P_Occup_id_1": "id_1", #CHR15 Multigenerational Home: working couple, 2 children, 2 seniors 
    #"6-9P_Occup_id_2": "id_1",
    "6-9P_Occup_id_3": "id_1",
    #"6-9P_Occup_id_4": "id_1",
    #"10+P_Occup": "ND10 (Ten or more People Occupied Dwellings)",
    "10+P_Occup_id_1": "id_1",
    "10+P_Occup_id_2": "id_1"
}
#%%
if __name__ == "__main__":
    # Define the path to the directory containing the CSV files
    PATH = r"C:\\Users\\Oleksandr-MSI\\Documentos\\GitHub\\spacer-hb-framework\\LoadProGen\\Bilbao"
    # Instantiate and use the class
    #pv_pct_list = [0.25, 0.5, 0.75]
    #for pv_pct in pv_pct_list:
    aggregator = EnergyProfileAggregator(PATH, profile_mapping, pv_pct=0.25)
    #aggregator.plot_profiles()
    result_pv_df, result_no_pv_df = aggregator.save_results()

# %%
