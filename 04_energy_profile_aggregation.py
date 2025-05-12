#%%
import pandas as pd
import os
import matplotlib.pyplot as plt

#%%
#PATH = r"C:\Users\oleksandr.husiev\My Drive\01_PHD_CODE\chapter_5\LoadProGen\Work\Bilbao"#"C:\Users\oleksandr.husiev\Desktop\Work"
PATH = r"D:\oleksandr.husiev@deusto.es\01_PHD_CODE\chapter_5\LoadProGen\Work\Bilbao"
#PATH = r"H:\My Drive\01_PHD_CODE\chapter_5\LoadProGen\Work\Bilbao"

files = [file for file in os.listdir(PATH) if file.endswith('.csv')]
file = os.path.join(PATH, files[0])
df_dwellings_data = pd.read_csv("data/04_energy_consumption_profiles/04_1_energy_consumption_profiles_real_data.csv", index_col=0)
# 
#%%
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
#%% """FILTER FOR THE PROFILES"""
# We assume a that from our dwellings per census_id only 50% will get a direct PV system for consumption,
# while the other 50% will not have a PV system

pv_pct = 0.25#1#0.25
no_pv_pct = 1 - pv_pct

data_no_pv=(df_dwellings_data[list(profile_mapping.keys())]*no_pv_pct).round(0)
data = (df_dwellings_data[list(profile_mapping.keys())]*pv_pct).round(0)

#%%
# Create a matrix of all the electric energy profiles
def matrix_of_profiles(PATH, files):
    df_matrix = None
    for i, file_name in enumerate(files):
        if df_matrix is None:
            df_matrix = pd.read_csv(f"{PATH}/{file_name}", sep=';', usecols=["Time", "Sum [kWh]"], parse_dates=['Time'])
            file_name = file_name.strip(".csv")
            df_matrix = df_matrix.rename(columns={"Sum [kWh]": file_name})
        else:
            df_temp = pd.read_csv(f"{PATH}/{file_name}", sep=';', usecols=["Time", "Sum [kWh]"], parse_dates=['Time'])
            file_name = file_name.strip(".csv")
            df_temp = df_temp.rename(columns={"Sum [kWh]": file_name})
            df_matrix = pd.merge(df_matrix, df_temp[['Time', file_name]], on='Time', how='left')
    
    # Format the 'Time' column # works on personal computer but not on work computer
    #df_matrix['Time'] = pd.to_datetime(df_matrix['Time'])
    #df_matrix['Time'] = df_matrix['Time'].dt.strftime('%d/%m/%Y %H:%M')
    
    return df_matrix
#%% Buikd the matrix of profiles
df_matrix = matrix_of_profiles(PATH, files)

#%% plotting the profiles
df_matrix.plot(kind='line', figsize=(10, 6), marker='o')
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Sensor Reading')
plt.title('Sensor Readings Over Time')
plt.legend(title='Sensors')

#%%
# Iterate over each row in df1
def aggregate_profiles(data, df_matrix, profile_mapping):
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
        result_df[row_id] = row_sum #.sum()  # sum() is used to sum across the series
    
    return result_df
#%%
result_pv_df = aggregate_profiles(data, df_matrix, profile_mapping)
result_no_pv_df = aggregate_profiles(data_no_pv, df_matrix, profile_mapping)
# Show the result
result_pv_df
#%%
result_columns_sum = result_pv_df.sum(axis=0)
#make a directory to save the results if it does not exist
if not os.path.exists(f"data/04_energy_consumption_profiles/dwell_share_{pv_pct}"):
    os.makedirs(f"data/04_energy_consumption_profiles/dwell_share_{pv_pct}")

result_pv_df.round(4).to_csv(f"data/04_energy_consumption_profiles/dwell_share_{pv_pct}/04_2_aggregated_1h_profiles_with_pv_dwell_share_{pv_pct}.csv", index=False)
#result_columns_sum.to_csv("data/04_energy_consumption_profiles/04_2_aggregated_1h_profiles_sum.csv")

#%%
result_no_pv_df.round(4).to_csv(f"data/04_energy_consumption_profiles/dwell_share_{pv_pct}/04_2_aggregated_1h_profiles_no_pv_dwell_share_{no_pv_pct}.csv", index=False)

#%% SIMPLE EXERCISE TO DESIGNED THE FUNCTION
""" 
# Sample DataFrames
df1 = pd.DataFrame({
    'A': [1, 0, np.nan],
    'B': [0, 2, 3],
    'C': [4, 0, 5]
}, index=[100, 101, 102])

df2 = pd.DataFrame({
    'A': [10, 20, 30],
    'B': [40, 50, 60],
    'C': [70, 80, 90]
}, index=[0, 1, 2])

# Initialize an empty DataFrame for the results with same index as df1 and one column for sums
df3_resuts = pd.DataFrame(index=df2.index, columns=df1.index)

# Iterate over each row in df1
for row_id in df1.index:
    # Get the row from df1 and filter for non-zero, non-NaN values
    row_values = df1.loc[row_id]
    non_zero_columns = row_values[(row_values != 0) & row_values.notna()].index
    print(non_zero_columns)
    # Initialize the sum for this row
    row_sum = pd.Series(0, index=df2.index)
    
    # Multiply corresponding columns in df2 by the values in df1 and sum them
    for column in non_zero_columns:
        value = df1.loc[row_id, column]
        row_sum += df2[column] * value
    print(row_sum)
    # Store the summed result in df3
    df3_resuts[row_id] = row_sum #.sum()  # sum() is used to sum across the series

# Show the result
df3_resuts
"""
# %%
