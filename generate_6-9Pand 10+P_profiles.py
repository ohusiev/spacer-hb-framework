#%%
import pandas as pd
import os

profile_mapping = {
    "Stu_Share": "CHR52 Student Flatsharing",
    "6-9P_Occup_id_1": "id_1", #CHR15 Multigenerational Home: working couple, 2 children, 2 seniors 
    "Stu_Work": "CHR13 Student with Work",
    #"6-9P_Occup_id_3": "id_1",
    #"6-9P_Occup_id_4": "id_1",
    #"10+P_Occup": "ND10 (Ten or more People Occupied Dwellings)",
    #"10+P_Occup_id_1": "id_1",
    #"10+P_Occup_id_2": "id_1"
}
#PATH = r"D:\oleksandr.husiev@deusto.es\01_PHD_CODE\chapter_5\LoadProGen\Work\Bilbao"
PATH = r"C:\Users\oleksandr.husiev\My Drive\01_PHD_CODE\chapter_5\LoadProGen\Work\Bilbao"
files = [file for file in os.listdir(PATH) if file.endswith('.csv')]
file = os.path.join(PATH, files[0])
#%%
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
    df_matrix['Time'] = df_matrix['Time'].dt.strftime('%d/%m/%Y %H:%M')
    
    return df_matrix

df_matrix = matrix_of_profiles(PATH, files)
# %%
df_matrix_select = df_matrix[list(profile_mapping.keys())+["Time"]]
# %%
#df_6_9P_Occup_id_3 = (df_matrix_select["Stu_Share"] + df_matrix_select["Stu_Share"] * 1.5)
#df_10_P_Occup_id_1 = df_matrix_select["10+9P_Occup_id_1"] * 2
#df_10_P_Occup_id_3=df_matrix_select["10+9P_Occup_id_3"]+df_matrix_select["Stu_Share"]*1.5 +df_matrix_select["Time"]
df_6_9P_Occup_id_3 = pd.DataFrame({
    "Time": df_matrix_select["Time"],
    "Sum [kWh]": df_matrix_select["Stu_Share"] + df_matrix_select["Stu_Share"] * 1.5,
})

#Uncomment and modify the following lines if needed
df_10_P_Occup_id_1 = pd.DataFrame({
     "Time": df_matrix_select["Time"],
     "Sum [kWh]": df_matrix_select["6-9P_Occup_id_1"] * 1.5
})

df_10_P_Occup_id_3 = pd.DataFrame({
     "Time": df_matrix_select["Time"],
     "Sum [kWh]": df_matrix_select["6-9P_Occup_id_1"] + df_matrix_select["Stu_Share"] * 1.5
})
#%%
# Save the profiles to a csv file
df_6_9P_Occup_id_3.to_csv(f"{PATH}/6-9P_Occup_id_3.csv", index=False, sep=';')
df_10_P_Occup_id_1.to_csv(f"{PATH}/10+P_Occup_id_1.csv", index=False, sep=';')
df_10_P_Occup_id_3.to_csv(f"{PATH}/10+P_Occup_id_2.csv", index=False, sep=';')
# %%
