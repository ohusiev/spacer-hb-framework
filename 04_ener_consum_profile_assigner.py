# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:43:52 2024

@author: oleksandr.husiev
"""
#%%
import pandas as pd
import os
import random
import numpy as np
#%%
#PATH = r"C:\Users\oleksandr.husiev\My Drive\01_PHD_CODE\chapter_5\LoadProGen\Work\Bilbao"#"C:\Users\oleksandr.husiev\Desktop\Work"
#PATH = r"D:\oleksandr.husiev@deusto.es\01_PHD_CODE\chapter_5\LoadProGen\Work\Bilbao"
PATH = r"H:\My Drive\01_PHD_CODE\chapter_5\LoadProGen\Work\Bilbao"

files = [file for file in os.listdir(PATH) if file.endswith('.csv')]
file = os.path.join(PATH, files[0])

df = pd.read_csv(file, sep=";")
#%% Sample DataFrame
data = pd.DataFrame({
    'census_id': [1, 2, 3],
    'Total number of dwellings': [100, 150, 200],
    'Total number of population': [200, 300, 400],
    'age_group_1': [40, 60, 80], # age_group_1: 0-19 years
    'age_group_2': [120, 180, 240], # age_group_2: 20-24 years
    'age_group_3': [60, 90, 120], # age_group_3: 25-64 years
    'age_group_4': [20, 30, 40], # age_group_4: 65+ years
    'Percentage of people employed': [60, 65, 70], #
    "Single Occupied Dwellings per municipality": [0.10, 0.265, 0.270], # "% Single Occupied Dwellings per municipality"
    "Two People Occupied Dwellings per municipality": [0.10, 0.265, 0.270], # "% Two People Occupied Dwellings per municipality" 
    "Three to Five People Occupied Dwellings per municipality": [0.10, 0.265, 0.270], # "% Three to Five People Occupied Dwellings per municipality"
    "Six to Nine People Occupied Dwellings per municipality": [0.10, 0.265, 0.270], # "% Six to Nine People Occupied Dwellings per municipality"
    "Ten or more People Occupied Dwellings per municipality": [0.10, 0.265, 0.270],# "% Ten or more People Occupied Dwellings per municipality"
    "unemployment_rate": [0.05,0.06,0.07] # "% Percentage of people unemployed"
})

#%% Real DataFrame
file_dir=os.getcwd()
data = pd.read_excel(os.path.join(file_dir,"data/04_energy_consumption_profiles/00_data_census_id_ener_consum_profiling.xlsx"), sheet_name="04_dwelling_profiles_census", index_col=0) # `census_id` - statistical cencus id is an index
data = data.iloc[:,:12]
#%% Define the fixed percentages for the different types of dwellings

# GIPUZKOA

# Define the percentage for Single dwellings
percentage_of_people_20_24_live_alone = 0.5 # "% of people between 20-24 that live alone per region" https://bit.ly/4bCh4Eh
percentage_of_people_25_65_live_alone = 0.25 # "% of people between 25-65 that live alone per region" https://bit.ly/4bCh4Eh)
percentage_of_people_65_live_alone = 0.33 # "% of people >65 that live alone per region" https://bit.ly/4bCh4Eh)

# Define the percentage for Two people dwellings
couples_25_65_without_kids = 0.11 # "% of couples between 25-65 without kids per region" https://bit.ly/3VLVMxI
couples_65_without_kids = 0.33 # "% of couples >65 without kids per region" https://bit.ly/3VLVMxI
monoparental_25_65 = 0.1 # "% of monoparental between 25-65 per region" https://bit.ly/3VLVMxI
# difference of "father with kids" and "mother with kids" for Euskadi is roughly average 1:5.5 https://bit.ly/4eGsPMw

# Define the percentage for Three to Five people dwellings
couples_25_65_with_kids = 0.47 # "% of couples between 25-65 with kids per region" https://bit.ly/3VLVMxI
coeff_1_children = 0.46 # 15.4/33.5 https://www.ine.es/prensa/ech_2020.pdf
coeff_2_children = 0.44 # 14.7 / 33.5  https://www.ine.es/prensa/ech_2020.pdf
coeff_3_more_children = 0.1 # 3/33.5 https://www.ine.es/prensa/ech_2020.pdf

# Unemployement range used as a coefficient to represent a possible weigth of families with employed and unemployed people

data["unemployment_rate"]=1 # REmove if needed a coefficient
data["employment_rate"] = 1 #1-data["unemployment_rate"]

coeff_unemployement = 1 #0.2 # parado/total # table Población de 16 y más años por sexo y relación con la actividad (agrupado); 1-inactive
coeff_employement= 1#(1-coeff_unemployement) # parado/total # table Población de 16 y más años por sexo y relación con la actividad (agrupado); 1-inactive

# Define the percentage for Six to Nine people dwellings
rest_16_25 = 1-percentage_of_people_20_24_live_alone # asusming that kids under 16 can't live with parents or any tutelage
rest_25_65= 1-percentage_of_people_25_65_live_alone - couples_25_65_without_kids - couples_25_65_with_kids - monoparental_25_65
rest_65 = 1-percentage_of_people_65_live_alone - couples_65_without_kids

#%%
def handle_column_multiplication(df, col1, col2, result_col):
    """
    Multiplies two columns, handles NaN and inf values by replacing them with 0, rounds, and converts to integers.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    col1 (str): The name of the first column.
    col2 (str): The name of the second column.
    result_col (str): The name of the result column to store the output.
    
    Returns:
    pd.DataFrame: The dataframe with the new calculated column.
    """
    # Perform the multiplication, handle NaN, inf, and rounding
    df[result_col] = (df[col1] * df[col2]).replace([np.inf, -np.inf], np.nan).fillna(0).round(0).astype(int)
    return df

def calculate_dwellings(df):
    """
    Calculate various dwelling types by multiplying 'Total number of dwellings' with other dwelling proportions.
    
    Parameters:
    df (pd.DataFrame): The input dataframe
    
    Returns:
    pd.DataFrame: The dataframe with newly calculated columns.
    """
    # List of dwelling types with their corresponding columns
    dwelling_types = [
        ('ND1 (Single Occupied Dwellings)', 'Single Occupied Dwellings per municipality'),
        ('ND2 (Two People Occupied Dwellings)', 'Two People Occupied Dwellings per municipality'),
        ('ND3_5 (Three to Five People Occupied Dwellings)', 'Three to Five People Occupied Dwellings per municipality'),
        ('ND6_9 (Six to Nine People Occupied Dwellings)', 'Six to Nine People Occupied Dwellings per municipality'),
        ('ND10 (Ten or more People Occupied Dwellings)', 'Ten or more People Occupied Dwellings per municipality')
    ]
    
    # Loop through the dwelling types and apply the multiplication
    for result_col, proportion_col in dwelling_types:
        df = handle_column_multiplication(df, 'Total number of dwellings', proportion_col, result_col)
    
    return df

# Apply the function to your dataframe
data = calculate_dwellings(data)
print(data)
data= data[data["Total number of dwellings"].notna()]
#%%
def calculate_single_dwellings_types(df_input):
    # Calculate intermediate values
    df=df_input.copy()
    #df['People 0-16'] = df['Kids under 16 (male)'] + df['Kids under 16 (female)']
    #df['People 16-25'] = df['People 0-25'] - df['People 0-16']
    df['Stu_Work'] =df['age_group_2']* percentage_of_people_20_24_live_alone #df['percentage_of_people_0_25_live_alone']

    #df['People 25-65'] = df['People 25-65 (male)'] + df['People 25-65 (female)']
    df['1P_Work'] =df['age_group_3'] * percentage_of_people_25_65_live_alone#df['percentage_of_people_25_65_live_alone']

    #df['People 65'] = df['Senior people (male)'] + df['Senior people (female)']
    df['1P_Ret'] =df['age_group_4'] * percentage_of_people_65_live_alone #df['percentage_of_people_65_live_alone']

    # Calculate percentages
    df['% 1P_Work'] = df['1P_Work'] / df['age_group_3']
    df['% Stu_Work'] = df['Stu_Work'] / df['age_group_2']
    df['% 1P_Ret'] = df['1P_Ret'] / df['age_group_4']

    # Calculate Sum of all ND1
    df['Sum of all ND1'] = df['1P_Work'] + df['Stu_Work'] + df['1P_Ret']

    # Transforming into "weight" coefficients and rounding
    df['1P_Work'] = (df['ND1 (Single Occupied Dwellings)'] * (df['1P_Work'] / df['Sum of all ND1'])).astype(int)
    df['Stu_Work'] = (df['ND1 (Single Occupied Dwellings)'] * (df['Stu_Work'] / df['Sum of all ND1'])).astype(int)
    df['1P_Ret'] = (df['ND1 (Single Occupied Dwellings)'] * (df['1P_Ret'] / df['Sum of all ND1'])).astype(int)

    return pd.concat([df_input, df[['1P_Work', 'Stu_Work', '1P_Ret']]], axis=1)

data = calculate_single_dwellings_types(data)
#%%
# Function to calculate the number of dwellings of two people within a municipality
def calculate_two_people_dwellings_types(df_input):
    df=df_input.copy()
    # Calculate intermediate values
    df['Couple_Work'] = df['age_group_3']  * couples_25_65_without_kids * data["unemployment_rate"]
    # df['Couple, 1 at work, 1 work from home'] = df['People 25-65']  * couples_25_65_without_kids * (1-coeff_employement)
    df['Couple_65+'] = df['age_group_4'] * couples_65_without_kids
    df['1P_1Ch_Work'] = df['age_group_3'] * monoparental_25_65 +df['age_group_1'] * monoparental_25_65 # 'People 0-16' is included to account for original population data, assuming more kids indicate a higher percentage of dwellings with families and kids.

    # Calculate Sum of all ND2
    df['Sum of all ND2'] = df['Couple_Work'] + df['Couple_65+'] + df['1P_1Ch_Work']

    # Transforming into "weight" coefficients and rounding
    df['Couple_Work'] = (df['ND2 (Two People Occupied Dwellings)'] * (df['Couple_Work'] / df['Sum of all ND2'])).astype(int)
    df['Couple_65+'] = (df['ND2 (Two People Occupied Dwellings)'] * (df['Couple_65+'] / df['Sum of all ND2'])).astype(int)
    df['1P_1Ch_Work'] = (df['ND2 (Two People Occupied Dwellings)'] * (df['1P_1Ch_Work'] / df['Sum of all ND2'])).astype(int)

    return pd.concat([df_input, df[['Couple_Work','Couple_65+','1P_1Ch_Work']]], axis=1)

data = calculate_two_people_dwellings_types(data)
#%%
def calculate_three_five_people_dwellings_types(df_input):
    df= df_input.copy()
    # Calculate the family work columns
    df['Fam_1Ch_Work'] = df['age_group_3'] * couples_25_65_with_kids * data["employment_rate"] + df['age_group_1'] * coeff_1_children
    df['Fam_1Ch_1Wrk1Hm'] = df['age_group_3'] * couples_25_65_with_kids  * data["unemployment_rate"] + df['age_group_1'] * coeff_1_children
    df['Fam_2Ch_Work'] = df['age_group_3'] * couples_25_65_with_kids* data["unemployment_rate"] + df['age_group_1'] * coeff_2_children
    df['Fam_3Ch_Work'] = df['age_group_3']  * couples_25_65_with_kids* data["employment_rate"] + df['age_group_1'] * coeff_3_more_children
    df['Fam_3Ch_1Wrk1Hm'] = df['age_group_3'] * couples_25_65_with_kids * data["employment_rate"] + df['age_group_1'] * coeff_3_more_children
    df['Stu_Share'] = df['age_group_2'] * rest_16_25*0.5 # assuming that 50% of students share a flat, while other might live with their parents
    df['Fam_3Ch_HmWrk'] = df['age_group_3'] * couples_25_65_with_kids * data["unemployment_rate"] + df['age_group_1'] * coeff_3_more_children

    df['Sum of all ND3_5'] = df['Fam_1Ch_Work'] + df['Fam_1Ch_1Wrk1Hm'] + df['Fam_2Ch_Work'] + df['Fam_3Ch_Work'] + df['Fam_3Ch_1Wrk1Hm']+df['Stu_Share']+ df['Fam_3Ch_HmWrk']

    # Transforming into "weight" coefficients and rounding
    df['Fam_1Ch_Work'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)']*(df['Fam_1Ch_Work'] / df['Sum of all ND3_5'])).astype(int))
    df['Fam_1Ch_1Wrk1Hm'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)']*(df['Fam_1Ch_1Wrk1Hm'] / df['Sum of all ND3_5'])).astype(int))
    df['Fam_2Ch_Work'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)']*(df['Fam_2Ch_Work'] / df['Sum of all ND3_5'])).astype(int))
    df['Fam_3Ch_Work'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)']*(df['Fam_3Ch_Work'] / df['Sum of all ND3_5'])).astype(int))
    df['Fam_3Ch_1Wrk1Hm'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)']*(df['Fam_3Ch_1Wrk1Hm'] / df['Sum of all ND3_5'])).astype(int))
    df['Stu_Share'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)']*(df['Stu_Share'] / df['Sum of all ND3_5'])).astype(int))
    df['Fam_3Ch_HmWrk'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)']*(df['Fam_3Ch_HmWrk'] / df['Sum of all ND3_5'])).astype(int))
    return pd.concat([df_input, df[['Fam_1Ch_Work','Fam_1Ch_1Wrk1Hm','Fam_2Ch_Work',
                                        'Fam_3Ch_Work','Fam_3Ch_1Wrk1Hm','Stu_Share','Fam_3Ch_HmWrk']]], axis=1)

data = calculate_three_five_people_dwellings_types(data)
#%%
list_younger_people = ["6-9P_Occup_id_1", "6-9P_Occup_id_3"] # two random profiles with dominant Wokring people, and Working Students, with 6,8,9 people in the house
list_older_people = ["6-9P_Occup_id_3", "6-9P_Occup_id_3"] # two random profiles with dominant Seniour people, and Working people in the house

from collections import Counter
def select_random_elements(list, count):
    return random.choices(list, k=count)

def calculate_six_and_more_people_dwellings_types(df_input):
    df = df_input.copy()
    df['6-9P_Occup_id_1'] = 0
    #df['6-9P_Occup_id_2'] = 0
    df['6-9P_Occup_id_3'] = 0
    #df['6-9P_Occup_id_4'] = 0

    for index, row in df[df['age_group_3'] > df['age_group_4']].iterrows(): # if there are more people between 25-65 than over 65
        #selected_elements = select_random_elements(list_younger_people, int(row['ND6_9 (Six to Nine People Occupied Dwellings)']))
        counter = int(row['ND6_9 (Six to Nine People Occupied Dwellings)'])#Counter(selected_elements)
        df.at[index, '6-9P_Occup_id_1'] = counter#.get('6-9P_Occup_id_1', 0) # get the value of the key, if not found return 0
        #df.at[index, '6-9P_Occup_id_2'] = counter.get('6-9P_Occup_id_2', 0)
    for index, row in df[df['age_group_3'] <= df['age_group_4']].iterrows(): # if there are more people over 65 than between 25-65
        #selected_elements = select_random_elements(list_older_people, int(row['ND6_9 (Six to Nine People Occupied Dwellings)']))
        counter = int(row['ND6_9 (Six to Nine People Occupied Dwellings)'])# Counter(selected_elements)
        df.at[index, '6-9P_Occup_id_3'] = counter#.get('6-9P_Occup_id_3', 0)
        #df.at[index, '6-9P_Occup_id_4'] = counter.get('6-9P_Occup_id_4', 0)
    # Drop the comparison column
    #df.drop(columns=['comparison'], inplace=True)
    return pd.concat([df_input, df[['6-9P_Occup_id_1', '6-9P_Occup_id_3']]], axis=1)

data = calculate_six_and_more_people_dwellings_types(data)
# %%
def calculate_ten_and_more_people_dwellings_types(df_input):
    df = df_input.copy()
    df['10+P_Occup_id_1'] = 0 # id_old_people
    df['10+P_Occup_id_2'] = 0 #id_mix_students_work

    # Condition 1: More people between 25-65 than over 65
    for index, row in df[(df['age_group_3'] >= df['age_group_4']) & (df['age_group_2'] >= df['age_group_4'])].iterrows(): 
        selected_elements = select_random_elements(list_younger_people, int(row['ND10 (Ten or more People Occupied Dwellings)']))
        counter = Counter(selected_elements)
        df.at[index, '10+P_Occup_id_1'] = counter.get('10+P_Occup_id_1', 0)  # Corrected the key name
        
    # Condition 2: More people over 65 than between 25-65
    for index, row in df[(df['age_group_3'] < df['age_group_4']) & (df['age_group_2'] >= df['age_group_4'])].iterrows():
        selected_elements = select_random_elements(list_older_people, int(row['ND10 (Ten or more People Occupied Dwellings)']))
        counter = Counter(selected_elements)
        df.at[index, '10+P_Occup_id_2'] = counter.get('10+P_Occup_id_2', 0)
    
    return pd.concat([df_input, df[['10+P_Occup_id_1', '10+P_Occup_id_2']]], axis=1)

data = calculate_ten_and_more_people_dwellings_types(data).round(4)
#%%
# Save to csv
data.to_csv("data/04_energy_consumption_profiles/04_1_energy_consumption_profiles_real_data.csv", index="census_id")

#%% OLD non "optimised" function
# Define a function to calculate the number of dwellings of different sizes within a municipality
def calculate_dwellings(df):
    df['ND1 (Single Occupied Dwellings)'] = (df['Total number of dwellings'] * df['Single Occupied Dwellings per municipality']).astype(int)
    df['ND2 (Two People Occupied Dwellings)'] = (df['Total number of dwellings'] * df['Two People Occupied Dwellings per municipality']).round(0)
    df['ND3_5 (Three to Five People Occupied Dwellings)'] = (df['Total number of dwellings'] * df['Three to Five People Occupied Dwellings per municipality']).round(0)
    df['ND6_9 (Six to Nine People Occupied Dwellings)'] = (df['Total number of dwellings'] * df['Six to Nine People Occupied Dwellings per municipality']).round(0)
    df['ND10 (Ten or more People Occupied Dwellings)'] = (df['Total number of dwellings'] * df['Ten or more People Occupied Dwellings per municipality']).round(0)
    return df

data = calculate_dwellings(data)
print(data)
# %% OLD version that was corrected with Logical operator (&)
def calculate_ten_and_more_people_dwellings_types(df_input):
    df = df_input.copy()
    df['id_old_people'] = 0
    df['id_mix_students_work'] = 0

    for index, row in df[df['age_group_3'] >= df['age_group_4'] | df['age_group_2'] >= df['age_group_4']].iterrows(): # if there are more people between 25-65 than over 65
        selected_elements = select_random_elements(list_younger_people, int(row['ND10 (Ten or more People Occupied Dwellings)']))
        counter = Counter(selected_elements)
        df.at[index, 'id_old_people'] = counter.get('id_old_people', 0) # get the value of the key, if not found return 0
        
    for index, row in df[df['age_group_3'] < df['age_group_4'] | df['age_group_2'] >= df['age_group_4']].iterrows(): # if there are more people over 65 than between 25-65
        selected_elements = select_random_elements(list_older_people, int(row['ND10 (Ten or more People Occupied Dwellings)']))
        counter = Counter(selected_elements)
        df.at[index, 'id_mix_students_work'] = counter.get('id_mix_students_work', 0)
    return pd.concat([df_input, df[['id_old_people', 'id_mix_students_work']]], axis=1)

data = calculate_ten_and_more_people_dwellings_types(data)

