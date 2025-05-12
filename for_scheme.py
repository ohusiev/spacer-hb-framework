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
#%% FIRST EXPLANATION OF LOGIC FOR CONSUMPTION PROFILES
"""
COUNSUMPTION PROFILES RULES

# ND1 (Single Occupied Dwellings)="Total number of dwellings" * "% Single Occupied Dwellings" per municipality https://bit.ly/3xCooBO
'Total number of population' https://bit.ly/3XMnuxj 
"Total number of dwellings" per census section https://bit.ly/3L00naT
    "CHR07 Single with work.General.Electricity.Bilbao"
       Single with work = ('People 25-65' * percentage_of_people_25_65_live_alone)
       % Single with work = Single with work /'People 25-65'
    "CHR13 Student with Work.Electricity.Bilbao"
       Student with Work = ('People 0-25' * percentage_of_people_0_25_live_alone) 
       % Student with Work = Student with Work/ 'People 0-25'
    "CHR30 Single, Retired Man.Electricity.Bilbao"
       Single, Retired Man = ('People >65' * percentage_of_people_65_live_alone) 
       % Single, Retired Man = Single, Retired Man / 'People >65'
    Sum of all ND1 =  Single with work + Student with Work + Single, Ret # transforming into a "weight" coeffs

    Single with work = round(ND1 * (Single with work/Sum of all ND1),0)
    Student with Work = round(ND1 * (Student with Work/Sum of all ND1),0)
    Single, Retired Man = round(ND1 * (Single, Retired Man/Sum of all ND1),0)

%by civil status per municipality https://bit.ly/3VGeNBU
    
# ND2 (Two People Occupied Dwellings)="Total number of dwellings" * "% Two People Occupied Dwellings" per municipality https://bit.ly/3xCooBO
coeff_Age_couples_= https://bit.ly/3XMnuxj

    "CHR01 Couple both at Work.General.Electricity.Bilbao"
        Couple both at Work = ('People 25-65' -  Single with work) * % couples_25_65_without_kids per region https://bit.ly/4eJNCyD
    "CHR16 Couple over 65 years.General.Electricity.Bilbao"
        Couple over 65 years = ('People >65' - Single, Retired Man) * % couples_without_kids per region https://bit.ly/4eJNCyD
    "CHR22 Single woman, 1 child, with work.General.Electricity.Bilbao"
        Single woman, 1 child = ('People 25-65' - Single with work) * % of monoparental per region https://bit.ly/4eJNCyD

    Sum of all ND2 = Couple both at Work + Couple over 65 years + Single woman, 1 child # transforming into a "weight" coeffs
    Couple both at Work = round(ND2 * (Couple both at Work/Sum of all ND2),0)
    Couple over 65 years = round(ND2 * (Couple over 65 years/Sum of all ND2),0)
    Single woman, 1 child = round(ND2 * (Single woman, 1 child/Sum of all ND2),0)
        
# ND3_5 (Three to Five People Occupied Dwellings)="Total number of dwellings" * "% Three to Five People Occupied Dwellings" per municipality https://bit.ly/3xCooBO
coef_family_1_child= random()    
coeff_employement = parado/total # table Población de 16 y más años por sexo y relación con la actividad (agrupado)
coeff_1_children = 0.46 # 15.4/33.5 https://www.ine.es/prensa/ech_2020.pdf
coeff_2_children = 0.44 # 14.7 / 33.5  https://www.ine.es/prensa/ech_2020.pdf
coeff_3_more_children = 0.1 # 3/33.5 https://www.ine.es/prensa/ech_2020.pdf

    "CHR03 Family, 1 child, both at work.General.Electricity.Bilbao" 
    "CHR45 Family with 1 child, 1 at work, 1 at home.General.Electricity.Bilbao"
    "CHR27 Family both at work, 2 children.General.Electricity.Bilbao"
    "CHR41 Family with 3 children, both at work.General.Electricity.Bilbao"
    "CHR20 Family one at work, one work home, 3 children.General.Electricity.Bilbao"
    "CHR52 Student Flatsharing.General.Electricity.Bilbao"

    Family, 1 child, both at work = (ND3_5 *coeff_1_children) * (coeff_employement)
    Family with 1 child, 1 at work, 1 at home = (ND3_5 *coeff_1_children) * (1-coeff_employement)
    Family both at work, 2 children = (ND3_5 *coeff_2_children) * (coeff_employement)
    Family with 3 children, both at work = (ND3_5 *coeff_3_more_children) * (coeff_employement)
    Family one at work, one work home, 3 children = (ND3_5 *coeff_3_more_children) * (1-coeff_employement)
    # Student Flatsharing =

# ND6_9 (Six to Nine People Occupied Dwellings)="Total number of dwellings" * "% Six to Nine People Occupied Dwellings" per municipality https://bit.ly/3xCooBO
list = [id_1, id_2, id_3] # three random profiles with Wokring peole, and unemployed, with 6,8,9 people in the house
    def select_random_elements(list, count):
        return random.choices(list, k=count)

    random_profiles = select_random_elements(list, ND6_9)
randmom(list) # random choice of the profile
# ND10 (Ten and more People Occupied Dwellings)="Total number of dwellings" * "% Ten and more People Occupied Dwellings" per municipality https://bit.ly/3xCooBO
list_2 = [id_1, id_2, id_3] # three random profiles with Wokring peole, and unemployed, with 10,11,12 people in the house
random_profiles = select_random_elements(list_2, ND10)

"""
""" 
# ND1 Single Occupied Dwellings:
    CHR07 Single with work	
    CHR13 Student with Work	
    CHR30 Single, Retired Man/Woman	
# ND2 Two People Occupied Dwellings:
    CHR01 Couple both at Work	
    CHR16 Couple over 65 years	
    CHR22 Single woman, 1 child, with work	

# ND3-5 Three to Five People Occupied Dwellings:
    CHR03 Family, 1 child, both at work.	
    CHR52 Student Flatsharing	
    CHR27 Family both at work, 2 children	
    CHR41 Family with 3 children, both at work	
    CHR45 Family with 1 child, 1 at work, 1 at home	
    CHR20 Family one at work, one work home, 3 children	
# ND6-9 Six to Nine People Occupied Dwellings:

# ND10 More than 10 People Occupied Dwellings:
"""


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
