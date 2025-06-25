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
class EnergyConsumptionProfileAssigner:
    def __init__(self, dwelling_percentages_dict=None):
        file_dir = os.getcwd()
        self.PATH = os.path.join(file_dir, "LoadProGen", "Bilbao")
        files = [file for file in os.listdir(self.PATH) if file.endswith('.csv')]
        file = os.path.join(self.PATH, files[0])
        self.df = pd.read_csv(file, sep=";")

        self.data = pd.read_excel(
            os.path.join(file_dir, "00_mod_04_input_data_census_id_ener_consum_profiling.xlsx"),
            sheet_name="04_dwelling_profiles_census", index_col=0
        )
        self.data = self.data.iloc[:, :12]

        # Fixed percentages for the different types of dwellings in Pais Vasco (Spain) (BASED On GIPUZKOA, as Example)
        # Define dictionaries for each group
        if dwelling_percentages_dict is None:
            dwelling_percentages_dict = {
                "single_dwellings": {
                "percentage_of_people_20_24_live_alone": 0.5,
                "percentage_of_people_25_65_live_alone": 0.25,
                "percentage_of_people_65_live_alone": 0.33
                },
                "two_people_dwellings": {
                "couples_25_65_without_kids": 0.11,
                "couples_65_without_kids": 0.33,
                "monoparental_25_65": 0.1
                },
                "three_five_people_dwellings": {
                "couples_25_65_with_kids": 0.47,
                "coeff_1_children": 0.46,
                "coeff_2_children": 0.44,
                "coeff_3_more_children": 0.1
                }
            }
            print("Using default dwelling percentages dictionary. BASED On GIPUZKOA (Pais Vasco) as Example")

        # Assign values from the main dictionary
        self.percentage_of_people_20_24_live_alone = dwelling_percentages_dict["single_dwellings"]["percentage_of_people_20_24_live_alone"]
        self.percentage_of_people_25_65_live_alone = dwelling_percentages_dict["single_dwellings"]["percentage_of_people_25_65_live_alone"]
        self.percentage_of_people_65_live_alone = dwelling_percentages_dict["single_dwellings"]["percentage_of_people_65_live_alone"]

        self.couples_25_65_without_kids = dwelling_percentages_dict["two_people_dwellings"]["couples_25_65_without_kids"]
        self.couples_65_without_kids = dwelling_percentages_dict["two_people_dwellings"]["couples_65_without_kids"]
        self.monoparental_25_65 = dwelling_percentages_dict["two_people_dwellings"]["monoparental_25_65"]

        self.couples_25_65_with_kids = dwelling_percentages_dict["three_five_people_dwellings"]["couples_25_65_with_kids"]
        self.coeff_1_children = dwelling_percentages_dict["three_five_people_dwellings"]["coeff_1_children"]
        self.coeff_2_children = dwelling_percentages_dict["three_five_people_dwellings"]["coeff_2_children"]
        self.coeff_3_more_children = dwelling_percentages_dict["three_five_people_dwellings"]["coeff_3_more_children"]
        # Unemployement range used as a coefficient to represent a possible weigth of families with employed and unemployed people

        self.data["unemployment_rate"] = 1 # REmove if needed a coefficient
        self.data["employment_rate"] = 1 #1-data["unemployment_rate"]

        self.coeff_unemployement = 1
        self.coeff_employement = 1
        # For Six to Nine and More people dwellings
        self.rest_16_25 = 1 - self.percentage_of_people_20_24_live_alone # asusming that kids under 16 can't live with parents or any tutelage
        self.rest_25_65 = 1 - self.percentage_of_people_25_65_live_alone - self.couples_25_65_without_kids - self.couples_25_65_with_kids - self.monoparental_25_65
        self.rest_65 = 1 - self.percentage_of_people_65_live_alone - self.couples_65_without_kids

        self.list_younger_people = ["6-9P_Occup_id_1", "6-9P_Occup_id_3"] # 
        self.list_older_people = ["6-9P_Occup_id_2", "6-9P_Occup_id_4"]

    def handle_column_multiplication(self, df, col1, col2, result_col):
        df[result_col] = (df[col1] * df[col2]).replace([np.inf, -np.inf], np.nan).fillna(0).round(0).astype(int)
        return df

    def calculate_dwellings(self, df):
        dwelling_types = [
            ('ND1 (Single Occupied Dwellings)', 'Single Occupied Dwellings per municipality'),
            ('ND2 (Two People Occupied Dwellings)', 'Two People Occupied Dwellings per municipality'),
            ('ND3_5 (Three to Five People Occupied Dwellings)', 'Three to Five People Occupied Dwellings per municipality'),
            ('ND6_9 (Six to Nine People Occupied Dwellings)', 'Six to Nine People Occupied Dwellings per municipality'),
            ('ND10 (Ten or more People Occupied Dwellings)', 'Ten or more People Occupied Dwellings per municipality')
        ]
        for result_col, proportion_col in dwelling_types:
            df = self.handle_column_multiplication(df, 'Total number of dwellings', proportion_col, result_col)
        return df

    def calculate_single_dwellings_types(self, df_input):
        df = df_input.copy()
        df['Stu_Work'] = df['age_group_2'] * self.percentage_of_people_20_24_live_alone
        df['1P_Work'] = df['age_group_3'] * self.percentage_of_people_25_65_live_alone
        df['1P_Ret'] = df['age_group_4'] * self.percentage_of_people_65_live_alone

        df['% 1P_Work'] = df['1P_Work'] / df['age_group_3']
        df['% Stu_Work'] = df['Stu_Work'] / df['age_group_2']
        df['% 1P_Ret'] = df['1P_Ret'] / df['age_group_4']

        df['Sum of all ND1'] = df['1P_Work'] + df['Stu_Work'] + df['1P_Ret']

        df['1P_Work'] = (df['ND1 (Single Occupied Dwellings)'] * (df['1P_Work'] / df['Sum of all ND1'])).astype(int)
        df['Stu_Work'] = (df['ND1 (Single Occupied Dwellings)'] * (df['Stu_Work'] / df['Sum of all ND1'])).astype(int)
        df['1P_Ret'] = (df['ND1 (Single Occupied Dwellings)'] * (df['1P_Ret'] / df['Sum of all ND1'])).astype(int)

        return pd.concat([df_input, df[['1P_Work', 'Stu_Work', '1P_Ret']]], axis=1)

    def calculate_two_people_dwellings_types(self, df_input):
        df = df_input.copy()
        df['Couple_Work'] = df['age_group_3'] * self.couples_25_65_without_kids * self.data["unemployment_rate"]
        df['Couple_65+'] = df['age_group_4'] * self.couples_65_without_kids
        df['1P_1Ch_Work'] = df['age_group_3'] * self.monoparental_25_65 + df['age_group_1'] * self.monoparental_25_65

        df['Sum of all ND2'] = df['Couple_Work'] + df['Couple_65+'] + df['1P_1Ch_Work']

        df['Couple_Work'] = (df['ND2 (Two People Occupied Dwellings)'] * (df['Couple_Work'] / df['Sum of all ND2'])).astype(int)
        df['Couple_65+'] = (df['ND2 (Two People Occupied Dwellings)'] * (df['Couple_65+'] / df['Sum of all ND2'])).astype(int)
        df['1P_1Ch_Work'] = (df['ND2 (Two People Occupied Dwellings)'] * (df['1P_1Ch_Work'] / df['Sum of all ND2'])).astype(int)

        return pd.concat([df_input, df[['Couple_Work', 'Couple_65+', '1P_1Ch_Work']]], axis=1)

    def calculate_three_five_people_dwellings_types(self, df_input):
        df = df_input.copy()
        df['Fam_1Ch_Work'] = df['age_group_3'] * self.couples_25_65_with_kids * self.data["employment_rate"] + df['age_group_1'] * self.coeff_1_children
        df['Fam_1Ch_1Wrk1Hm'] = df['age_group_3'] * self.couples_25_65_with_kids * self.data["unemployment_rate"] + df['age_group_1'] * self.coeff_1_children
        df['Fam_2Ch_Work'] = df['age_group_3'] * self.couples_25_65_with_kids * self.data["unemployment_rate"] + df['age_group_1'] * self.coeff_2_children
        df['Fam_3Ch_Work'] = df['age_group_3'] * self.couples_25_65_with_kids * self.data["employment_rate"] + df['age_group_1'] * self.coeff_3_more_children
        df['Fam_3Ch_1Wrk1Hm'] = df['age_group_3'] * self.couples_25_65_with_kids * self.data["employment_rate"] + df['age_group_1'] * self.coeff_3_more_children
        df['Stu_Share'] = df['age_group_2'] * self.rest_16_25 * 0.5
        df['Fam_3Ch_HmWrk'] = df['age_group_3'] * self.couples_25_65_with_kids * self.data["unemployment_rate"] + df['age_group_1'] * self.coeff_3_more_children

        df['Sum of all ND3_5'] = df['Fam_1Ch_Work'] + df['Fam_1Ch_1Wrk1Hm'] + df['Fam_2Ch_Work'] + df['Fam_3Ch_Work'] + df['Fam_3Ch_1Wrk1Hm'] + df['Stu_Share'] + df['Fam_3Ch_HmWrk']

        df['Fam_1Ch_Work'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)'] * (df['Fam_1Ch_Work'] / df['Sum of all ND3_5'])).astype(int))
        df['Fam_1Ch_1Wrk1Hm'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)'] * (df['Fam_1Ch_1Wrk1Hm'] / df['Sum of all ND3_5'])).astype(int))
        df['Fam_2Ch_Work'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)'] * (df['Fam_2Ch_Work'] / df['Sum of all ND3_5'])).astype(int))
        df['Fam_3Ch_Work'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)'] * (df['Fam_3Ch_Work'] / df['Sum of all ND3_5'])).astype(int))
        df['Fam_3Ch_1Wrk1Hm'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)'] * (df['Fam_3Ch_1Wrk1Hm'] / df['Sum of all ND3_5'])).astype(int))
        df['Stu_Share'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)'] * (df['Stu_Share'] / df['Sum of all ND3_5'])).astype(int))
        df['Fam_3Ch_HmWrk'] = ((df['ND3_5 (Three to Five People Occupied Dwellings)'] * (df['Fam_3Ch_HmWrk'] / df['Sum of all ND3_5'])).astype(int))
        return pd.concat([df_input, df[['Fam_1Ch_Work', 'Fam_1Ch_1Wrk1Hm', 'Fam_2Ch_Work',
                                       'Fam_3Ch_Work', 'Fam_3Ch_1Wrk1Hm', 'Stu_Share', 'Fam_3Ch_HmWrk']]], axis=1)

    def select_random_elements(self, lst, count):
        return random.choices(lst, k=count)

    def calculate_six_to_nine_people_dwellings_types(self, df_input):
        df = df_input.copy()
        df['6-9P_Occup_id_1'] = 0
        df['6-9P_Occup_id_3'] = 0

        for index, row in df[df['age_group_3'] > df['age_group_4']].iterrows():
            counter = int(row['ND6_9 (Six to Nine People Occupied Dwellings)'])
            df.at[index, '6-9P_Occup_id_1'] = counter
        for index, row in df[df['age_group_3'] <= df['age_group_4']].iterrows():
            counter = int(row['ND6_9 (Six to Nine People Occupied Dwellings)'])
            df.at[index, '6-9P_Occup_id_3'] = counter
        return pd.concat([df_input, df[['6-9P_Occup_id_1', '6-9P_Occup_id_3']]], axis=1)

    def calculate_ten_and_more_people_dwellings_types(self, df_input):
        from collections import Counter
        df = df_input.copy()
        df['10+P_Occup_id_1'] = 0
        df['10+P_Occup_id_2'] = 0

        for index, row in df[(df['age_group_3'] >= df['age_group_4']) & (df['age_group_2'] >= df['age_group_4'])].iterrows():
            selected_elements = self.select_random_elements(self.list_younger_people, int(row['ND10 (Ten or more People Occupied Dwellings)']))
            counter = Counter(selected_elements)
            df.at[index, '10+P_Occup_id_1'] = counter.get('10+P_Occup_id_1', 0)
        for index, row in df[(df['age_group_3'] < df['age_group_4']) & (df['age_group_2'] >= df['age_group_4'])].iterrows():
            selected_elements = self.select_random_elements(self.list_older_people, int(row['ND10 (Ten or more People Occupied Dwellings)']))
            counter = Counter(selected_elements)
            df.at[index, '10+P_Occup_id_2'] = counter.get('10+P_Occup_id_2', 0)
        return pd.concat([df_input, df[['10+P_Occup_id_1', '10+P_Occup_id_2']]], axis=1)

    def process(self):
        self.data = self.calculate_dwellings(self.data)
        print(self.data)
        self.data = self.data[self.data["Total number of dwellings"].notna()]
        self.data = self.calculate_single_dwellings_types(self.data)
        self.data = self.calculate_two_people_dwellings_types(self.data)
        self.data = self.calculate_three_five_people_dwellings_types(self.data)
        self.data = self.calculate_six_to_nine_people_dwellings_types(self.data)
        self.data = self.calculate_ten_and_more_people_dwellings_types(self.data).round(4)
        self.data.to_csv("data/04_energy_consumption_profiles/04_1_energy_consumption_profiles_real_data.csv", index="census_id")
#%%
# Usage
if __name__ == "__main__":
    assigner = EnergyConsumptionProfileAssigner()
    assigner.process()

# %%
