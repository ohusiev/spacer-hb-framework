import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

class GeoHeatmapVisualizer:
    def __init__(self, census_file, data_file, sheet_name='Sheet3', filter_value='Comb2S1'):
        # Load census shapefile
        self.df_census = gpd.read_file(census_file)
        # Load data from Excel
        self.df_data = pd.read_excel(data_file, sheet_name=sheet_name, dtype={'census_id': str})
        # Filter data using the provided filter_value
        self.df_filter = self.df_data[self.df_data["Filter"] == filter_value]
        # Merge census and filtered data
        self.df = self.df_census.merge(self.df_filter, on='census_id')

    def plot_energy_co2_heatmaps(self):
        # 1. Heatmap of Energy Demand and CO₂ Emissions by Census Section (Geospatial)
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # Energy demand heatmap
        self.df.plot(column='Envelope_Energy_kWh', cmap='Reds', legend=True, ax=ax[0], edgecolor='black', linewidth=0.5)
        ax[0].set_title("Heatmap of Energy Demand by Census Section")
        # Adding labels for each element
        for x, y, label in zip(self.df.geometry.centroid.x, self.df.geometry.centroid.y, self.df['Envelope_Energy_kWh']):
            ax[0].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black')

        # CO2 Emissions heatmap
        self.df.plot(column='CO2_per_m2', cmap='Blues', legend=True, ax=ax[1], edgecolor='black', linewidth=0.5)
        ax[1].set_title("Heatmap of CO₂ Emissions by Census Section")
        # remove labels of y and x axis
        ax[0].set_yticklabels([])
        ax[0].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[1].set_xticklabels([])
        # Adding labels for each element
        for x, y, label in zip(self.df.geometry.centroid.x, self.df.geometry.centroid.y, self.df['CO2_per_m2']):
            ax[1].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black')

        plt.show()

    def geo_heatmap(self, column_1='NRPE_kWh_per_dwelling', column_2='EUAC_per_dwelling', vmin1=None, vmax1=None, vmin2=None, vmax2=None):
        # ADJUSTED HEATMAPS WITH adjustable colorbar
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # Column_1 heatmap
        self.df.plot(column=column_1, cmap='autumn', legend=True, ax=ax[0], edgecolor='black', linewidth=0.5, vmin=vmin1, vmax=vmax1)
        ax[0].set_title(f"Heatmap of `{column_1}` by Census Section")
        # Adding labels for each element
        for x, y, label in zip(self.df.geometry.centroid.x, self.df.geometry.centroid.y, self.df[column_1]):
            ax[0].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))

        # Column_2 heatmap
        self.df.plot(column=column_2, cmap='summer', legend=True, ax=ax[1], edgecolor='black', linewidth=0.5, vmin=vmin2, vmax=vmax2)
        ax[1].set_title(f"`{column_2}`")
        # remove labels of y and x axis
        ax[0].set_yticklabels([])
        ax[0].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[1].set_xticklabels([])
        # Adding labels for each element
        for x, y, label in zip(self.df.geometry.centroid.x, self.df.geometry.centroid.y, self.df[column_2]):
            ax[1].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))

        plt.subplots_adjust(wspace=0.008)  # Adjusts the width spacing between subplots

        plt.show()

    def geo_heatmap_no_colorbar(self, column_1='NRPE_kWh_per_dwelling', column_2='EUAC_per_dwelling'):
        # ADJUSTED HEATMAPS with non adjustable colorbar
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # Column_1 heatmap
        self.df.plot(column=column_1, cmap='autumn', legend=True, ax=ax[0], edgecolor='black', linewidth=0.5)
        ax[0].set_title(f"Heatmap of `{column_1}` by Census Section")
        # Adding labels for each element
        for x, y, label in zip(self.df.geometry.centroid.x, self.df.geometry.centroid.y, self.df[column_1]):
            ax[0].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))

        # Column_2 heatmap
        self.df.plot(column=column_2, cmap='summer', legend=True, ax=ax[1], edgecolor='black', linewidth=0.5)
        ax[1].set_title(f"Heatmap of `{column_2}` by Census Section")
        # remove labels of y and x axis
        ax[0].set_yticklabels([])
        ax[0].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[1].set_xticklabels([])
        # Adding labels for each element
        for x, y, label in zip(self.df.geometry.centroid.x, self.df.geometry.centroid.y, self.df[column_2]):
            ax[1].text(x, y, round(label, 2), fontsize=8, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))

        plt.subplots_adjust(wspace=0.008)  # Adjusts the width spacing between subplots

        plt.show()

# Example usage:
if __name__ == "__main__":
    census_file = r'H:\My Drive\01_PHD_CODE\chapter_5\spain\bilbao_otxarkoaga\vector\stat_census\Otxarkoaga.shp'
    data_file = r'H:\My Drive\01_PHD_CODE\chapter_5\spain\bilbao_otxarkoaga\data\06_buildings_with_energy_and_co2_values+HDemProj_facade_costs+PV_economic+EUAC_ORIGIN_ADD_SHEET.xlsx'

    # Pass the desired filter value as an argument
    visualizer = GeoHeatmapVisualizer(census_file, data_file, filter_value='Comb2S1')
    visualizer.plot_energy_co2_heatmaps()
    visualizer.geo_heatmap(column_1='NRPE_Envelope_kWh_per_m2', column_2='Envelope_EUAC_per_m2', vmin1=0, vmax1=200, vmin2=0, vmax2=50)
    visualizer.geo_heatmap_no_colorbar(column_1='NRPE_kWh_per_dwelling', column_2='EUAC_per_dwelling')
