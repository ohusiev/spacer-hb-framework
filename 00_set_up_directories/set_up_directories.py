import os

class CustomDirectoryStructure:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.directory_structure = [
            f"{base_dir}/",
            f"{base_dir}/lidar/",
            f"{base_dir}/pyqgis_whitebox_tool/",
            f"{base_dir}/pyqgis_whitebox_tool/working_files/",
            f"{base_dir}/raster/",
            f"{base_dir}/vector/",
            f"{base_dir}/vector/cadaster/",
            f"{base_dir}/vector/cadaster/INSPIRE_BUILDING/",
            f"{base_dir}/vector/cadaster/KATASTROA_CATASTRO/",
            f"{base_dir}/vector/height/",
            f"{base_dir}/vector/osm_buildings/",
            f"{base_dir}/vector/osm_buildings/cadastral_import/",
            f"{base_dir}/vector/osm_buildings/etrs_25830/",
            f"{base_dir}/vector/stat_census/",
            f"{base_dir}/vector/whitebox_tool/",
            f"{base_dir}/LoadProGen/"
        ]

    def create_directories(self):
        for directory in self.directory_structure:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            else:
                print(f"Directory already exists: {directory}")
""" 
# Principal root directory and subdirectory structure
base_dir = "your_custom_name"  # Replace "your_custom_name" with any directory name you want

# Define the directory structure

directory_structure = [
    f"{base_dir}/",
    f"{base_dir}/lidar/",
    f"{base_dir}/pyqgis_whitebox_tool/",
    f"{base_dir}/pyqgis_whitebox_tool/working_files/",
    f"{base_dir}/raster/",
    f"{base_dir}/vector/",
    f"{base_dir}/vector/cadaster/",
    f"{base_dir}/vector/cadaster/INSPIRE_BUILDING/",
    f"{base_dir}/vector/cadaster/KATASTROA_CATASTRO/",
    f"{base_dir}/vector/height/",
    f"{base_dir}/vector/osm_buildings/",
    f"{base_dir}/vector/osm_buildings/cadastral_import/",
    f"{base_dir}/vector/osm_buildings/etrs_25830/",
    f"{base_dir}/vector/stat_census/",
    f"{base_dir}/vector/whitebox_tool/",
    f"{base_dir}/LoadProGen/Bilbao/"
]
# Create all directories if they do not exist
def create_directories(directory_structure):
    for directory in directory_structure:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

create_directories(directory_structure)
# %%
"""