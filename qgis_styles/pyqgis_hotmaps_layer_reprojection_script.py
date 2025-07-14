import processing


CRS = 'EPSG:25830'

# Step 1: Polygonize the raster
polygonized = processing.run("gdal:polygonize", {
    'INPUT': 'C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework/raster/heat_res_curr_density_clip.tif',
    'BAND': 1,
    'FIELD': 'HDemProj',
    'EIGHT_CONNECTEDNESS': False,
    'EXTRA': '',
    'OUTPUT': 'C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework/vector/hotmaps_raster2vector.geojson'
})['OUTPUT']
"""
# Step 2: Reproject to EPSG:25830
reprojected = processing.run("native:reprojectlayer", {
    'INPUT': polygonized,
    'TARGET_CRS': QgsCoordinateReferenceSystem(CRS),
    'OUTPUT': 'C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework/vector/hotmaps_raster2vector_reprojected.shp'
})['OUTPUT']
"""
# Copied command from the `Reproject layer` Processing Toolbox of QGIS
processing.run("native:reprojectlayer", {
    'INPUT':'C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework/vector/hotmaps_raster2vector.geojson',
    'TARGET_CRS':QgsCoordinateReferenceSystem(CRS),
    'CONVERT_CURVED_GEOMETRIES':False,
    'OPERATION':'+proj=pipeline +step +inv +proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +step +proj=utm +zone=30 +ellps=GRS80',
    'OUTPUT':'C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework/vector/hotmaps_raster2vector_reprojected.geojson'})
    

"""
# Step 3: Join attributes by location (Assuming you join with another layer in EPSG:25830)
# Define the layer to join with (replace with actual path or layer reference)
join_layer_path = 'C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework/vector/buildings_footprint/etrs_25830/buildings_inspire_clip_oxarkoaga+census.shp'

joined = processing.run("native:joinattributesbylocation", {
    'INPUT': join_layer_path,
    'PREDICATE': [0],  # intersects
    'JOIN': reprojected ,
    'JOIN_FIELDS': ['HDem'],
    'METHOD': 2,  # Take attributes of the first matching feature only
    'DISCARD_NONMATCHING': False,
    'PREFIX': '',
    'OUTPUT': 'C:/Users/Oleksandr-MSI/Documentos/GitHub/spacer-hb-framework/vector/hotmaps_raster2vector+HDem.shp'
})['OUTPUT']
"""
