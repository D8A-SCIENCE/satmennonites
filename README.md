# satmennonites
Description of files in this directory:
./CNN-ResNet50-FT-final.py        --> Script with ResNet50 model implementation 
./MennoniteCommuneMap.gpkg        --> Geographic database of Mennonite Communes in North and South America [based on de la Waroux et al (2021), DOI: 10.1080/1747423X.2020.1855266]
./geoBoundaries-BOL-ADM1.geojson  --> Level 1 administrative boundaries of Bolivia [geoboundaries.org]
./dlGridImgs.py                   --> Script to generate units of observations by gridding Santa Cruz department of Bolivia, and also to download NICFI images of each grid
./libraries.txt                   --> A list of Python libraries used by all scripts and their respective versions
./data/scruzGridGDF.shp           --> Vector file to store geometries of the grids and their metadata, such as land cover type, path to satellite image, image class (COMM or OTHER)
    
This code workflow assumes a few things:
- That you have access to Google Earth Engine (GEE)
- That you have access to NICFI Satellite Data Program mosaic on GEE. See this link on on how to get access:
  https://developers.planet.com/docs/integrations/gee/nicfi/

To test this code, first execute the script dlGridImgs.py to generate units of observation and to download satellite imagery, then CNN-ResNet50-FT-final.py to run the CNN classifier model.
