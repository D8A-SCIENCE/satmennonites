# Replication Code for the Paper "Satellite Based Detection of Sociocultural Patterns: The Case of Mennonite Colonies in Bolivia"

## Project Description
This repository contains code and data for analyzing Mennonite communes in North and South America using satellite imagery and deep learning.

## Directory Contents

- `CNN-ResNet50-FT-final.py` - Script with ResNet50 model implementation
- `MennoniteCommuneMap.gpkg` - Geographic database of Mennonite Communes in North and South America [based on de la Waroux et al (2021), DOI: 10.1080/1747423X.2020.1855266]
- `geoBoundaries-BOL-ADM1.geojson` - Level 1 administrative boundaries of Bolivia [geoboundaries.org]
- `dlGridImgs.py` - Script to generate observation units by gridding Santa Cruz department of Bolivia, and to download NICFI images of each grid
- `libraries.txt` - List of Python libraries and their versions used by all scripts
- `data/scruzGridGDF.shp` - Vector file storing grid geometries and metadata (land cover type, satellite image path, image class: COMM or OTHER)

## Prerequisites

- Access to Google Earth Engine (GEE)
- Access to NICFI Satellite Data Program mosaic on GEE
  - [How to get access](https://developers.planet.com/docs/integrations/gee/nicfi/)

## Usage

1. First, execute `dlGridImgs.py` to generate observation units and download satellite imagery
2. Then, run `CNN-ResNet50-FT-final.py` to execute the CNN classifier model
