import geopandas as gpd
import shapely, geojson
import numpy as np
import os, datetime, time, cv2

from google.oauth2 import service_account
import requests, io
import ee

## authenticate to GEE/GCP using a service account file (.JSON file)
GCP_AUTH_FILE = os.getenv('GCP_AUTH_FILE')
credentials = ee.ServiceAccountCredentials(service_account, GCP_AUTH_FILE)
ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')

# read Bolivia vector file and filter for Santa Cruz department 
bolivia = gpd.read_file('geoBoundaries-BOL-ADM1.geojson')
scruz = gpd.GeoSeries(bolivia.iloc[7].geometry, crs=bolivia.crs)

# read polygon geometries for Mennonite communes of Bolivia
communes = gpd.read_file('MennoniteCommuneMap.gpkg', crs="EPSG:4326")
communes = shapely.unary_union(communes.make_valid().geometry).intersection(scruz.geometry)

xmin,ymin,xmax,ymax = scruz.total_bounds

# spatial resolution of planet imagery
resn = 4.77731426716

# how many cells across and down
cellSize = (resn*512)/(1e6/9) # in degrees 
# cellSize= (xmax-xmin)/30

gridCells=[]
for x0 in np.arange(xmin, xmax+cellSize, cellSize):
    for y0 in np.arange(ymin, ymax+cellSize, cellSize):
        x1 = x0-cellSize # bounds
        y1 = y0+cellSize
        gridCells.append(shapely.box(x0, y0, x1, y1))

scruzGrid = gpd.GeoSeries(gridCells, crs=bolivia.crs)

# create columns for lcType and imgFP
scruzGridGDF = gpd.GeoDataFrame(scruzGrid, geometry=scruzGrid).drop(columns=[0])
N = len(scruzGrid)
scruzGridGDF['lcType'] = ['lcType']*N
scruzGridGDF['gridClass'] = ['NULL']*N
scruzGridGDF['imgFP'] = ['imgFP']*N

# create directory to save downloaded grid image files
os.makedirs('GridImgsDL', exist_ok=True)

lcTypeMap = {10: 'TREECOVER', 20: 'SHRUB', 30: 'GRASSLAND',40: 'CROPLAND', 50: 'BUILT', 
              60: 'BARE', 70: 'SNOW/ICE', 80: 'WATER' , 90: 'HERBACEOUS WETLAND', 95: 'MANGROVES', 100: 'MOSS/LICHEN'}

dlTileCount=1
gridCount=1
for i in range(len(scruzGrid)):
    print(f'processing tile {gridCount}/{len(scruzGrid)}')
    gridCount+=1
    
    gridGeom = scruzGrid[i]
    geomEE = ee.Geometry(gridGeom.__geo_interface__)

    # because gridding was on the geometry of the BBox of Santa Cruz, 
    # here we check for tiles actually fully within Santa Cruz 
    # or within extent of already downloaded tile
    if scruz.geometry[0].intersection(gridGeom).area != gridGeom.area:
        continue

    # here we'll create COMM1,COMM2, NULL, OTHER classes
    # COMM1: ;COMM2: ;NULL: ;OTHER
    gridClass='NULL'
    commIntsec = communes.geometry[0].intersection(gridGeom)
    if (commIntsec.area > gridGeom.area*0.35):
        gridClass='COMM1'
    elif ((commIntsec.area > 0) and (commIntsec.area < gridGeom.area*0.35)):
        gridClass='COMM2'
    elif commIntsec.area == 0:
        gridClass='OTHER'

    scruzGridGDF.loc[i,'gridClass'] = gridClass

    ## Determining LC class based on ESA WorldCover V100
    lcESA = ee.ImageCollection('ESA/WorldCover/v200')\
                .filterBounds(geomEE)\
                .filterDate('2020-01-01','2022-01-01').first()
    
    pixCounts = lcESA.reduceRegion(
        reducer= ee.Reducer.frequencyHistogram(),
        geometry= geomEE, scale= 10, maxPixels= 1e9).getInfo()['Map']
    lcTypes = {lcTypeMap[int(k)]:v for k,v in pixCounts.items()}
    lcType = max(lcTypes, key=lcTypes.get)
    
    scruzGridGDF.loc[i,'lcType'] = lcType
    
    # skip grid if land cover type is not CROPLAND; we're only processing CROPLAND grids
    if lcType != 'CROPLAND':
        pass

    try:
        NICFI = ee.ImageCollection('projects/planet-nicfi/assets/basemaps/americas')
        IMG_COLL = NICFI.filterDate('2020-10-01','2020-11-01').filterBounds(geomEE)
        
        # We only download a tile if all of it fits in ONE satellite SCENE, else we skip!
        if IMG_COLL.size().getInfo() > 1:
            continue
            
        imgID = f'GRID_{i}_{lcType}_{gridClass}'
        print(f'processing {imgID}')

        # select Red, Green, Blue, Near-Infrared bands
        IMG = ee.Image(IMG_COLL.first().select(['R','G','B','N']).multiply(255/1e4)).toUint8()
        url = IMG.getDownloadUrl({
            'bands': ['R','G','B','N'],
            'region': geomEE, 'scale': resn, 
            'format': 'NPY'})
        resp = requests.get(url)
        resData = np.load(io.BytesIO(resp.content))
        imgArrDl = np.dstack([resData[b] for b in ['R','G','B','N']])

        # save grid image as a .npy file
        with open(f'GridImgsDL/{imgID}.npy', 'wb') as f:
            np.save(f, imgArrDl, allow_pickle=False)
        
        scruzGridGDF.loc[i,'imgFP'] = imgID
        
        # log grid image download in geoDF file every 25 grids
        if dlTileCount%25==0:
            # GEEUtils.downloadDriveImages(f'PBMAPS_DL/GRIDS_{dt}', delete_from_drive=True)
            scruzGridGDF.to_file(f'{TEMP_DIR}/scruzGridGDF_{dt}.shp')
            time.sleep(30)

    except ConnectionError as e:
        # allow script to sleep for 5 mins, retry 25 times, authenticate
        # and if connection errors still persist quit
        print('Connection error occured, pausing for 3 mins then restarting')
        print(e)
        time.sleep(300)
        GEEUtils.AuthGEE()
        if retry>25:
            print('Quiting afer retrying for too long!')
            break
        retry+=1
        continue

    dlTileCount+=1

os.makedirs('data', exist_ok=True)
scruzGridGDF.to_file('data/scruzGridGDF.shp')
print(dlTileCount)
