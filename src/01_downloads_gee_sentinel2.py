"""
-------------------------------------------------------------------------------
ML Pipeline for Remote Sensing of Agricultural Investment
Step 1: Data Retrieval
-------------------------------------------------------------------------------

Fetches and preprocesses Sentinel-2 imagery (L2A products) for detecting
irrigation ponds as a proxy for agricultural investment.

Queries, filters, and composites Sentinel-2 data (near-daily availability)
into annual mosaics. Data is publicly available from ESA but challenging to
process at scale directly, so this script leverages Google Earth Engine for
access and preprocessing, exporting one image per year to Google Drive.

Key Operations:
    1. Harmonization: Merges Sentinel-2 L2A data with cloud probability.
    2. Cloud Masking: Aggressive masking (SCL + probability < 20%).
    3. Compositing: Quarterly quality mosaics → annual medians.
    4. Export: Batches and submits tasks to Google Earth Engine.

Prerequisites:
    First runs may fail with authentication errors. Follow the links in
    the error messages to enable and register the GEE API, then re-run.
    Select non-academic use (unless not applicable) to register the GEE
    project that will make the API calls. It requires filling a small form.
"""

import time
import ee

# Configuration
# -----------------
PROJECT_ID = 'mlgis-ponds' # GEE project ID (must create before running)
COUNTRY = 'Mexico'
STATE = 'Michoacan'
START_YEAR = 2024
END_YEAR = 2024
FOLDER_NAME_PREFIX = 'GoogleEarthEngine_MexMichoacan'
SCALE = 10 # Export resolution in meters

# Earth Engine Initialization and Helpers
# ---------------------------------------
def get_aoi(country_name, state_name):
    """Get Area of Interest geometry (polygon) from FAO GAUL."""
    fc = ee.FeatureCollection('FAO/GAUL/2015/level1')  # L1=state, L0=country
    fc = fc.filter(ee.Filter.And(
        ee.Filter.eq('ADM0_NAME', country_name),
        ee.Filter.eq('ADM1_NAME', state_name)
    ))
    return fc.geometry()

def get_sentinel2_collection(aoi, start_date, end_date):
    """Fetch Sentinel-2 imagery joined with cloud probability."""
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(aoi)
                 .filterDate(start_date, end_date))

    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterBounds(aoi)
                        .filterDate(start_date, end_date))

    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(
        primary=s2_sr_col,
        secondary=s2_cloudless_col,
        condition=ee.Filter.equals(leftField='system:index',
                                   rightField='system:index')
    ))

def mask_sentinel2_clouds(image):
    """Cloud masking using SCL, Cloud Prob, and Brightness."""
    scl = image.select('SCL') # Scene Classification Layer (SCL)
    # Keep Veg (4), Soil (5), Water (6), Unclassified (7)
    scl_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))

    # Handle missing cloud probability bands conservatively
    cloud_prob = ee.Image(ee.Algorithms.If(
        image.get('s2cloudless'), # Check if cloud probability band exists
        ee.Image(image.get('s2cloudless')).select('probability'),
        ee.Image.constant(100) # If not, assign high cloud probability
    ))

    # If cloud pr <20% and not too bright (blue<2000), keep; else, mask out
    # Brightness measured on Blue band (B2), range 0-10000 (standard practice)
    # Ref: https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
    cloud_mask = cloud_prob.lt(20)
    brightness_mask = image.select('B2').lt(2000)

    return image.updateMask(scl_mask.And(cloud_mask).And(brightness_mask))

def create_quarterly_composite(aoi, start_date, end_date):
    """Creates a Best-Available-Pixel composite for a quarter."""
    s2_col = get_sentinel2_collection(aoi, start_date, end_date)
    masked_col = s2_col.map(mask_sentinel2_clouds)

    bands = ['B2', 'B3', 'B4', 'B5', 'B6',
             'B7', 'B8', 'B8A', 'B11', 'B12']

    # Guard against empty collection (prevents quarters with no data crashing)
    count = masked_col.size()
    if count.getInfo() == 0:
        return {
            'bands': None,
            'count': count
        }

    # Add quality score band (quality score = 100 - cloud probability)
    def add_quality_band(image):
        cloud_prob = ee.Image(ee.Algorithms.If(
            image.get('s2cloudless'),
            ee.Image(image.get('s2cloudless')).select('probability'),
            ee.Image.constant(100)
        ))
        quality = ee.Image.constant(100).subtract(cloud_prob).rename('quality')
        return image.addBands(quality)

    quality_col = masked_col.map(add_quality_band)

    # Create best-available-pixel composite using qualityMosaic
    bap_composite = (quality_col.qualityMosaic('quality')
                                .select(bands))

    return {
        'bands': bap_composite,
        'count': count
    }

def export_image(image, folder_name, file_name_prefix, aoi):
    """Handles the GEE export task submission."""
    task = ee.batch.Export.image.toDrive(
        image=image.clip(aoi).toFloat(),
        description=file_name_prefix,
        folder=folder_name,
        fileNamePrefix=file_name_prefix,
        region=aoi,
        scale=SCALE,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task.start()

def process_year(year, aoi):
    """Orchestrates quarterly processing and annual composite generation."""
    print(f'\n=== Processing Year {year} ===')

    valid_quarters = []
    folder_name = f'{FOLDER_NAME_PREFIX}_S2_{year}'

    for q in range(1, 5): #quarters (1, 2, 3, 4)
        start_month = (q - 1) * 3 + 1 #start months (1, 4, 7, 10)
        q_start = ee.Date.fromYMD(year, start_month, 1)
        q_end = q_start.advance(3, 'month')

        # Generate composite (Compute ONCE)
        result = create_quarterly_composite(aoi, q_start, q_end)
        count = result['count'].getInfo()

        if count > 0:
            valid_quarters.append(result['bands'])
            print(f'  Q{q} ({count} images): Valid.')
        else:
            print(f'  Q{q}: No images found. Skipping.')

    # Annual Median Generation
    if not valid_quarters:
        print(f'Error: No valid data for {year}. Skipping annual composite.')
        return

    print(f'  Building annual median from {len(valid_quarters)} quarters.')
    annual_median = ee.ImageCollection.fromImages(valid_quarters).median()

    # Export Annual Composite
    annual_desc = f'S2_{year}_ANNUAL_MEDIAN'
    export_image(annual_median, folder_name, annual_desc, aoi.bounds())


ee.Initialize(project=PROJECT_ID)
aoi_geom = get_aoi(COUNTRY, STATE)

start_time = time.time()

ALL_YEARS = list(range(START_YEAR, END_YEAR + 1))

print(f'--- PROCESSING YEARS {START_YEAR} to {END_YEAR} ---')
print(f'Generating {len(ALL_YEARS)} annual composite(s).')

for year in ALL_YEARS:
    try:
        # Year processing: quarterly exports + QA-filtered annual composite
        process_year(year, aoi_geom)
        time.sleep(5)  # Small delay between years
    except Exception as e:
        print(f'ERROR during processing for year {year}: {e}')
        continue  # Continue with next year even if one fails

print('\n--- All tasks submitted. Monitor GEE Tasks page. ---')
print(f'Elapsed time for submission: {time.time() - start_time:.1f}s')
