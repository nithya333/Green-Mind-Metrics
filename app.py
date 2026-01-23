import os
import io
import numpy as np
import pandas as pd
import rasterio
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
import io
from flask import send_file
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, jsonify, send_file, abort, render_template

app = Flask(__name__)

# --- CONFIGURATION ---
DATA_FOLDER = "./data_folder"

# 1. METRIC GROUPS (For the frontend checkboxes)
METRICS_CONFIG = {
    "Student Well-being (Survey)": ["Avg_Stress", "Avg_Mood", "Avg_Stress_Relief", "Avg_Focus_Boost", "Perceived_Greenness", "Avg_Time_In_Green"],
    "Vegetation Indices": ["NDVI_mean", "GNDVI_mean", "SAVI_mean", "png_1_ratio", "png_2_ratio"],
    "Urban & Structure": ["NDBI_mean", "NDUI_mean", "UHSI_mean", "LST_mean", "built_area_km2", "built_height_m"],
    "Water & Env": ["NDWI_mean", "NMDI_mean", "Shade_mean", "AOT_mean", "WVP_mean", "pm25(microg/10m3)"],
    "LULC Ratios": ["1_ratio", "2_ratio", "3_ratio", "4_ratio", "5_ratio", "6_ratio", "7_ratio", "8_ratio", "9_ratio"]
}

# 2. IMAGE LAYERS CONFIG
LAYER_CONFIG = {
    # --- VISUALS (From your snippet) ---
    'zoomout': {'type': 'tif_rgb', 'file_pattern': 'Satellite_{code}.tif', 'name': 'Satellite (Zoom Out)'},
    'zoomin':  {'type': 'png',     'file_pattern': 'High_Res_Satellite_{code}.png', 'name': 'Satellite (High Res)'},
    'lulcout': {'type': 'png',     'file_pattern': 'lulc_mask_{code}.png', 'name': 'LULC Mask (Zoom Out)'},
    'lulcin':  {'type': 'png',     'file_pattern': 'png_lulc_mask_{code}.png', 'name': 'LULC Mask (High Res)'},

        # --- VEGETATION INDICES ---
    'ndvi':   {'type': 'heatmap', 'band_index': 1,  'cmap': 'RdYlGn', 'name': 'NDVI (Vegetation)'},
    'gndvi':  {'type': 'heatmap', 'band_index': 4,  'cmap': 'Greens', 'name': 'GNDVI (Green Chlorophyll)'},
    'savi':   {'type': 'heatmap', 'band_index': 14, 'cmap': 'YlGn',   'name': 'SAVI (Soil Adjusted Veg)'},
    
    # --- URBAN & BUILT-UP ---
    'ndbi':   {'type': 'heatmap', 'band_index': 7,  'cmap': 'Reds',   'name': 'NDBI (Built-Up Index)'},
    'ndui':   {'type': 'heatmap', 'band_index': 3,  'cmap': 'OrRd',   'name': 'NDUI (Urban Index)'},
    'uhsi':   {'type': 'heatmap', 'band_index': 15, 'cmap': 'inferno','name': 'UHSI (Urban Heat Stress)'},
    
    # --- WATER & MOISTURE ---
    'ndwi':   {'type': 'heatmap', 'band_index': 5,  'cmap': 'Blues',  'name': 'NDWI (Water Content)'},
    'nmdi':   {'type': 'heatmap', 'band_index': 8,  'cmap': 'PuBu',   'name': 'NMDI (Soil Moisture)'},
    'wvp':    {'type': 'heatmap', 'band_index': 11, 'cmap': 'cool',   'name': 'Water Vapor (Atmosphere)'},
    
    # --- TEMPERATURE & SOIL ---
    'lst':    {'type': 'heatmap', 'band_index': 13, 'cmap': 'magma',  'name': 'LST (Land Surface Temp)'},
    'ndbsi':  {'type': 'heatmap', 'band_index': 9,  'cmap': 'YlOrBr', 'name': 'NDBSI (Bare Soil)'},
    'nbadi':  {'type': 'heatmap', 'band_index': 10, 'cmap': 'copper', 'name': 'NBaDI (Barren Land)'},
    
    # --- FIRE & SHADOW ---
    'nbr':    {'type': 'heatmap', 'band_index': 2,  'cmap': 'Gist_Heat', 'name': 'NBR (Burn Ratio)'},
    'shade':  {'type': 'heatmap', 'band_index': 6,  'cmap': 'gray',      'name': 'Shade Index'},
    
    # --- ATMOSPHERE ---
    'aot':    {'type': 'heatmap', 'band_index': 12, 'cmap': 'cividis', 'name': 'Aerosol Optical Thickness'}
}

LULC_CLASSES = {
    "Water" : "#1A5BAB",
    "Trees" : "#358221",
    "Flooded Vegetation" : "#87D19E",
    "Crops" : "#FFDB5C",
    "Built Area" : "#ED022A",
    "Bare Ground" : "#EDE9E4",
    "Snow/Ice" : "#F2FAFF",
    "Clouds" : "#C8C8C8",
    "Rangeland" : "#C6AD8D"
}

LEGEND_RANGES = {
    # --- VEGETATION ---
    # NDVI: -1 is water, 0 is bare soil, 1 is dense forest.
    # We clip -0.2 to 0.8 to make vegetation green variation clearly visible.
    'ndvi':  (-0.2, 0.8),  

    # --- URBAN ---
    # NDBI: Positive values indicate built-up/urban areas. Negative is vegetation.
    # -0.5 to 0.5 provides the best contrast between city and nature.
    'ndbi':  (-0.5, 0.5),

    # --- TEMPERATURE ---
    # LST: In Degrees Celsius. Adjust based on your season/region.
    # 20°C (Blue/Cool) to 50°C (Red/Hot) covers most Indian contexts well.
    'lst':   (20, 50),

    # --- ATMOSPHERE ---
    # AOT (Air Quality): Aerosol Optical Thickness usually ranges from 0 (Clear) to 1 (Hazy).
    # Values > 1.0 indicate severe pollution/dust storms.
    'aot':   (0.0, 1.0),

    # --- SHADE ---
    # Shade Index: Assuming this is a normalized ratio or probability.
    # 0 = No Shadow, 1 = Deep Shadow.
    'shade': (0.0, 1.0),
    
    # Fallback for others
    'default': (-1, 1)
}
def normalize_df(series, invert=False):
    """
    Normalizes a pandas series to a 0-100 scale.
    If invert=True, lower values get higher scores (good for pollution/stress).
    """
    min_val = series.min()
    max_val = series.max()
    
    # Avoid division by zero if all values in the column are identical
    if max_val == min_val:
        return 50  # Return a neutral middle score
    
    if invert:
        # Formula for "Lower is Better" (e.g., Pollution)
        return ((max_val - series) / (max_val - min_val)) * 100
    else:
        # Formula for "Higher is Better" (e.g., Greenery)
        return ((series - min_val) / (max_val - min_val)) * 100

def get_leaderboard_data():
    # 1. Load Data
    geo_df = pd.read_csv('geospatial_college_data_full.csv')
    psycho_df = pd.read_csv('psychometric_college_data.csv', encoding='cp1252')

    # 2. Process Psychometric Data
    # Map qualitative responses to numeric scores
    green_stress_map = {'Decreases a lot': 100, 'Decreases moderately': 75, 'No change': 50, 'Increases slightly': 25, 'Increases a lot': 0}
    refresh_map = {'Always': 100, 'Often': 75, 'Sometimes': 50, 'Rarely': 25, 'Never': 0}

    psycho_df['green_stress_score'] = psycho_df['When you spend time in green areas, how does your stress level change?'].map(green_stress_map).fillna(50)
    psycho_df['refresh_score'] = psycho_df['How often do you feel mentally refreshed after being outdoors in greenery?'].map(refresh_map).fillna(50)

    # Aggregation by college
    psycho_agg = psycho_df.groupby('Short Name').agg({
        'How would you rate your overall mood generally on a scale from 1 to 10': 'mean',
        'How would you describe your average stress level?': 'mean',
        'How visually green do you consider your campus?': 'mean',
        'green_stress_score': 'mean',
        'refresh_score': 'mean'
    }).reset_index()

    psycho_agg.columns = ['short_name', 'mood_score_raw', 'stress_level_raw', 'perceived_greenery_raw', 'green_impact_score', 'refresh_freq_score']

    # Normalize Psychometric Aggregates
    psycho_agg['mood_score'] = psycho_agg['mood_score_raw'] * 10  # Scale 1-10 to 10-100
    psycho_agg['stress_score_inverted'] = (11 - psycho_agg['stress_level_raw']) * 10 # Lower stress is better
    psycho_agg['perceived_greenery'] = psycho_agg['perceived_greenery_raw'] * 10

    # 3. Merge Datasets
    merged_df = pd.merge(geo_df, psycho_agg, on='short_name', how='left')
    merged_df.fillna(merged_df.mean(numeric_only=True), inplace=True) # Handle missing values

    # 4. Calculate Category Scores (0-100 Scale)
    
    # Category 1: Greenery (Physical + Perceived)
    merged_df['score_greenery'] = (
        normalize_df(merged_df['NDVI_mean']) + 
        normalize_df(merged_df['GNDVI_mean']) + 
        merged_df['perceived_greenery']
    ) / 3

    # Category 2: Environment (Air, Temp - Lower is better)
    merged_df['score_environment'] = (
        normalize_df(merged_df['pm25(microg/10m3)'], invert=True) + 
        normalize_df(merged_df['NO2_mol_per_m2'], invert=True) + 
        normalize_df(merged_df['LST_mean'], invert=True)
    ) / 3

    # Category 3: Socio-Urban (Built-up, Lights, Urbanization)
    merged_df['score_socio_urban'] = (
        normalize_df(merged_df['NDBI_mean']) + 
        normalize_df(merged_df['night_light_intensity']) + 
        normalize_df(merged_df['degree_of_urbanization'])
    ) / 3

    # Category 4: Student Health/Mind (Mood, Stress, Refreshment)
    merged_df['score_health'] = (
        merged_df['mood_score'] + 
        merged_df['stress_score_inverted'] + 
        merged_df['green_impact_score'] + 
        merged_df['refresh_freq_score']
    ) / 4

    # Overall Score (Average of 4 categories)
    merged_df['score_overall'] = (
        merged_df['score_greenery'] + 
        merged_df['score_environment'] + 
        merged_df['score_socio_urban'] + 
        merged_df['score_health']
    ) / 4

    # Rounding
    score_cols = ['score_greenery', 'score_environment', 'score_socio_urban', 'score_health', 'score_overall']
    merged_df[score_cols] = merged_df[score_cols].round(1)

    # 5. Prepare Response
    return merged_df[['name', 'short_name', 'score_overall', 'score_greenery', 'score_environment', 'score_socio_urban', 'score_health', 'mood_score_raw', 'stress_level_raw']].to_dict(orient='records')

@app.route('/api/leaderboard')
def leaderboard_api():
    data = get_leaderboard_data()
    return jsonify(data)

def load_data():
    try:
        # 1. Load Geospatial Data
        geo_df = pd.read_csv("geospatial_college_data_full.csv", encoding='cp1252')
        geo_df['short_name'] = geo_df['short_name'].astype(str).str.strip().str.upper()

        # 2. Load Psychometric Data
        psych_file = "psychometric_college_data.csv"
        
        if os.path.exists(psych_file):
            psych_df = pd.read_csv(psych_file, encoding='cp1252')
            psych_df['Short Name'] = psych_df['Short Name'].astype(str).str.strip().str.upper()

            # --- MAPPINGS: Convert Text to Numbers ---
            stress_relief_map = {
                'Decreases a lot': 3, 'Decreases moderately': 2, 'Decreases slightly': 1,
                'No change': 0, 'Increases slightly': -1, 'Increases moderately': -2, 'Increases a lot': -3
            }
            focus_map = {
                'Improves significantly': 3, 'Improves slightly': 1, 
                'No change': 0, 'Worsens': -1
            }
            
            # Apply Mappings safely
            col_stress_change = 'When you spend time in green areas, how does your stress level change?'
            if col_stress_change in psych_df.columns:
                psych_df['Avg_Stress_Relief'] = psych_df[col_stress_change].map(stress_relief_map).fillna(0)

            col_focus = 'How does your concentration/focus change in green over non-green areas?'
            if col_focus in psych_df.columns:
                psych_df['Avg_Focus_Boost'] = psych_df[col_focus].map(focus_map).fillna(0)

            # # Aggregate by College
            # agg_rules = {
            #     'How would you describe your average stress level?': 'mean',
            #     'How would you rate your overall mood generally on a scale from 1 to 10': 'mean',
            #     'How visually green do you consider your campus?': 'mean',
            #     'On a typical day, how much of your total time on campus do you spend in open green areas?': 'mean'
            # }
            # # Only aggregate columns that actually exist
            # valid_agg = {k: v for k, v in agg_rules.items() if k in psych_df.columns}
            
            # # Add calculated columns to aggregation
            # if 'Avg_Stress_Relief' in psych_df.columns: valid_agg['Avg_Stress_Relief'] = 'mean'
            # if 'Avg_Focus_Boost' in psych_df.columns: valid_agg['Avg_Focus_Boost'] = 'mean'

            # college_stats = psych_df.groupby('Short Name').agg(valid_agg).reset_index()

            # # Rename for Dashboard
            # rename_map = {
            #     'How would you describe your average stress level?': 'Avg_Stress',
            #     'How would you rate your overall mood generally on a scale from 1 to 10': 'Avg_Mood',
            #     'How visually green do you consider your campus?': 'Perceived_Greenness',
            #     'On a typical day, how much of your total time on campus do you spend in open green areas?': 'Avg_Time_In_Green'
            # }
            # college_stats.rename(columns=rename_map, inplace=True)

            # # 3. Merge (Left Join to keep map data even if no survey data)
            # merged_df = pd.merge(geo_df, college_stats, left_on='short_name', right_on='Short Name', how='left')
            # return merged_df

            # Aggregate
            college_stats = psych_df.groupby('Short Name').agg({
                'How would you describe your average stress level?': 'mean',
                'How would you rate your overall mood generally on a scale from 1 to 10': 'mean',
                'How visually green do you consider your campus?': 'mean'
            }).reset_index()

            college_stats.rename(columns={
                'How would you describe your average stress level?': 'Avg_Stress',
                'How would you rate your overall mood generally on a scale from 1 to 10': 'Avg_Mood',
                'How visually green do you consider your campus?': 'Perceived_Greenness'
            }, inplace=True)

            merged_df = pd.merge(geo_df, college_stats, left_on='short_name', right_on='Short Name', how='left')

            # --- NEW CALCULATION LOGIC ---
            # 1. Green Score (0-100): Weighted avg of NDVI (Veg) and GNDVI (Health)
            # NDVI is -1 to 1. We assume values > 0.
            merged_df['Green_Score'] = ((merged_df['NDVI_mean'] + merged_df['GNDVI_mean']) / 2) * 100
            
            # 2. Mind Score (0-100): High Mood + Low Stress
            # Stress is 1-10 (Bad), Mood is 1-10 (Good).
            # Formula: Average of (Mood) and (10 - Stress) * 10
            merged_df['Mind_Score'] = ( (merged_df['Avg_Mood'] + (10 - merged_df['Avg_Stress'])) / 2 ) * 10

            return merged_df.fillna(0)

        else:
            print("Warning: Psychometric data not found. Loading only geospatial.")
            return geo_df.fillna(0)

    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

def normalize(array):
    array = np.nan_to_num(array)
    if array.max() == array.min(): return array
    return (array - array.min()) / (array.max() - array.min())

# --- ROUTES ---

@app.route('/')
def index():
    return send_file('./templates/dashboard.html') # Ensure dashboard.html is in same folder or templates

@app.route('/psychometric')
def psychometric():
    return send_file('./templates/psychometric.html')

@app.route('/leaderboard')
def leaderboard():
    return send_file('./templates/leaderboard.html')

@app.route('/api/config')
def get_config():
    """Send the Metric Groups to Frontend"""
    return jsonify(METRICS_CONFIG)

@app.route('/api/colleges')
def get_colleges():
    if df.empty: return jsonify([])
    # Return all columns so frontend can graph them
    # df_clean = df.where(pd.notnull(df), None)
    df_clean = df.fillna(0)
    return jsonify(df_clean.to_dict(orient='records'))

@app.route('/api/image/<college_code>/<layer_type>')
def get_image(college_code, layer_type):
    if college_code == 'undefined': abort(404)
    
    config = LAYER_CONFIG.get(layer_type)
    if not config: abort(404)

    # base_path = f"{DATA_FOLDER}/{college_code}"
    base_path = "./Deployment_EarthEngineExports"
    
    try:
        img_data = None
        cmap = None

        # --- TYPE 1: PNG FILES (Direct Load) ---
        if config['type'] == 'png':
            fpath = f"{base_path}/{config['file_pattern'].format(code=college_code)}"
            if not os.path.exists(fpath): 
                print(f"Missing: {fpath}")
                abort(404)
            # img_data = plt.imread(fpath) # Returns numpy array
            return send_file(fpath, mimetype='image/png')
            
        # --- TYPE 2: TIF RGB (Satellite Zoom Out) ---
        elif config['type'] == 'tif_rgb':
            fpath = f"{base_path}/{config['file_pattern'].format(code=college_code)}"
            if not os.path.exists(fpath): abort(404)
            with rasterio.open(fpath) as src:
                # Assuming Bands 1,2,3 are RGB based on your snippet
                r = normalize(src.read(1))
                g = normalize(src.read(2))
                b = normalize(src.read(3))
                img_data = np.dstack((r, g, b))

        # --- TYPE 3: SENTINEL FEATURES (Calculated) ---
        elif config['type'] == 'heatmap':
            # Assuming features are in sentinel_features.tif
            fpath = f"{base_path}/Features_{college_code}.tif"
            if not os.path.exists(fpath): abort(404)
            with rasterio.open(fpath) as src:
                img_data = src.read(config['band_index']+1, masked=True)
                img_data = np.nan_to_num(img_data)
                cmap = config['cmap']

        # # --- RENDER TO BUFFER ---
        # fig = plt.figure(figsize=(5, 5))
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)

        # if cmap:
        #     ax.imshow(img_data, cmap=cmap, interpolation='nearest')
        # else:
        #     ax.imshow(img_data)

        # buf = io.BytesIO()
        # plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        # buf.seek(0)
        # plt.close(fig)
        
        # return send_file(buf, mimetype='image/png')

        # 2. USE THREAD-SAFE FIGURE
        # Do NOT use plt.figure() or plt.Axes() in web servers
        fig = Figure(figsize=(5, 5)) 
        canvas = FigureCanvas(fig)
        
        # Add axes filling the whole figure
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.set_axis_off()

        # 3. ROBUST NORMALIZATION (Optional but Recommended)
        # Calculate 2nd and 98th percentiles to ignore outliers
        # This prevents the map from turning solid black/white due to one bad pixel
        try:
            vmin, vmax = np.nanpercentile(img_data.filled(np.nan), [2, 98])
        except:
            vmin, vmax = None, None # Fallback if data is empty

        # 4. RENDER
        cmap = config.get('cmap', 'viridis')
        # vmin/vmax ensures the colors are stretched across the *real* data range
        ax.imshow(img_data, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)

        # 5. OUTPUT TO BUFFER
        output = io.BytesIO()
        # explicitly set transparent=False to ensure visibility on dark dashboards
        # or set facecolor='black' if you want it to blend in
        fig.savefig(output, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        output.seek(0)
        
        return send_file(output, mimetype='image/png')

    except Exception as e:
        print(f"Img Error {college_code} {layer_type}: {e}")
        abort(500)

@app.route('/get_legend/<layer_key>')
def get_legend(layer_key):
    config = LAYER_CONFIG.get(layer_key)
    if not config: return "Layer not found", 404
    
    # Setup Figure
    # Standard legend size: wide and short
    fig = plt.figure(figsize=(4, 0.8)) 
    ax = fig.add_axes([0.05, 0.5, 0.9, 0.3]) # [left, bottom, width, height]

    # CASE 1: LULC (Discrete Categories)
    if 'lulc' in layer_key:
        ax.set_axis_off()
        patches = []
        for label, color in LULC_CLASSES.items():
            patches.append(mpatches.Patch(color=color, label=label))
        
        # Place legend centered in the figure
        fig.legend(handles=patches, loc='center', ncol=len(patches), 
                   frameon=False, fontsize='small')

    # CASE 2: HEATMAPS (Continuous Colorbar)
    elif config.get('type') == 'heatmap':
        cmap_name = config['cmap']
        # Determine Range (vmin, vmax)
        # Use specific range if defined, else default -1 to 1
        vmin, vmax = LEGEND_RANGES.get(layer_key, LEGEND_RANGES['default'])
        
        norm = Normalize(vmin=vmin, vmax=vmax)
        cb = ColorbarBase(ax, cmap=plt.get_cmap(cmap_name),
                          norm=norm,
                          orientation='horizontal')
        
        cb.set_label(config['name'], fontsize=8)
        cb.ax.tick_params(labelsize=7)

    else:
        # Fallback for Satellite/Zoom images which don't need a legend
        ax.text(0.5, 0.5, "No Legend Required", ha='center', va='center')
        ax.set_axis_off()

    # Output to Buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    plt.close(fig) # Close to free memory
    
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)