import numpy as np
import os
import pickle
import time
import polars as pl
import argparse
from collections import defaultdict
import joblib
from src.utils.logging import CustomLogger
from src.preprocessing.preprocessing import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, SPEED_MAX as SOG_MAX

# Define column indices
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI, SHIPTYPE  = list(range(10))

SHIP_TYPE_MAP = {
    # Main types
    "Cargo": 70,
    "Tanker": 80,
    "Fishing": 30,

    # Other standard types ---
    "Passenger": 60,
    "HSC": 40,
    "Tug": 52,
    "Towing": 31,
    "Sailing": 37,
    "Pleasure": 36,
    "Pilot": 50,
    "SAR": 51,
    "Diving": 33,
    "Dredging": 34,
    "Law enforcement": 56,
    "Military": 35,
    "Medical": 57,
    "Anti-pollution": 55,
    "Port tender": 58,
    "WIG": 20,

    # Undefined / Other
    "Other": 59,
    "Undefined": 0,
    "Spare 1": 0,
    "Spare 2": 0,
    "Reserved": 9,
    "Not party to conflict": 59,
    }

def csv2pkl(lon_min=LON_MIN, lon_max=LON_MAX,
            lat_min=LAT_MIN, lat_max=LAT_MAX,
            sog_max=SOG_MAX,
            input_dir="data/files/",
            output_dir="data/pickle_files",
            logger: CustomLogger = None):
      
    l_csv_filename = [filename for filename in os.listdir(input_dir) if filename.endswith('.csv')]
    logger.info(f"Found {len(l_csv_filename)} CSV files in {input_dir}.")
    
    logger.log_config({
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "sog_max": sog_max,
        "input_dir": input_dir,
        "output_dir": output_dir,
    })
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {file_name: {"total_messages": 0, "filtered_messages": 0} for file_name in l_csv_filename}

    files_processed = 0
    messages_processed = 0
    unique_vessels = set()
    for csv_filename in logger.tqdm(l_csv_filename, desc=f'Reading csvs'):
        try:
            t_date_str = '-'.join(csv_filename.split('.')[0].split('-')[1:4])
            t_min = time.mktime(time.strptime(t_date_str + ' 00:00:00', "%Y-%m-%d %H:%M:%S"))
            t_max = time.mktime(time.strptime(t_date_str + ' 23:59:59', "%Y-%m-%d %H:%M:%S"))
            
            lf = pl.scan_csv(os.path.join(input_dir, csv_filename),
                            schema_overrides={
                                "# Timestamp": pl.Utf8,
                                "MMSI": pl.Int64,
                                "Latitude": pl.Float64,
                                "Longitude": pl.Float64,
                                "Navigational status": pl.Utf8,
                                "ROT": pl.Float64,
                                "SOG": pl.Float64,
                                "COG": pl.Float64,
                                "Heading": pl.Int64,
                                "Ship type": pl.Utf8
                            })
            total_messages = lf.select(pl.len()).collect()[0,0]
            messages_processed += total_messages
            results[csv_filename]["total_messages"] = total_messages

            lf = (
                lf.with_columns(
                    pl.col("# Timestamp").str.to_datetime("%d/%m/%Y %H:%M:%S").dt.epoch("s").alias("Timestamp"), # Convert to UNIX timestamp
                    pl.col("Navigational status").replace_strict( # Map navigational status to integers
                    {
                        "Under way using engine": 0,
                        "At anchor": 1,
                        "Not under command": 2,
                        "Restricted maneuverability": 3,
                        "Constrained by her draught": 4,
                        "Moored": 5,
                        "Aground": 6,
                        "Engaged in fishing": 7,
                        "Under way sailing": 8,
                        "Reserved for future amendment [HSC]": 9,
                        "Reserved for future amendment [WIG]": 10,
                        "Power-driven vessel towing astern": 11,
                        "Power-driven vessel pushing ahead or towing alongside": 12,
                        "Reserved for future use": 13,
                        "Unknown value": 15
                    },
                    default=15
                    ),
                )
                .filter(
                    (pl.col("Latitude") >= lat_min) & (pl.col("Latitude") <= lat_max) &
                    (pl.col("Longitude") >= lon_min) & (pl.col("Longitude") <= lon_max) &
                    (pl.col("SOG") >= 0) & (pl.col("SOG") <= sog_max) &
                    (pl.col("COG") >= 0) & (pl.col("COG") <= 360) &
                    (pl.col("Timestamp") >= t_min) & (pl.col("Timestamp") <= t_max)
                )
                .select( # Select only the 9 columns needed for the track + ship type
                    pl.col("Latitude"),        # 0
                    pl.col("Longitude"),        # 1
                    pl.col("SOG"),        # 2
                    pl.col("COG"),        # 3
                    pl.col("Heading"),    # 4
                    pl.col("ROT"),        # 5
                    pl.col("Navigational status"),  # 6
                    pl.col("Timestamp"),  # 7
                    pl.col("MMSI"),        # 8
                    pl.col("Ship type")    # 9
                )
            )
                    
            ### Vessel Type Mapping
            vessel_type_dir = os.path.join(output_dir, "vessel_types")
            os.makedirs(vessel_type_dir, exist_ok=True)
            
            global SHIP_TYPE_MAP
            
            vt_df = (
                lf.with_columns(
                    pl.col("Ship type").replace_strict(SHIP_TYPE_MAP, default=0)
                )
                .filter(pl.col("Ship type") != 0) # Ignore "Undefined"
                .group_by("MMSI")
                .agg(
                    # This gets the most frequent type, just like your Counter logic
                    pl.col("Ship type").mode().first().alias("VesselType") 
                )
                .collect()
            )
            
            unique_vessels.update(vt_df["MMSI"].to_list())
            
            # Convert Polars DF to the dict format you need
            VesselTypes = {row[0]: row[1] for row in vt_df.iter_rows()}
            
            # ... save VesselTypes to pickle ...
            vt_output_filename = csv_filename.replace('csv', 'pkl')
            with open(os.path.join(vessel_type_dir, vt_output_filename), "wb") as f:
                pickle.dump(VesselTypes, f)
                
            df = lf.drop("Ship type").collect() # Ship type column no longer needed
            results[csv_filename]["filtered_messages"] = df.height
            
            # Build tracks
            Vs_list = defaultdict(list)
            for row_tuple in logger.tqdm(df.iter_rows(named=False), total=len(df), desc="Building track lists..."):
                mmsi = row_tuple[MMSI] 
                Vs_list[mmsi].append(row_tuple)
                
            del df # Free memory
            
            Vs = {} # Final dictionary
            for mmsi, track_list in logger.tqdm(Vs_list.items(), desc="Sorting and converting to NumPy..."):
                track_list.sort(key=lambda x: x[TIMESTAMP])
                Vs[mmsi] = np.array(track_list, dtype=np.float64)

            del Vs_list # Free memory
            
            output_filename = csv_filename.replace('csv', 'pkl') 
            output_path = os.path.join(output_dir, output_filename)
            joblib.dump(Vs, output_path, compress=3)

            del Vs  # Free memory
            
            files_processed += 1
    
        except Exception as e:
            logger.warning(f"Error processing file {csv_filename}: {e}")
            
        logger.log_metrics({
            "messages_processed": messages_processed,
            "unique_vessels": len(unique_vessels)
        }, step=files_processed)
        
    logger.info("Conversion completed.")
    
    total_messages = sum(info["total_messages"] for info in results.values())
    total_filtered = sum(info["filtered_messages"] for info in results.values())
    logger.info(f"Total messages processed: {total_messages}")
    logger.info(f"Total messages after filtering: {total_filtered}")
    
    logger.log_summary({
        "total_files_processed": files_processed,
        "total_messages_processed": messages_processed,
        "total_unique_vessels": len(unique_vessels),
        "total_messages": total_messages,
        "total_filtered_messages": total_filtered
    })
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV AIS data to PKL format.")
    parser.add_argument("--input_dir", type=str, default="data/files/", help="Directory containing input CSV files.")
    parser.add_argument("--output_dir", type=str, default="data/pickle_files", help="Directory to save output PKL files.")
    parser.add_argument("--run_name", type=str, default=None, help="Name of the logging run.")
    args = parser.parse_args()
    logger = CustomLogger(project_name="Computational-Tools", group="csv2pkl_conversion", run_name=args.run_name, use_wandb=True)
    csv2pkl(input_dir=args.input_dir,
            output_dir=args.output_dir,
            logger=logger)