# Adapted from [CIA-Oceanix/GeoTrackNet](https://github.com/CIA-Oceanix/GeoTrackNet)
"""
A script to merge AIS messages into AIS tracks.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#sys.path.append("..")
#import utils
import pickle
import copy
import csv
from datetime import datetime
import time
from io import StringIO
from tqdm import tqdm as tqdm
import polars as pl
import argparse
from collections import Counter
from src.preprocessing.preprocessing import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, SPEED_MAX as SOG_MAX

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI, SHIPTYPE  = list(range(10))

CARGO_TANKER_ONLY = False

def map_nav_status_to_int(nav_status):
    map_dict = {
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
        "Reserved for future use": 14,
        "Unknown value": 15
    }
    return map_dict.get(nav_status, 15)

def convert_str_to_unix(time_str):
    return datetime.strptime(time_str, "%d/%m/%Y %H:%M:%S").timestamp()

def map_ship_type_to_int(ship_type):
    mapping_dict = SHIP_TYPE_MAP = {
    # --- Script-Critical Types ---
    "Cargo": 70,    # (Script checks for 70-79)
    "Tanker": 80,   # (Script checks for 80-89)
    "Fishing": 30,  # (Script checks for 30)

    # --- Other Standard Types ---
    "Passenger": 60,
    "HSC": 40,      # High-Speed Craft
    "Tug": 52,
    "Towing": 31,
    "Sailing": 37,
    "Pleasure": 36,
    "Pilot": 50,
    "SAR": 51,      # Search and Rescue
    "Diving": 33,
    "Dredging": 34,
    "Law enforcement": 56,
    "Military": 35,
    "Medical": 57,
    "Anti-pollution": 55,
    "Port tender": 58,
    "WIG": 20,      # Wing in Ground

    # --- Undefined / Other / Reserved ---
    "Other": 59,
    "Undefined": 0,
    "Spare 1": 0,
    "Spare 2": 0,
    "Reserved": 9,
    "Not party to conflict": 59, # Mapping to 'Other' as it has no standard code

    # A default value for any string not in the map
    "DEFAULT_OTHER": 0
    }
    return mapping_dict.get(ship_type, 0)

def save_vessel_types(m_msg, vessel_type_dir, t_date_str):
        """Extract and save vessel types for each MMSI."""
        VesselTypes = dict()
        l_mmsi = []
        
        for v_msg in tqdm(m_msg, desc="Mapping vessel types...", leave=False):
            try:
                mmsi_ = int(v_msg[MMSI])
                type_ = int(v_msg[SHIPTYPE])
                if mmsi_ not in l_mmsi:
                    VesselTypes[mmsi_] = [type_]
                    l_mmsi.append(mmsi_)
                else:
                    VesselTypes[mmsi_].append(type_)
            except:
                continue
        
        for mmsi_ in list(VesselTypes.keys()):
            VesselTypes[mmsi_] = [t for t in VesselTypes[mmsi_] if t != 0]
            
            if len(VesselTypes[mmsi_]) > 0:
                # Use most common vessel type
                counts = Counter(VesselTypes[mmsi_])
                max_count = max(counts.values())
                modes = [k for k, v in counts.items() if v == max_count]
                VesselTypes[mmsi_] = modes[-1]  # Choose the last one in case of tie
            else:
                VesselTypes[mmsi_] = 0
        
        # Save VesselTypes dict
        vessel_type_filename = os.path.join(vessel_type_dir, f"vessel_types_{t_date_str}.pkl")
        os.makedirs(vessel_type_dir, exist_ok=True)
        with open(vessel_type_filename, "wb") as f:
            pickle.dump(VesselTypes, f)
        tqdm.write(f"Total number of vessels' types for date {t_date_str}: {len(VesselTypes)}")
        
def filter_messages(m_msg, lat_min, lat_max, lon_min, lon_max, sog_max, t_min, t_max):
    """Filter AIS messages by location, speed, and time bounds."""
    m_msg = m_msg[m_msg[:,LAT]>=lat_min]
    m_msg = m_msg[m_msg[:,LAT]<=lat_max]
    m_msg = m_msg[m_msg[:,LON]>=lon_min]
    m_msg = m_msg[m_msg[:,LON]<=lon_max]
    # SOG
    m_msg = m_msg[m_msg[:,SOG]>=0]
    m_msg = m_msg[m_msg[:,SOG]<=sog_max]
    # COG
    m_msg = m_msg[m_msg[:,SOG]>=0]
    m_msg = m_msg[m_msg[:,COG]<=360]

    # TIME
    m_msg = m_msg[m_msg[:,TIMESTAMP]>=0]

    m_msg = m_msg[m_msg[:,TIMESTAMP]>=t_min]
    m_msg = m_msg[m_msg[:,TIMESTAMP]<=t_max]
    
    return m_msg

def process_single_csv(csv_filename,
                       input_dir,
                       output_dir,
                       vessel_type_dir,
                       lat_min,
                       lat_max,
                       lon_min,
                       lon_max,
                       sog_max,
                       test):
    
    t_date_str = '-'.join(csv_filename.split('.')[0].split('-')[1:4])
    t_min = time.mktime(time.strptime(t_date_str + ' 00:00:00', "%Y-%m-%d %H:%M:%S"))
    t_max = time.mktime(time.strptime(t_date_str + ' 23:59:59', "%Y-%m-%d %H:%M:%S"))
    
    l_l_msg = [] # list of AIS messages, each row is a message (list of AIS attributes)
    data_path = os.path.join(input_dir, csv_filename)
    
    BUFFER_SIZE = 8 * 1024 * 1024  # 8MB buffer
    
    with open(data_path,"r", buffering=BUFFER_SIZE) as f:
        tqdm.write(f"Reading {csv_filename} ...")
        csvReader = csv.reader(f)
        next(csvReader) # skip the legend row
        count = 1
        lf = pl.scan_csv(data_path)
        total_rows = lf.select(pl.len()).collect().item()
        for row in tqdm(csvReader, total=total_rows-1, leave=False):
            count += 1
            try:
                l_l_msg.append([float(row[3]), # Latitude
                                float(row[4]), # Longitude
                                float(row[7]), # SOG
                                float(row[8]), # COG
                                int(row[9]), # Heading
                                float(row[6]), # ROT
                                int(map_nav_status_to_int(row[5])), # Navigation status
                                int(convert_str_to_unix(row[0])), # Timestamp
                                int(float(row[2])), # MMSI
                                int(map_ship_type_to_int(row[13]))]) # Ship type
            except:
                continue
            
    m_msg = np.array(l_l_msg)

    if m_msg[0,TIMESTAMP] > 1767222000: 
        m_msg[:,TIMESTAMP] = m_msg[:,TIMESTAMP]/1000 # Convert to suitable timestamp format

    if vessel_type_dir is not None:
        save_vessel_types(m_msg, vessel_type_dir, t_date_str)


    #print(f"Total msgs for date {t_date_str}: ", len(m_msg))

    ## MERGING INTO DICT
    #======================================
    # Creating AIS tracks from the list of AIS messages.
    # Each AIS track is formatted by a dictionary.

    # Full set
    Vs = dict()
    for v_msg in tqdm(m_msg, desc="Convert to dicts of vessel's tracks...", leave=False):
        mmsi = int(v_msg[MMSI])
        if not (mmsi in list(Vs.keys())):
            Vs[mmsi] = np.empty((0,9))
        Vs[mmsi] = np.concatenate((Vs[mmsi], np.expand_dims(v_msg[:9],0)), axis = 0)
    for key in tqdm(list(Vs.keys())):
        #if cargo_tanker_only and (not key in l_cargo_tanker):
        #    del Vs_train[key]
        Vs[key] = np.array(sorted(Vs[key], key=lambda m_entry: m_entry[TIMESTAMP]))
            

## PICKLING
#======================================
#for filename, filedict in zip([pkl_filename_train,pkl_filename_valid,pkl_filename_test],
#                              [Vs,Vs_valid,Vs_test]
#                             ):
    output_filename = csv_filename.replace('csv', 'pkl') 
    #print("Writing to ", os.path.join(output_dir,output_filename),"...")
    with open(os.path.join(output_dir,output_filename),"wb") as f:
        pickle.dump(Vs,f)
    tqdm.write(f"Total number of tracks for date {t_date_str}: {len(Vs)}")
    
def csv2pkl_parallel(lon_min=LON_MIN, lon_max=LON_MAX,
            lat_min=LAT_MIN, lat_max=LAT_MAX,
            sog_max=SOG_MAX,
            input_dir="data/files/",
            output_dir="data/pickle_files",
            vessel_type_dir=None,
            cargo_tanker_only=CARGO_TANKER_ONLY,
            test=False):
    
    from multiprocessing import Pool, cpu_count
    
    l_csv_filename = [filename for filename in os.listdir(input_dir) if filename.endswith('.csv')]
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare task list
    tasks = [(csv_file, input_dir, output_dir, vessel_type_dir, 
            LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, SOG_MAX, test) 
            for csv_file in l_csv_filename]

    # Process in parallel
    n_workers = cpu_count() - 1  # Leave 1 core free
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.starmap(process_single_csv, tasks),
            total=len(tasks),
            desc="Processing CSV files"
        ))
        
    print("CSV to PKL conversion complete.")

def csv2pkl(lon_min=LON_MIN, lon_max=LON_MAX,
            lat_min=LAT_MIN, lat_max=LAT_MAX,
            sog_max=SOG_MAX,
            input_dir="data/files/",
            output_dir="data/pickle_files",
            vessel_type_dir=None,
            cargo_tanker_only=CARGO_TANKER_ONLY,
            test=False):
    
    l_csv_filename = [filename for filename in os.listdir(input_dir) if filename.endswith('.csv')]
    os.makedirs(output_dir, exist_ok=True)

    ## LOADING CSV FILES
    #======================================
    n_error = 0
    for csv_filename in tqdm(l_csv_filename, desc=f'Reading csvs'):
        
        t_date_str = '-'.join(csv_filename.split('.')[0].split('-')[1:4])
        t_min = time.mktime(time.strptime(t_date_str + ' 00:00:00', "%Y-%m-%d %H:%M:%S"))
        t_max = time.mktime(time.strptime(t_date_str + ' 23:59:59', "%Y-%m-%d %H:%M:%S"))
        
        l_l_msg = [] # list of AIS messages, each row is a message (list of AIS attributes)
        data_path = os.path.join(input_dir, csv_filename)
        with open(data_path,"r") as f:
            tqdm.write(f"Reading {csv_filename} ...")
            csvReader = csv.reader(f)
            next(csvReader) # skip the legend row
            count = 1
            lf = pl.scan_csv(data_path)
            total_rows = lf.select(pl.len()).collect().item()
            for row in tqdm(csvReader, total=total_rows-1, leave=False):
                count += 1
                if test and count > total_rows * 0.1:
                    break
                try:
                    l_l_msg.append([float(row[3]), # Latitude
                                    float(row[4]), # Longitude
                                    float(row[7]), # SOG
                                    float(row[8]), # COG
                                    int(row[9]), # Heading
                                    float(row[6]), # ROT
                                    int(map_nav_status_to_int(row[5])), # Navigation status
                                    int(convert_str_to_unix(row[0])), # Timestamp
                                    int(float(row[2])), # MMSI
                                    int(map_ship_type_to_int(row[13]))]) # Ship type
                except:
                    n_error += 1
                    continue
                
        m_msg = np.array(l_l_msg)

        if m_msg[0,TIMESTAMP] > 1767222000: 
            m_msg[:,TIMESTAMP] = m_msg[:,TIMESTAMP]/1000 # Convert to suitable timestamp format
            
        if vessel_type_dir is not None:
            VesselTypes = dict()
            l_mmsi = []
            n_error = 0
            for v_msg in tqdm(m_msg, desc="Mapping vessel types...", leave=False):
                try:
                    mmsi_ = int(v_msg[MMSI])
                    type_ = int(v_msg[SHIPTYPE])
                    if mmsi_ not in l_mmsi :
                        VesselTypes[mmsi_] = [type_]
                        l_mmsi.append(mmsi_)
                    else:
                        VesselTypes[mmsi_].append(type_)
                except:
                    n_error += 1
                    continue
            for mmsi_ in list(VesselTypes.keys()):
                VesselTypes[mmsi_] = [t for t in VesselTypes[mmsi_] if t != 0]
                
                if len(VesselTypes[mmsi_]) > 0:
                    # Use most common vessel type
                    counts = Counter(VesselTypes[mmsi_])
                    max_count = max(counts.values())
                    modes = [k for k, v in counts.items() if v == max_count]
                    VesselTypes[mmsi_] = modes[-1] # Choose the last one in case of tie
                else:
                    VesselTypes[mmsi_] = 0
                            
            # Save VesselTypes dict
            vessel_type_filename = os.path.join(vessel_type_dir, f"vessel_types_{t_date_str}.pkl")
            os.makedirs(vessel_type_dir, exist_ok=True)
            with open(vessel_type_filename,"wb") as f:
                pickle.dump(VesselTypes,f)
            tqdm.write(f"Total number of vessels' types for date {t_date_str}: {len(VesselTypes)}")

        ## LAT LON
        m_msg = m_msg[m_msg[:,LAT]>=lat_min]
        m_msg = m_msg[m_msg[:,LAT]<=lat_max]
        m_msg = m_msg[m_msg[:,LON]>=lon_min]
        m_msg = m_msg[m_msg[:,LON]<=lon_max]
        # SOG
        m_msg = m_msg[m_msg[:,SOG]>=0]
        m_msg = m_msg[m_msg[:,SOG]<=sog_max]
        # COG
        m_msg = m_msg[m_msg[:,SOG]>=0]
        m_msg = m_msg[m_msg[:,COG]<=360]

        # TIME
        m_msg = m_msg[m_msg[:,TIMESTAMP]>=0]

        m_msg = m_msg[m_msg[:,TIMESTAMP]>=t_min]
        m_msg = m_msg[m_msg[:,TIMESTAMP]<=t_max]

        #print(f"Total msgs for date {t_date_str}: ", len(m_msg))

        ## MERGING INTO DICT
        #======================================
        # Creating AIS tracks from the list of AIS messages.
        # Each AIS track is formatted by a dictionary.

        # Full set
        Vs = dict()
        for v_msg in tqdm(m_msg, desc="Convert to dicts of vessel's tracks...", leave=False):
            mmsi = int(v_msg[MMSI])
            if not (mmsi in list(Vs.keys())):
                Vs[mmsi] = np.empty((0,9))
            Vs[mmsi] = np.concatenate((Vs[mmsi], np.expand_dims(v_msg[:9],0)), axis = 0)
        for key in tqdm(list(Vs.keys())):
            #if cargo_tanker_only and (not key in l_cargo_tanker):
            #    del Vs_train[key]
            Vs[key] = np.array(sorted(Vs[key], key=lambda m_entry: m_entry[TIMESTAMP]))
                

    ## PICKLING
    #======================================
    #for filename, filedict in zip([pkl_filename_train,pkl_filename_valid,pkl_filename_test],
    #                              [Vs,Vs_valid,Vs_test]
    #                             ):
        output_filename = csv_filename.replace('csv', 'pkl') 
        #print("Writing to ", os.path.join(output_dir,output_filename),"...")
        with open(os.path.join(output_dir,output_filename),"wb") as f:
            pickle.dump(Vs,f)
        tqdm.write(f"Total number of tracks for date {t_date_str}: {len(Vs)}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV AIS data to PKL format.")
    parser.add_argument("--input_dir", type=str, default="data/files/", help="Directory containing input CSV files.")
    parser.add_argument("--output_dir", type=str, default="data/pickle_files", help="Directory to save output PKL files.")
    #parser.add_argument("--vessel_type_dir", type=str, default=None, help="Directory to save vessel type information.")
    parser.add_argument("--test", action='store_true', help="Run in test mode with limited data.")
    args = parser.parse_args()
    
    csv2pkl_parallel(input_dir=args.input_dir,
            output_dir=args.output_dir,
            vessel_type_dir=os.path.join(args.output_dir, 'vessel_types'),
            test=args.test)