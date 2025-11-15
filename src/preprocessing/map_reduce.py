import os
from time import time
import joblib
from multiprocessing import Pool, cpu_count
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
import gc

from src.preprocessing.preprocessing import preprocess_mmsi_track
from src.utils.logging import CustomLogger

def map_and_shuffle(input_dir: str, temp_dir: str, logger: CustomLogger):
    """ Goes through all input files and re-sorts them by MMSI into a temporary directory. """
    
    # Input files from chunking step
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pkl")]
    
    logger.info(f"Starting map and shuffle phase on {len(input_files)} files...")

    for file_path in input_files:
        data_dict = joblib.load(file_path)
            
        for mmsi, track_segment in data_dict.items():
            
            # Create a directory for this specific MMSI
            mmsi_dir = os.path.join(temp_dir, str(mmsi))
            os.makedirs(mmsi_dir, exist_ok=True)
            
            # Save this segment into the MMSI's folder
            # We name it after the original file to avoid collisions
            segment_filename = os.path.basename(file_path)
            output_path = os.path.join(mmsi_dir, segment_filename)
            
            joblib.dump(track_segment, output_path, compress=3)
        
        del data_dict  # Free memory

def process_single_mmsi(mmsi_info):
    """ Process a single MMSI's track segments. """
    mmsi, mmsi_dir_path, final_dir = mmsi_info
    
    results = {}
    
    # Load all segments for this MMSI
    all_segments = []
    segment_files = [f for f in os.listdir(mmsi_dir_path) if f.endswith(".pkl") and not f.startswith("vessel_types_")]
    if not segment_files:
        return {"error": f"No segment files found for MMSI {mmsi}",
                "error_code": 0}
    for seg_file in segment_files:
            segment_path = os.path.join(mmsi_dir_path, seg_file)
            track_segment = joblib.load(segment_path)
            all_segments.append(track_segment)
    
    results['num_segments'] = len(all_segments)
    
    # Merge into one track
    try:
        full_track = np.concatenate(all_segments, axis=0)
        del all_segments  # Free memory
        gc.collect()
    except ValueError as e:
        return {"error": f"Error concatenating segments for MMSI {mmsi}: {str(e)}",
                "error_code": 1}

    # Run processing for single MMSI's track
    try:
        processed_data, preprocessing_results = preprocess_mmsi_track(full_track)
        results.update(preprocessing_results)
        del full_track, preprocessing_results  # Free memory
        gc.collect()
    except Exception as e:
        return {"error": f"Error processing track for MMSI {mmsi}: {str(e)}",
                "error_code": 2}
    
    # Save final result
    if processed_data:
        for k, traj in processed_data.items(): # Constitues a sample
            final_output_path = os.path.join(final_dir, f"{mmsi}_{k}_processed.pkl")
            data_item = {'mmsi': mmsi, 'traj': traj}
            joblib.dump(data_item, final_output_path, compress=3)
    
    del processed_data # Free memory
    gc.collect()
            
    return results
    
def reduce(final_dir: str,
           temp_dir: str,
           n_workers: int = None,
           chunk_size: int = 10,
           logger: CustomLogger = None):
    """
    Preprocess vessel trajectories by MMSI in parallel.
    """
    os.makedirs(final_dir, exist_ok=True)
    
    logger.info(f"Output directory for final results: {final_dir}")
    
    mmsi_folders = os.listdir(temp_dir)
    
    logger.info(f"Starting reduce phase on {len(mmsi_folders)} MMSI folders")
    
    # Prepare list of (mmsi, path, output_dir) tuples for parallel processing
    mmsi_tasks = []
    for mmsi in mmsi_folders:
        mmsi_dir_path = os.path.join(temp_dir, mmsi)
        if os.path.isdir(mmsi_dir_path):
            mmsi_tasks.append((mmsi, mmsi_dir_path, final_dir))
    
    results = defaultdict(int) # To count preprocessing statistics
    
    # Process in parallel using imap_unordered to avoid accumulating results in memory
    t0 = time()
    e0 = 0
    logging_interval = 1000
    with Pool(processes=n_workers, maxtasksperchild=min(1,1000//chunk_size)) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_mmsi, mmsi_tasks, chunksize=chunk_size), 1):
            if "error" in result:
                logger.warning(result["error"])
                results[f"error_code_{result['error_code']}"] += 1
            else:
                for key, value in result.items():
                    results[key] += value
            if i % logging_interval == 0:
                elapsed = time() - t0
                errors = sum([results[f"error_code_{code}"] for code in range(3)]) - e0
                logger.log_metrics({
                    'reduce_avg_time': elapsed / float(i),
                    'errors': errors,
                    'pct_done': i / len(mmsi_tasks) * 100
                    }, step=i//logging_interval)
                t0 = time()
                e0 = errors
    
    logger.log_summary(results)
        
def combine_vessel_types(input_dir: str, final_dir: str, logger: CustomLogger):
    """
    Combine vessel type information from all vessel type files into a single mapping.
    """
    vessel_type_map = dict()
    
    logger.info("Combining vessel type information from all chunks...")
    
    vessel_type_files = [f for f in os.listdir(os.path.join(input_dir, 'vessel_types')) if f.endswith(".pkl")]
    for vt_file in vessel_type_files:
        vt_path = os.path.join(input_dir, 'vessel_types', vt_file)
        vt_data = joblib.load(vt_path)
        vessel_type_map.update(vt_data)
    
    # Save combined vessel type map
    combined_vt_path = os.path.join(final_dir, "vessel_types.pkl")
    joblib.dump(vessel_type_map, combined_vt_path, compress=3)
    
    logger.info(f"Combined vessel type map saved to {combined_vt_path}")
        

if __name__ == "__main__":
    parser = ArgumentParser(description="Map-Reduce preprocessing of vessel trajectory data.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with chunked input files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to store the final preprocessed files.')
    parser.add_argument('--run_name', type=str, default=None, help='Name of the logging run.')
    args = parser.parse_args()
    
    temp_dir = os.path.join(args.output_dir, 'temp_map_reduce')
    num_workers = cpu_count() - 1
    
    logger = CustomLogger(project_name='Computational-Tools', group='map_reduce_preprocessing', run_name=args.run_name, use_wandb=True)
    logger.log_config({
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "temp_dir": temp_dir,
        "num_workers": num_workers
    })
    
    map_and_shuffle(input_dir=args.input_dir, temp_dir=temp_dir, logger=logger)
    
    # Reduce in parallel
    reduce(final_dir=args.output_dir, temp_dir=temp_dir, n_workers=num_workers, logger=logger)
    
    combine_vessel_types(input_dir=args.input_dir, final_dir=args.output_dir, logger=logger)
    
    logger.finish()