import os
import random
import argparse

def move_to_dir(filename, src_dir, dest_dir):
    """ Moves a file from src_dir to dest_dir. """
    src_path = os.path.join(src_dir, filename)
    dest_path = os.path.join(dest_dir, filename)
    os.rename(src_path, dest_path)

def train_test_split_tracks(data_dir, val_size=0.2, test_size=0.2, random_state=42):
    """
    Splits the dataset into training, validation, and test sets. Saves into tree subdirectories: train, val, test.
    
    Parameters:
    - data_dir: Directory containing the preprocessed data files.
    - val_size: Proportion of the dataset to include in the validation set.
    - test_size: Proportion of the dataset to include in the test set.
    - random_state: Seed used by the random number generator.
    
    """
    
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    if not all_files:
        print("No data files found in the specified directory.")
        return
    
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test'), exist_ok=True)
    
    total_n = len(all_files)
    test_n = int(total_n * test_size)
    val_n = int(total_n * val_size)
    train_n = total_n - test_n - val_n
    
    random.seed(random_state)
    
    # Make a MMSI-aware split to avoid data leakage
    unassigned_files = all_files.copy()
    for partition_size, partition in zip([train_n, val_n, test_n],
                                         ['train', 'val', 'test']):
        running_count = 0
        while running_count < partition_size and unassigned_files:
            file = unassigned_files.pop(random.randint(0, len(unassigned_files)-1))
            move_to_dir(file, data_dir, os.path.join(data_dir, partition))
            running_count += 1
            mmsi = file.split('_')[0]
            additional_files_to_add = [f for f in unassigned_files if f.startswith(mmsi+'_')]
            for af in additional_files_to_add:
                move_to_dir(af, data_dir, os.path.join(data_dir, partition))
                running_count += 1
                unassigned_files.remove(af)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split preprocessed data into train, val, and test sets.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing preprocessed data files.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of data for validation set.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of data for test set.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    train_test_split_tracks(data_dir=args.data_dir,
                            val_size=args.val_size,
                            test_size=args.test_size,
                            random_state=args.random_state)