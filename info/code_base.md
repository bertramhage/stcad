# Codebase Explanation: Anomaly Detection in AIS Data

This document provides a comprehensive explanation of the project's codebase, which is designed to perform anomaly detection on large-scale Automatic Identification System (AIS) data from vessels.

## 1. Project Overview

The primary goal of this project is to identify anomalous vessel behavior from raw AIS data. Given the massive volume of data, the project employs a distributed processing strategy (MapReduce) and uses advanced deep learning techniques to learn meaningful representations of vessel trajectories. These representations are then used in clustering algorithms to group similar trajectories, with the assumption that anomalous trajectories will form their own small clusters or be outliers.

The core workflow is as follows:
1.  **Download** raw AIS data.
2.  **Preprocess** the data on a large scale using a MapReduce pattern to handle the volume and create clean, sessionized vessel voyages.
3.  **Learn Trajectory Embeddings** by training a BERT-style transformer model on the preprocessed trajectories. This model learns to encode an entire variable-length trajectory into a single, fixed-size vector (embedding).
4.  **Cluster** these trajectory embeddings to find groups of similar behavior and identify anomalies.

## 2. End-to-End Pipeline (`code_implementation.ipynb`)

The `code_implementation.ipynb` notebook serves as the main entry point and a complete, executable demonstration of the entire project pipeline. It walks through every major step, from downloading a sample of the data to preprocessing it, training the sequence model, and finally, using the model's output for clustering. The code within this notebook is a slightly simplified version of the more modular and robust scripts found in the `src/` directory.

## 3. File-by-File Breakdown

### `src/preprocessing/` - The Data Processing Core

This directory contains all the scripts related to the large-scale data processing pipeline, which is implemented following a **MapReduce** pattern to ensure scalability.

#### `download.py`
*   **Purpose**: Downloads and unzips daily AIS data archives from the Danish Maritime Authority.
*   **Functionality**: It takes a date range and destination folder, constructs the appropriate URLs, and fetches the data. It includes error handling for missing data and network issues. The script can be run from the command line and logs its progress using the custom logger.

#### `csv2pkl.py` - (MapReduce: Split Phase 1)
*   **Purpose**: This is the first part of the "Split" phase. It converts raw, daily CSV files into a more efficient intermediate format.
*   **Functionality**:
    *   Uses the high-performance `polars` library to read large CSV files.
    *   Performs initial filtering based on a geographical bounding box (Denmark), vessel speed, and valid timestamps.
    *   Maps categorical data like "Ship type" and "Navigational status" to numerical values.
    *   Groups all AIS messages by the vessel's unique identifier (`MMSI`).
    *   Saves the data for each day as a `joblib`-compressed pickle file, where each file is a dictionary mapping `MMSI` to its corresponding trajectory data for that day.

#### `map_reduce.py` - (MapReduce: Map, Shuffle & Reduce)
This script orchestrates the main distributed processing logic.
*   **`map_and_shuffle()` (Map & Shuffle Phase)**:
    *   **Purpose**: To group all data for a single vessel in one place.
    *   **Functionality**: It reads the daily pickle files from the `csv2pkl` step. For each `MMSI` in a file, it takes the corresponding trajectory segment and moves it into a dedicated folder for that `MMSI` inside a temporary directory. After this step, the temporary directory contains one folder per `MMSI`, with each folder containing all of that vessel's data segments from all days.

*   **`reduce()` and `process_single_mmsi()` (Reduce Phase)**:
    *   **Purpose**: To take all the data for a single vessel, clean it, identify distinct voyages, and prepare it for the neural network. This step is performed in parallel for each vessel.
    *   **Functionality**: The `reduce` function uses a `multiprocessing.Pool` to spawn multiple worker processes. Each worker executes `process_single_mmsi` on a different vessel's data. This function loads all segments for one vessel, merges them, and then passes the complete track to `preprocess_mmsi_track()` for detailed processing.

#### `preprocessing.py` - The Heart of the Reduce Phase
*   **Purpose**: Contains the detailed, step-by-step logic for cleaning and sessionizing a single vessel's complete trajectory.
*   **Functionality (`preprocess_mmsi_track`)**:
    1.  **Voyage Splitting**: Breaks a continuous track into separate "voyages" if the time between messages is greater than 2 hours.
    2.  **Filtering**: Removes very short voyages (e.g., < 4 hours or < 20 messages).
    3.  **Outlier Removal**: Uses `utils.detectOutlier` to find and remove physically impossible points within a voyage (e.g., vessel appears to travel faster than `SPEED_MAX`).
    4.  **Interpolation/Sampling**: Down-samples each voyage to a constant **5-minute interval** using linear interpolation. This creates uniform sequences, which is a requirement for the sequence model.
    5.  **Re-Splitting**: Splits long voyages into segments of max 24 hours to create more training samples.
    6.  **Normalization**: Normalizes the four key features (`lat`, `lon`, `sog`, `cog`) to a [0, 1] range.
    7.  The final output is a set of clean, normalized trajectory segments, which are saved as individual pickle files.

#### `train_test_split.py`
*   **Purpose**: To split the final preprocessed data into training, validation, and test sets.
*   **Key Feature**: It performs an **MMSI-aware split**. This is critical to prevent data leakage. It ensures that all trajectories from a single vessel end up in the *same* dataset (e.g., all in 'train' or all in 'val'). This prevents the model from simply memorizing vessel-specific patterns.

#### `utils.py` (Preprocessing Utilities)
*   **Purpose**: Provides specialized, low-level functions for the `preprocessing.py` script.
*   **Functionality**:
    *   `detectOutlier` & `trackOutlier`: Implements an efficient algorithm to find outlier points in a trajectory based on impossible calculated speeds between points. It cleverly uses sparse matrices to remain memory-efficient.
    *   `interpolate`: Performs the linear interpolation of vessel position and dynamics at a specific timestamp, used for the 5-minute resampling.

---

### `src/sequence_modelling/` - Trajectory Embedding with BERT

This directory contains the code for the deep learning model that learns to represent trajectories as vectors.

#### `configs.py`
*   **Purpose**: To centralize and manage the configuration for the BERT model and the training process.
*   **Functionality**: Provides functions (`get_bert_config`, `get_training_args`) that return configuration objects from the Hugging Face `transformers` library, with sensible project-specific defaults (e.g., 4 transformer layers, hidden size of 256).

#### `models.py` - The AISBERT Model
*   **Purpose**: Defines the custom PyTorch model and data handling classes.
*   **`AISEmbeddings`**: A custom embedding layer. Instead of a discrete token vocabulary, it uses a `nn.Linear` layer to project the 4 continuous features (`lat`, `lon`, `sog`, `cog`) into the model's 256-dimensional hidden space. It also adds positional embeddings.
*   **`AISBERT`**: The main model. It's a BERT-style transformer that is trained on a **Masked Token Modeling (MTM)** task. It takes a sequence of trajectory points, and its goal is to predict the original values of points that have been randomly masked (zeroed out).
*   **`AISDatasetBERT`**: A PyTorch `Dataset` that adds special `[CLS]` and `[SEP]` tokens (as specific feature vectors) to the start and end of each trajectory. The final hidden state of the `[CLS]` token is used as the embedding for the entire sequence.
*   **`DataCollator`**: A helper class that creates batches of data for training. It handles **padding** (making all sequences in a batch the same length) and **masking** (randomly masking 15% of the tokens for the MTM task).

#### `train_bert.py`
*   **Purpose**: A script to run the training process for the `AISBERT` model.
*   **Functionality**: It uses the Hugging Face `Trainer` class, which automates the entire training and evaluation loop. It initializes the model, datasets, and data collator, and then calls `trainer.train()` to start training.

---

### `src/utils/` - General Utilities

This directory contains general-purpose helper scripts used across the project.

#### `datasets.py`
*   **Purpose**: Defines the base `AISDataset`, a standard PyTorch `Dataset` for loading the preprocessed trajectory files from disk.

#### `logging.py`
*   **Purpose**: Implements a robust logging system.
*   **Functionality**: The `CustomLogger` class wraps a standard terminal logger and the **Weights & Biases (`wandb`)** library. This allows for rich experiment tracking (metrics, configurations, model artifacts). It also cleverly detects if it's running on an HPC cluster and disables `tqdm` progress bars to keep log files clean.

#### `plot.py`
*   **Purpose**: Provides functions for visualizing trajectories.
*   **Functionality**: The `plot_trajectories` function can plot multiple tracks on a single map. It uses `contextily` to add a real-world map background and handles de-normalization and aspect ratio correction for accurate visualization.

---

### `hpc_jobs/`
*   **Purpose**: This directory contains shell scripts for running the various stages of the pipeline (downloading, preprocessing, training) on a High-Performance Computing (HPC) cluster that uses the SLURM scheduler. These scripts typically load the required environment, activate the virtual environment, and then run the Python scripts from `src/` with the appropriate command-line arguments.

## 4. How to Run

As described in the `README.md`, the primary way to reproduce the results is via the `code_implementation.ipynb` notebook.

1.  **Create and activate a virtual environment**
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install requirements**
    This project uses `uv` for fast package management.
    ```bash
    pip install uv
    uv sync
    ```

3.  **Run the notebook**: Open and run the cells in `code_implementation.ipynb`.
