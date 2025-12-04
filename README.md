# STCAD: Scalable Trajectory Clustering and Anomaly Detection on Terabyte-Scale AIS Data (Codebase)

## How to run
The project includes a [Code Implementation Notebook](code_implementation.ipynb) containing the complete coding pipeline for reproducing the results, albeit with a subset of data. The full code and used helper functions are contained in the `src/` folder.

To run the notebook:

1. **Create and activate a virtual environment**
```bash
python3.11 -m venv .venv
. .venv/bin/activate
```

2. **Install requirements** \
This project uses uv. uv can be installed with `pip install uv`.\
Run to install requirements
```bash
uv sync
```

3. Run notebook