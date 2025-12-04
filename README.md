# STCAD: Scalable Trajectory Clustering and Anomaly Detection on Terabyte-Scale AIS Data (Codebase)

The project includes a [Code Implementation Notebook](code_implementation.ipynb) containing the complete coding pipeline for reproducing the results, albeit with a subset of data. The full code and used helper functions are contained in the `src/` folder. Furthermore, two pretrained models are provided in the `models/pretrained` folder.

## Setting up your environment
This project requires **Python 3.11 or newer**.

### Option 1: Using `uv` (Recommended)

(Prerequisites) **Install uv:** `pip install uv`

1.  **Navigate to the Project:**
    Open your terminal or command prompt and go to the root of the project.

    ```bash
    cd path/to/project
    ```

2.  **Create the Virtual Environment:**
    ```bash
    uv venv --python ">=3.11"
    ```

3.  **Activate the Environment:**
    | Operating System | Activation Command |
    | :--- | :--- |
    | **macOS/Linux** | `source .venv/bin/activate` |
    | **Windows (CMD)** | `.venv\Scripts\activate` |
    | **Windows (PowerShell)** | `.venv\Scripts\Activate.ps1` |

4.  **Install Dependencies:**
    ```bash
    uv sync
    ```

---

### Option 2: Using Conda (Fallback)

If you are unable to use `uv` and you have conda installed you may follow this.

1.  **Create and Activate Environment:**
    ```bash
    conda create --name stcad-project "python>=3.11"
    conda activate stcad-project
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```