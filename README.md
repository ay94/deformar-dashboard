# DeformAR Dashboard

This dashboard is part of our EMNLP 2025 System Demonstration submission.  
It provides an **interactive interface** for exploring model outputs, entity-level statistics, error patterns, and similarity matrices across fine-tuned variants.

Built using **Dash (Plotly)**, it allows dynamic navigation between qualitative and quantitative evaluation views, using cached experiment outputs.

---

## 🔁 Reproducibility Instructions

To run the dashboard with included demo data and outputs:

### Step 1: Download the `reproducability/` Folder

Download the full folder from the shared Google Drive link below:

🔗 [Download reproducability folder from Google Drive](https://drive.google.com/drive/folders/1uz06aJ9EuTRieOVpj3ynEv9DBRIBSEbs?usp=sharing)

This folder includes:

- `ANERCorp_CamelLab_arabertv02/` — raw corpus files
- `conll2003_bert/` — alternate dataset (if applicable)
- `ExperimentData/` — extracted outputs, model checkpoints, evaluation results
- `analysis-config.yaml` — config file already pointing to correct folder paths

---

### ⚙️ Step 2: Ensure the Config Path is Correct

In `main.py`, this line defines where the config file is loaded from:

```python
CONFIG_PATH = (Path(__file__).parents[1] / "reproducability" / "analysis-config.yaml").resolve()
```

> ✅ **If you move `analysis-config.yaml` somewhere else**, make sure to update this path in `main.py` accordingly.

---

### Step 3: Run the Dashboard

```bash
cd deformar-dashboard
python main.py
```

## 📊 Dashboard Structure

The `deformar-dashboard/` directory contains a modular Dash application. Key components include:

| Folder / File             | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `main.py`                 | Entry point — runs the dashboard                                           |
| `appSettings.py`          | Initializes Dash app, sets layout and global callbacks                     |
| `config/`                 | Configuration logic and YAML parsers (e.g., `dashboard-config.yaml`)       |
| `callbacks/`              | Callback functions for dynamic tab behavior                                |
| `layouts/`                | Tab layouts — includes layout managers and tab definitions                 |
| `managers/`               | Tab logic managers (e.g., plotting and tab data processing)                |
| `notebooks/`              | Development and exploratory notebooks (miscellaneous/experimental features)|
| `assets/`                 | Static assets (CSS, JS) for styling the Dash app                           |
| `cache-directory/`        | Used by Flask-Caching for variant-specific dashboard state                 |

---

## 🧠 Component Roles

### `main.py`
- Loads `analysis-config.yaml` to determine variant paths and config
- Starts the Dash server using `appSettings.start_app(...)`

```python
CONFIG_PATH = (Path(__file__).parents[1] / "analysis-config.yaml").resolve()
```

> 📌 Make sure `analysis-config.yaml` is placed **outside the dashboard folder**, at the root of the repo (or adjust this path accordingly).

---

### `appSettings.py`
- Creates the Dash app with Bootstrap styling
- Defines all tabs (`load`, `dataset`, `decision`, `instance`)
- Dynamically enables tabs once data is loaded
- Registers callbacks using `DataManager`

### `managers/` vs `layouts/`
- **Managers** handle **data processing + plotting logic**
  - Includes `plotting/` utilities and `tab_managers/`
- **Layouts** define **tab layout components**
  - Each tab layout uses its respective manager to fill in data

### `DataManager` and `DashboardData`
- Loads and caches fine-tuned model outputs and associated metadata
- Provides access to:
  - Token/entity evaluation results
  - Attention and centroid similarity matrices
  - Dataset loaders
  - Model paths and configurations

---

## 🚀 Running the Dashboard

```bash
cd deformar-dashboard
python main.py
```

- Make sure you've placed `analysis-config.yaml` at the expected path
- Ensure experiment output files exist and match the config structure

---

## 🧪 Development Notes

- The `notebooks/` folder contains exploratory analysis and **in-development features**.
- These may later become part of cross-component visualizations in the dashboard.

---
