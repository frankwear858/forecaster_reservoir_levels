Reservoirs Map Forecasting Model
=================================

This project provides a full-stack solution for forecasting reservoir levels using
a combination of scikit-learn linear regression and PyTorch-based CNN models, with
a Django-based map dashboard for visualization.

Contact: Frank Wear - frankcwear@gmail.com

-------------------------------------------------------------------------------
System Requirements
-------------------------------------------------------------------------------
- Python 3.10
- PostgreSQL (version 14 recommended) with a database instance created
- CUDA 12.3.1 (optional, for GPU acceleration)
  Download: https://developer.nvidia.com/cuda-12-3-1-download-archive

- PyTorch:
  * For GPU: use "torch==2.4.0+cu118"
  * For CPU: replace the torch line in requirements.txt and run:
    pip install torch

Note: The project was initially tested on CPU with PyTorch 2.1.1 before moving to GPU.

-------------------------------------------------------------------------------
How It Works
-------------------------------------------------------------------------------
The pipeline trains two models:
1. scikit-learn linear regression
2. PyTorch CNN model (with hyperparameter tuning)

CNN training on CPU will take time due to parameter sweeps. Models are already saved
in the "models/" directory if you'd prefer not to retrain.

-------------------------------------------------------------------------------
Steps to Run the Project
-------------------------------------------------------------------------------

Step 0: Configure PostgreSQL
----------------------------
Edit ./settings.py and update your PostgreSQL username and password.

Step 1: Ingest CSVs and Train Models
------------------------------------
1a] python ingest_csvs.py
1b] python models.py

Step 2: Launch the Django Dashboard
-----------------------------------
2a] python manage.py makemigrations
2b] python manage.py migrate
2c] python manage.py runserver

Visit http://127.0.0.1:8000/

Click any reservoir marker and then "View Predictions" to see Task 2.2.
Click the blue "View Reservoir Predictions" button at the top left for Task 2.4
to view 7-day forecasts for each site.

-------------------------------------------------------------------------------
Model Performance Summary
-------------------------------------------------------------------------------

Linear Regression (Best: month + day of month features)
-------------------------------------------------------
Lake Meredith    - MAE: 160.6   RMSE: 303.9
Joe Pool Lake    - MAE: 918.4   RMSE: 10,768.5
Lake Conroe      - MAE: 2004.9  RMSE: 18,296.3
Lake Georgetown  - MAE: 49.8    RMSE: 70.2

Used features: 14-day lags, month/day, rolling mean/std of cs, precip, avg_temp

CNN Model Results (Autotuned)
-----------------------------

Without Metadata
----------------
Lake Meredith    - LR: 0.0005  Epochs: 500  MAE: 177.99  RMSE: 291.27
Joe Pool Lake    - LR: 0.1     Epochs: 500  MAE: 1112.06 RMSE: 10,934.0
Lake Conroe      - LR: 0.002   Epochs: 500  MAE: 1787.37 RMSE: 17,847.8
Lake Georgetown  - LR: 0.002   Epochs: 500  MAE: 50.16   RMSE: 68.26

With Metadata
-------------
Lake Meredith    - LR: 0.02    Epochs: 100  MAE: 249.59  RMSE: 401.02
Joe Pool Lake    - LR: 0.0025  Epochs: 500  MAE: 1109.48 RMSE: 10,867.5
Lake Conroe      - LR: 0.2     Epochs: 200  MAE: 1611.96 RMSE: 18,048.9
Lake Georgetown  - LR: 0.02    Epochs: 200  MAE: 50.92   RMSE: 70.09

-------------------------------------------------------------------------------
Additional Notes
-------------------------------------------------------------------------------
- All trained models are saved in the models/ directory.
- You can generate predictions using:
    models.make_predictions(reservoir_name, model_type, date)

- Scripts use `##` to denote PyCharm-style executable cells
  (requires PyCharm cell plugin to run them individually).

-------------------------------------------------------------------------------
Hardware and Training Environment
-------------------------------------------------------------------------------
- PyTorch Version: 2.4.0+cu118
- CUDA Available: True
- CUDA Version: 11.8
- GPU: NVIDIA GeForce RTX 3060
