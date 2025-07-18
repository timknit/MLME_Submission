# MLME Submission
## Setup
    1. install packages from reqiurements
      

## Beat-The-Felix
    1. load data into the beat-the-felix-data folder
    2. navigate to the folder cd 03_Beat_the_Felix
    3. activate virtuel environment in the terminal if necessary
    4. Execute python main.py
    

## Data Preprocessing
    0. navigate to the folder 04_data_preprocessing
    1. raw data into the folder "Data"
    2. Run data_preporcessing.py (from main-path)
    3. data_laggen.py (from main-path)
    4. stored data in folder "narx_data"

## NARX
  Requires Data from Preprocessing pipeline in the respective narx_data folder.
  
  ### Training
    1. Run train.py
    2. Prediction and Parity plots found in plots folder
    3. Trained weights and hyperparameters are saved to models and best_params folder
  ### Tuning
    1. Run tune.py 
    2. Manually take params obtained from optuna and implement into train.py script
    3. Obtain tuned models

  ### CQR
    1. Run CQR_execute.py
    2. CQR Plots are found in plots folder
  
  ### Evaluation
    1. run evaluate.py
    2. obtain metrics as prints
