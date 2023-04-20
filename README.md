# PANCDR

PANCDR is precision medicine prediction using an adversary networks for cancer drug response. PANCDR consists of two steps: training the discriminator and training the CDR prediction model. The discriminator reduces the gap between the two datasets while the CDR prediction model extracts features and predicts the response.

Here is the [paper](http://www.pancdrpaper.com).

<img src="https://user-images.githubusercontent.com/44110710/232651050-5d768fde-7de7-47fd-bf80-685173e1bf44.png" width="60%" height="60%"/>

## Requirements

python==3.6.10  
pytorch==1.10.1  
hickle==3.4.5  
pandas==1.0.1

First, create a virtual environment and install the requirements.

    conda create -n [ENVIRONMENT NAME] python==3.6.10
    conda activate [ENVIRONMENT NAME]
    pip install -r requirements.txt
    

## Data descriptions

- GDSC
  - `drug_graph_feat/` - The graph features from GDSC drug data.
  - `Cell_Lines_Details.txt` - Cell line annotations
  - `GDSC_IC50_response_357.csv` - Continuous drug response data (IC50)
  - `GDSC_binary_response_151.csv` - Binary drug response data (resistant/sensitive)
  - `GDSC_drug_IC50.csv` - Drug name and PubChem ID of continuous data
  - `GDSC_drug_binary.csv` - Drug name and PubChem ID of binary data
  - `GDSC_expr_z_702.csv` - z-normalized gene expressions with 702 cancer gene census

- TCGA
  - `drug_graph_feat/` - The graph features from GDSC drug data.
  - `Pretrain_TCGA_expr_702_01A.zip` - Gene expression data without annotation for pretraining
  - `TCGA_drug_new.csv` - Drug name and PubChem ID
  - `TCGA_expr_z_702.csv` - z-normalized gene expressions with 702 cancer gene census
  - `TCGA_response_new.csv` - Binary drug response data
  - `TCGA_type_new.txt` - Barcode and cancer type of TCGA patients

## Model training
Run files in `src/` directory to train the model  
Here is the example:

    python run_PANCDR.py

- `run_PANCDR.py` - Train PANCDR 100 times with optimal hyperparameters.
- `run_PANCDR_nested.py` - Train PANCDR using 10-fold outer cross-validation with optimal hyperparameters for each fold.
- `run_PANCDR_regr.py` - Train regression model of PANCDR with optimal hyperparameters
- `run_PANCDR_regr_nested.py` - Train regression model of PANCDR using 10-fold outer cross-validation with optimal hyperparameters for each fold.
- `run_baseline.py` - Train DeepCDR 100 times with optimal hyperparameters.
- `run_baseline_nested.py` - Train DeepCDR using 10-fold outer cross-validation with optimal hyperparameters for each fold.
