## Scripts Folder Structure 

The scripts included need to be placed together with certain data to run properly.

The folder structure is shown below. Files that need their paths to be manually changed are indicated in the comments. 

## Folder Structure

scripts/  
├── Feature Extraction  
│   ├── DMDS_No_Unzip_Updated.py         # Feature extraction script (update paths inside)  
│   ├── newNpy/                          # Directory for new .npy files  
│   ├── malNpy/                          # Directory for malicious .npy files  
│   ├── benignNpy/                          # Directory for benign .npy files  
│   ├── Dataframe.csv                    # Metadata/labels CSV  
│   ├── SplitTTP.py                      # Script for splitting data into TTP categories  
│   └── Sort_NewNpy.py                   # Script for sorting .npy files  
├── Sorted NPY  
│   ├── SortedNpy/  
│       ├── [TA0001] Initial Access/     # .npy files for TA0001 (Initial Access)  
│       └── NO [TA0001] Initial Access/  # .npy files excluding TA0001  
├── Training Data Generation  
│   ├── Train_Data_Gen.py                # Script for generating training data (set paths inside)  
│   ├── Train_Data_Gen_Tactic2.py        # Generates data for Tactic 2 (set paths inside)  
│   └── Train_Data_Gen_Tactic3.py        # Generates data for Tactic 3 (set paths inside)  
├── Model Training  
│   ├── Main.py                          # Main entry script for model training  
│   ├── main_torch_detection.py          # Detection model training using PyTorch  
│   ├── main_torch_tactics.py            # Tactics-specific model training using PyTorch  
│   ├── model.py                         # Model architecture (non-PyTorch)  
│   ├── model_torch.py                   # Model architecture for PyTorch  
.
