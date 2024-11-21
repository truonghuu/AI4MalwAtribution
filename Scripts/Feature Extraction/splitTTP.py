import os
import shutil
import numpy as np
import pandas as pd
import time
 # Path to base folder (with dataframe.csv inside and malicious npy FOLDER inside)
PATH = "E:\\"

DEFAULT_FOLDER = PATH + 'malNpy' # Name of Folder containing Malicious Npys
SAVE_FOLDER = 'Sorted Npy' # Name of Folder to contain all the different TTP categories

# DONT TOUCH THIS, BASED ON dataframe.csv
TTP_NAMES = ["[TA0001] Initial Access","[TA0002] Execution","[TA0003] Persistence","[TA0004] Privilege Escalation","[TA0005] Defense Evasion","[TA0006] Credential Access","[TA0007] Discovery","[TA0008] Lateral Movement","[TA0009] Collection","[TA0010] Exfiltration","[TA0011] Command and Control","[TA0034] Impact","[TA0040] Impact","[TA0043] Reconnaissance"]

# Read dataframe csv
df = pd.read_csv('dataframe.csv')

# Get list of npy files
npy_files = os.listdir(DEFAULT_FOLDER)

# Target Folder
if not os.path.exists(SAVE_FOLDER): 
    os.mkdir(SAVE_FOLDER)

# Function to search dataframe for filename entry and fetch labels
# Returns None if the entry has stated "NOT_FOUND" as 1
def get_ttp_labels(filename):
    row = df[df['File Name'] == filename]
    if not row.empty:
        if row.iloc[0,2] == 0:
            return row.iloc[0,3:].values
        else:
            return None
    else:
        return np.zeros(len(df.columns) - 3)
    
# Function to categorize/copy files by TTP found in dataframe.csv
def categorize_files_by_ttp(files, ttp_index):
    ttp_folder = os.path.join(SAVE_FOLDER, f'{TTP_NAMES[ttp_index]}')
    no_ttp_folder = os.path.join(SAVE_FOLDER, f'NO {TTP_NAMES[ttp_index]}')
    os.makedirs(ttp_folder, exist_ok=True)
    os.makedirs(no_ttp_folder, exist_ok=True)

    for file in files:
        file_name = os.path.splitext(file)[0]
        labels = get_ttp_labels(file_name)
        if labels is None:
            pass
        else:
            if labels[ttp_index] == 1:
                shutil.copy(os.path.join(DEFAULT_FOLDER, file), os.path.join(ttp_folder, file))
            else:
                shutil.copy(os.path.join(DEFAULT_FOLDER, file), os.path.join(no_ttp_folder, file))


start = time.time()

for ttp_index in range(14):
    categorize_files_by_ttp(npy_files, ttp_index)

end = time.time()
print(f"Time Elapsed: {end-start}s")