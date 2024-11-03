import os

TXT = "sa_mal_samples_43967.txt" # SAMPLE LIST FILE
TARGET = "newNpy" # DIRECTORY WITH ALL SAMPLES
OUTPUT = "malNpy" # DIRECTORY TO PUT MALICIOUS SAMPLES

# List of all the files in target folder
files = os.listdir(TARGET)
# Remove file extension
files = [os.path.splitext(file)[0] for file in files]

# Check is malicious npy directory created
if not os.path.isdir(OUTPUT):
    os.mkdir(OUTPUT)
    print("dir created")
duplicateCount = 0
with open(TXT,'r') as mfile:
    mtext = mfile.readlines()
    mtext = [text.split('\t')[1] for text in mtext]
    print(mtext[0])
    for file in files:
        extra = None
        if "-" in file:
            file, extra = file.split("-")
        if any(file in entry for entry in mtext):
            if extra:
                try:
                    os.rename(f"{TARGET}/{file}-{extra}.npy", f"{OUTPUT}/{file}.npy")
                except FileExistsError as e:
                    duplicateCount+=1
                    print(e)
                    os.remove(f"{TARGET}/{file}-{extra}.npy")
                except Exception as e:
                    print("That file has not been parsed yet.")
            else:
                try:
                    os.rename(f"{TARGET}/{file}.npy", f"{OUTPUT}/{file}.npy")
                except FileExistsError as e:
                    duplicateCount+=1
                    print(e)
                    os.remove(f"{TARGET}/{file}.npy")
            print("moved")

print(duplicateCount)