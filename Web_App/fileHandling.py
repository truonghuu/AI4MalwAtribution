import os

# Check which stage of file processing is at
def checkFileStatus(filename):
    # filename, filetype = os.path.split(file.filename)

    # NPY array is ready and ready to predict
    if os.path.exists(f'files/npy/{filename}.npy'):
        return 3

    # Json report is ready. Yet to process into a numpy array
    elif os.path.exists(f'files/json/{filename}.json'):
        return 2
    
    # EXE file is found but yet to be sent through cuckoo sandbox
    # This is only a potential future feature but easier to just code it in for now. 
    elif os.path.exists(f'files/exe/{filename}.exe'):
        return 1

    # Attempted to predict a file that has yet to be uploaded. 
    else:
        return 0
    

# Save accordingly to type of file.
def saveFile(file):
    # Get type of file
    filetype = os.path.splitext(file.filename)[-1]

    if (filetype == '.exe'):
        file.save(f'files/exe/{file.filename}')
    elif (filetype == '.npy'):
        file.save(f'files/npy/{file.filename}')
    elif (filetype == '.json'):
        file.save(f'files/json/{file.filename}')
    else:
        file.save(f'files/unknown_type/{file.filename}')