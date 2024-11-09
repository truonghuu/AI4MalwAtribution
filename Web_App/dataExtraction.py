import DMDStest 
import numpy as np

# Pass in a example.json file located in files/json/.
# Will give a example.npy file located in files/npy/

# Will also clean and transpose the data so all ready for training
def feature_extract(filename):
    # dmds = DMDStest.DMDS('test.json', 'files/json/test.json', 'files/npy/test.npy', 1000, 40)
    print(filename)
    outfileName = filename.split('.')[0] + '.npy'
    print(outfileName)
    ab = DMDStest.DMDS(filename, f'files/json/{filename}', f'files/npy/{outfileName}', 1000, 40)
    if ab.parse() and ab.convert():
        ab.write()

    clean_npy(outfileName)
    reshape_npy(outfileName)

# Cleans the numpy array (changes all the NaN to 0)
def clean_npy(npy_name):
    array = np.load(f'files/npy/{npy_name}')
    array = np.nan_to_num(array)
    np.save(f'files/npy/{npy_name}', array)

# Takes a (258, 105) Array and Outputs a (1, 105, 258) array (suitable for our model)
def reshape_npy(npy_name):
    array = np.load(f'files/npy/{npy_name}')
    array_shape = np.shape(array)
    # Checking if array is already correctly shaped
    if len(array_shape) == 3:
        if array_shape[1] == 105 and array_shape[2] == 258:
            return 0
    # If array is 2d array
    elif len(array_shape) == 2:

        # If array is (258, 105), Reshape it to (1, 105, 258)
        if array_shape[0] == 258 and array_shape[1] == 105:
            array = np.transpose(array)
            array = np.expand_dims(array, axis=0)
        # If array shape is (105, 258), Reshape it to (1, 105, 258)
        elif array_shape[0] == 105 and array_shape[1] == 258:
            array = np.expand_dims(array, axis=0)
    else:
        raise f"Error is of inccorrect shape! Expected Shape:(any, 105, 258). Input Shape: {array_shape}"
    np.save(f'files/npy/{npy_name}', array)
