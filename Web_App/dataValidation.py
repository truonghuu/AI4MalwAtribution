import numpy as np

def validateArrayShape(array):
    print(f"array shape: {np.shape(array)}")
    # if array shape is (105, 258)
    if len(array.shape)==3 and array.shape[1] ==  105 and array.shape[2] == 258:
        return 1
    else:
        return 0
        # raise ("Shape of error not fitted to model")


# Somewhat hack. Reshape from (258,105) -> (any, 105, 258)
def fixShape(array):
    array = np.transpose(array)
    array = np.expand_dims(array, axis=0)
    # b,f,a = array.shape
    # array = np.reshape(array, [-1, f*a])
    print(f"Shape of data in: {array.shape}")
    print(f"Array: {array}")
    # np.save(f'files/npy/{filename}',array)
    return array