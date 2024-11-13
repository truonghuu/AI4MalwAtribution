import os
import numpy as np
import sklearn.utils as sk


'''
TODO Change path accordingly. 


Generates a X_train_{tacticNumber} and Y_train{tacticNumber}
Generates a X_test_{tacticNumber} and Y_test_{tacticNumber}

Generates X_train_fold_{} and Y_train_fold_{} from 0-4 (5 folds total)

'''


POSITIVE_PATH = 'E:\\sortedNpy\\[TA0008] Lateral Movement'
NEGATIVE_PATH = 'E:\\sortedNpy\\NO [TA0008] Lateral Movement'
TACTIC = 'Lateral Movement'

# Get number of samples available for this tactics. 
# Mulptiply by 2 for space to add Non-True samples
print(len(os.listdir(POSITIVE_PATH)))
print(len(os.listdir(NEGATIVE_PATH)))

if len(os.listdir(POSITIVE_PATH)) < len(os.listdir(NEGATIVE_PATH)):
    totalSamples = len(os.listdir(POSITIVE_PATH))*2
    print("More negative samples")
else:
    totalSamples = len(os.listdir(NEGATIVE_PATH))*2
    print("More positive samples")

# 80/20 Train-Test split
training_batch_size = int(totalSamples * 0.8)
testing_batch_size = int(totalSamples*0.2)

# Round down to a multiple of 5
training_batch_size = training_batch_size - (training_batch_size % 5)
testing_batch_size = testing_batch_size - (testing_batch_size % 5)

print(training_batch_size)
print(testing_batch_size)

positive_samples = os.listdir(POSITIVE_PATH)
negative_samples = os.listdir(NEGATIVE_PATH)


x_train = np.zeros((training_batch_size, 105, 258))
y_train = np.zeros((training_batch_size,1))


x_test = np.zeros((testing_batch_size, 105, 258))
y_test = np.zeros((testing_batch_size,1))

positiveCount = 0
negativeCount = 0

# Add all the positive samples into the training array
for i in range(0, training_batch_size //2):    
    x_train[i] = np.transpose(np.load(POSITIVE_PATH + '\\' + positive_samples[positiveCount]))
    y_train[i] = '1'
    positiveCount+=1
# Add a matching number of negative samples into the array
for i in range (training_batch_size//2,training_batch_size):
    x_train[i] = np.transpose(np.load(NEGATIVE_PATH + '\\' + negative_samples[negativeCount]))
    y_train[i] = '0'
    negativeCount+=1
   

positiveCount = 0
negativeCount = 0
# Add the postive samples into the testing array
for i in range(0, testing_batch_size//2):
    x_test[i] = np.transpose(np.load(POSITIVE_PATH + '\\' + positive_samples[positiveCount]))
    y_test[i] = '1'
    positiveCount+=1
# Add a matching number of samples into the array
for i in range (testing_batch_size//2, testing_batch_size):
    x_test[i] = np.transpose(np.load(NEGATIVE_PATH + '\\' + negative_samples[negativeCount]))
    y_test[i] = '0'
    negativeCount+=1



# Cleans Data. All 'NaN' are set to 0.
x_train, y_train, x_test, y_test = np.nan_to_num(x_train), np.nan_to_num(y_train), np.nan_to_num(x_test), np.nan_to_num(y_test)

# Shuffles data so training folds will have random distribution.
x_train_shuffled, y_train_shuffled = sk.shuffle(x_train, y_train)
x_test_shuffled, y_test_shuffled = sk.shuffle(x_test, y_test)
print(x_train_shuffled.shape, y_train_shuffled.shape)
print(x_test_shuffled.shape, y_test_shuffled.shape)

# Save Train-Test data.
np.save(f'x_test_{TACTIC}.npy', x_test_shuffled),np.save(f'x_train_{TACTIC}.npy', x_train_shuffled)
np.save(f'y_test_{TACTIC}.npy', y_test_shuffled),np.save(f'y_train_{TACTIC}.npy', y_train_shuffled)


# Join back the Train-Test split into 1.
x_data = np.concatenate((x_train_shuffled, x_test_shuffled), axis=0)
y_data = np.concatenate((y_train_shuffled, y_test_shuffled), axis=0)

# Deleting train-test split arrays from memory to prevent waste of memory. 
del y_train_shuffled, y_test_shuffled , x_train_shuffled, x_test_shuffled

# Shuffles data so training folds will have random distribution.
x_data, y_data = sk.shuffle(x_data, y_data)

# Split the whole data into 5 folds
x_data_subarrays = np.split(x_data,5)
y_data_subarrays = np.split(y_data, 5)

# Save the 5 folds
for i in range(5):
    np.save(f'x_train_{TACTIC}_'+str(i), x_data_subarrays[i])
    np.save(f'y_train_{TACTIC}_'+str(i), y_data_subarrays[i])








    


