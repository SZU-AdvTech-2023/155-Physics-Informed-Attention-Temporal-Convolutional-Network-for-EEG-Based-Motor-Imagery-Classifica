import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

def load_data_LOSO (data_path, subject): 

    
    X_train, y_train = [], []
    for sub in range (0,9):
        path = data_path+'s' + str(sub+1) + '/'
        
        X1, y1 = load_data(path, sub+1, True)
        X2, y2 = load_data(path, sub+1, False)
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
                   
        if (sub == subject):
            X_test = X
            y_test = y
        elif (X_train == []):
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    return X_train, y_train, X_test, y_test

def load_data(data_path, subject, training, all_trials = True):

	n_channels = 22
	n_tests = 6*48 	
	window_Length = 7*250 

	class_return = np.zeros(n_tests)
	data_return = np.zeros((n_tests, n_channels, window_Length))

	NO_valid_trial = 0
	if training:
		a = sio.loadmat(data_path+'A0'+str(subject)+'T.mat')
	else:
		a = sio.loadmat(data_path+'A0'+str(subject)+'E.mat')
	a_data = a['data']
	for ii in range(0,a_data.size):
		a_data1 = a_data[0,ii]
		a_data2= [a_data1[0,0]]
		a_data3= a_data2[0]
		a_X 		= a_data3[0]
		a_trial 	= a_data3[1]
		a_y 		= a_data3[2]
		a_artifacts = a_data3[5]

		for trial in range(0,a_trial.size):
 			if(a_artifacts[trial] != 0 and not all_trials):
 			    continue
 			data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
 			class_return[NO_valid_trial] = int(a_y[trial])
 			NO_valid_trial +=1


	return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]

def standardize_data(X_train, X_test, channels):
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test

#%%
def get_data(path, subject, LOSO = False, isStandard = True):
    fs = 250
    t1 = int(1.5*fs)
    t2 = int(6*fs)
    T = t2-t1

    if LOSO:
        X_train, y_train, X_test, y_test = load_data_LOSO(path, subject)
    else:
        path = path + 's1/'.format(subject + 1)
        X_train, y_train = load_data(path, subject+1, True)
        X_test, y_test = load_data(path, subject+1, False)

    N_tr, N_ch, _ = X_train.shape 
    X_train = X_train[:, :, t1:t2].reshape(N_tr, 1, N_ch, T)
    y_train_onehot = (y_train-1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)
    N_test, N_ch, _ = X_test.shape 
    X_test = X_test[:, :, t1:t2].reshape(N_test, 1, N_ch, T)
    y_test_onehot = (y_test-1).astype(int)
    y_test_onehot = to_categorical(y_test_onehot)	

    if (isStandard == True):
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot