import os
import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score

import models
from preprocess import get_data


def draw_learning_curves(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.close()

def draw_confusion_matrix(cf_matrix, sub, results_path):
    display_labels = ['Left hand', 'Right hand','Foot','Tongue']
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, 
                                display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    plt.title('Confusion Matrix of Subject: ' + sub )
    plt.savefig(results_path + '/subject_' + sub + '.png')
    plt.show()

def draw_performance_barChart(num_sub, metric, label):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub+1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model '+ label + ' per subject')
    ax.set_ylim([0,1])
    
    
# Training
def train(dataset_conf, train_conf, results_path):

    in_exp = time.time()
    best_models = open(results_path + "/best models.txt", "w")
    log_write = open(results_path + "/log.txt", "w")
    perf_allRuns = open(results_path + "/perf_allRuns.npz", 'wb')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')

    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))
    for sub in range(n_sub):
        in_sub = time.time()
        print('\nTraining on subject ', sub+1)
        log_write.write( '\nTraining on subject '+ str(sub+1) +'\n')
        BestSubjAcc = 0 
        bestTrainingHistory = []
        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(
            data_path, sub, LOSO, isStandard)

        for train in range(n_train):
            in_run = time.time()
            filepath = results_path + '/saved models/run-{}'.format(train+1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)        
            filepath = filepath + '/subject-{}.h5'.format(sub+1)

            model = getModel(model_name)
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])          
            callbacks = [
                ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, 
                                save_best_only=True, save_weights_only=True, mode='max'),
                EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
            ]
            history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), 
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

            model.load_weights(filepath)
            y_pred = model.predict(X_test).argmax(axis=-1)
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, train]  = accuracy_score(labels, y_pred)
            kappa[sub, train] = cohen_kappa_score(labels, y_pred)
            out_run = time.time()
            info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub+1, train+1, ((out_run-in_run)/60))
            info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}'.format(acc[sub, train], kappa[sub, train])
            print(info)
            log_write.write(info +'\n')
            if(BestSubjAcc < acc[sub, train]):
                 BestSubjAcc = acc[sub, train]
                 bestTrainingHistory = history

        best_run = np.argmax(acc[sub,:])
        filepath = '/saved models/run-{}/subject-{}.h5'.format(best_run+1, sub+1)+'\n'
        best_models.write(filepath)
        out_sub = time.time()
        info = '----------\n'
        info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub+1, best_run+1, ((out_sub-in_sub)/60))
        info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run], np.average(acc[sub, :]), acc[sub,:].std() )
        info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(kappa[sub, best_run], np.average(kappa[sub, :]), kappa[sub,:].std())
        info = info + '\n----------'
        print(info)
        log_write.write(info+'\n')
        if (LearnCurves == True):
            print('Plot Learning Curves ....... ')
            draw_learning_curves(bestTrainingHistory)

    out_exp = time.time()
    info = '\nTime: {:.1f} h   '.format( (out_exp-in_exp)/(60*60) )
    print(info)
    log_write.write(info+'\n')
    np.savez(perf_allRuns, acc = acc, kappa = kappa)
    best_models.close()   
    log_write.close() 
    perf_allRuns.close() 


# Evaluation
def test(model, dataset_conf, results_path, allRuns = True):
    log_write = open(results_path + "/log.txt", "a")
    best_models = open(results_path + "/best models.txt", "r")
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')

    acc_bestRun = np.zeros(n_sub)
    kappa_bestRun = np.zeros(n_sub)  
    cf_matrix = np.zeros([n_sub, n_classes, n_classes])

    if(allRuns):
        perf_allRuns = open(results_path + "/perf_allRuns.npz", 'rb')
        perf_arrays = np.load(perf_allRuns)
        acc_allRuns = perf_arrays['acc']
        kappa_allRuns = perf_arrays['kappa']
    for sub in range(n_sub):
        _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub, LOSO, isStandard)
        filepath = best_models.readline()
        model.load_weights(results_path + filepath[:-1])
        y_pred = model.predict(X_test).argmax(axis=-1)
        labels = y_test_onehot.argmax(axis=-1)
        acc_bestRun[sub] = accuracy_score(labels, y_pred)
        kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)
        cf_matrix[sub, :, :] = confusion_matrix(labels, y_pred, normalize='pred')
        draw_confusion_matrix(cf_matrix[sub, :, :], str(sub+1), results_path)

        info = 'Subject: {}   best_run: {:2}  '.format(sub+1, (filepath[filepath.find('run-')+4:filepath.find('/sub')]) )
        info = info + 'acc: {:.4f}   kappa: {:.4f}   '.format(acc_bestRun[sub], kappa_bestRun[sub] )
        if(allRuns): 
            info = info + 'avg_acc: {:.4f} +- {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(
                np.average(acc_allRuns[sub, :]), acc_allRuns[sub,:].std(),
                np.average(kappa_allRuns[sub, :]), kappa_allRuns[sub,:].std() )
        print(info)
        log_write.write('\n'+info)

    info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.4f}   Kappa = {:.4f}\n'.format(
        n_sub, np.average(acc_bestRun), np.average(kappa_bestRun)) 
    if(allRuns): 
        info = info + '\nAverage of {} subjects x {} runs (average of {} experiments):\nAccuracy = {:.4f}   Kappa = {:.4f}'.format(
            n_sub, acc_allRuns.shape[1], (n_sub * acc_allRuns.shape[1]),
            np.average(acc_allRuns), np.average(kappa_allRuns)) 
    print(info)
    log_write.write(info)

    draw_performance_barChart(n_sub, acc_bestRun, 'Accuracy')
    draw_performance_barChart(n_sub, kappa_bestRun, 'K-score')
    draw_confusion_matrix(cf_matrix.mean(0), 'All', results_path)
    log_write.close() 

def getModel(model_name):
    if(model_name == 'ATCNet'):
        model = models.ATCNet(
            n_classes = 4, 
            in_chans = 22, 
            in_samples = 1125,
            n_windows = 5,
            attention = 'mha',
            eegn_F1 = 16,
            eegn_D = 2, 
            eegn_kernelSize = 64,
            eegn_poolSize = 7,
            eegn_dropout = 0.3,
            tcn_depth = 2, 
            tcn_kernelSize = 4,
            tcn_filters = 32,
            tcn_dropout = 0.3, 
            tcn_activation='elu'
            )     
    elif(model_name == 'TCNet_Fusion'):
        model = models.TCNet_Fusion(n_classes = 4)      
    elif(model_name == 'EEGTCNet'):
        model = models.EEGTCNet(n_classes = 4)          
    elif(model_name == 'EEGNet'):
        model = models.EEGNet_classifier(n_classes = 4) 
    elif(model_name == 'EEGNeX'):
        model = models.EEGNeX_8_32(n_timesteps = 1125 , n_features = 22, n_outputs = 4)
    elif(model_name == 'DeepConvNet'):
        model = models.DeepConvNet(nb_classes = 4 , Chans = 22, Samples = 1125)
    elif(model_name == 'ShallowConvNet'):
        model = models.ShallowConvNet(nb_classes = 4 , Chans = 22, Samples = 1125)
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model
    
    
#%%
def run():
    data_path ='D:\EEG-ATCNet-main\BCI Competition IV-2a'
    results_path = os.getcwd() + "/results"
    if not  os.path.exists(results_path):
      os.makedirs(results_path)

    dataset_conf = { 'n_classes': 4, 'n_sub': 9, 'n_channels': 22, 'data_path': data_path,
                'isStandard': True, 'LOSO': False}
    train_conf = { 'batch_size': 64, 'epochs': 1000, 'patience': 300, 'lr': 0.0009,
                  'LearnCurves': True, 'n_train': 10, 'model':'ATCNet'}
    model = getModel(train_conf.get('model'))
    test(model, dataset_conf, results_path)    

if __name__ == "__main__":
    run()
    