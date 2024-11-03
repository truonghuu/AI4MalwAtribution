import numpy as np
import logging
import os, time
import tensorflow as tf
import model as model_tf
# from tensorflow import keras 
from tensorflow.keras.layers import Normalization
# from tensorflow.python.keras.layers import Normalization
# from tensorflow.python.keras.optimizers import RMSprop, Adamax, AdamW, Adam, SGD


from tensorflow.keras.layers import Normalization
from tensorflow.keras.optimizers import RMSprop, Adamax, AdamW, Adam, SGD
from datetime import datetime
import argparse
import json
from tensorflow.keras.layers import Normalization
# from tensorflow.python.keras.layers import Normalization
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate


# THIS TACTIC NEEDS TO BE CHANGED BEFORE RUNNING EVERYTIME. 
TACTIC = "Defense Evasion"

#logging.getLogger('matplotlib.font_manager').disabled = True
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")

def load_data(fold, args):
    logging.info(f'fold: {fold}')
    #Work starts here
    if fold == -1:
        X_test, y_test = np.load(f'./x_test_{TACTIC}.npy'.format(fold)), np.load(f'./y_test_{TACTIC}.npy')
        X_train, y_train = np.load(f'./x_train_{TACTIC}.npy'.format(fold)), np.load(f'./y_train_{TACTIC}.npy')
    
    if fold == 0: 
        X_test, y_test = np.load(f'./x_train_{TACTIC}_{fold}.npy'), np.load(f'./y_train_{TACTIC}_{fold}.npy')
        a,a2 =  np.load(f'./x_train_{TACTIC}_{1}.npy'), np.load(f'./y_train_{TACTIC}_{1}.npy')
        b,b2 =  np.load(f'./x_train_{TACTIC}_{2}.npy'), np.load(f'./y_train_{TACTIC}_{2}.npy')
        c,c2 =  np.load(f'./x_train_{TACTIC}_{3}.npy'), np.load(f'./y_train_{TACTIC}_{3}.npy')
        d,d2 =  np.load(f'./x_train_{TACTIC}_{4}.npy'), np.load(f'./y_train_{TACTIC}_{4}.npy')
        X_train = np.concatenate((a,b,c,d),axis=0)
        y_train = np.concatenate((a2,b2,c2,d2),axis=0)
    if fold == 1:
        X_test, y_test = np.load(f'./x_train_{TACTIC}_{fold}.npy'), np.load(f'./y_train_{TACTIC}_{fold}.npy')
        a,a2 =  np.load(f'./x_train_{TACTIC}_{0}.npy'), np.load(f'./y_train_{TACTIC}_{0}.npy')
        b,b2 =  np.load(f'./x_train_{TACTIC}_{2}.npy'), np.load(f'./y_train_{TACTIC}_{2}.npy')
        c,c2 =  np.load(f'./x_train_{TACTIC}_{3}.npy'), np.load(f'./y_train_{TACTIC}_{3}.npy')
        d,d2 =  np.load(f'./x_train_{TACTIC}_{4}.npy'), np.load(f'./y_train_{TACTIC}_{4}.npy')
        X_train = np.concatenate((a,b,c,d),axis=0)
        y_train = np.concatenate((a2,b2,c2,d2),axis=0)
    if fold == 2:
        X_test, y_test = np.load(f'./x_train_{TACTIC}_{fold}.npy'), np.load(f'./y_train_{TACTIC}_{fold}.npy')
        a,a2 =  np.load(f'./x_train_{TACTIC}_{1}.npy'), np.load(f'./y_train_{TACTIC}_{1}.npy')
        b,b2 =  np.load(f'./x_train_{TACTIC}_{0}.npy'), np.load(f'./y_train_{TACTIC}_{0}.npy')
        c,c2 =  np.load(f'./x_train_{TACTIC}_{3}.npy'), np.load(f'./y_train_{TACTIC}_{3}.npy')
        d,d2 =  np.load(f'./x_train_{TACTIC}_{4}.npy'), np.load(f'./y_train_{TACTIC}_{4}.npy')
        X_train = np.concatenate((a,b,c,d),axis=0)
        y_train = np.concatenate((a2,b2,c2,d2),axis=0)
    if fold == 3:
        X_test, y_test = np.load(f'./x_train_{TACTIC}_{fold}.npy'), np.load(f'./y_train_{TACTIC}_{fold}.npy')
        a,a2 =  np.load(f'./x_train_{TACTIC}_{1}.npy'), np.load(f'./y_train_{TACTIC}_{1}.npy')
        b,b2 =  np.load(f'./x_train_{TACTIC}_{2}.npy'), np.load(f'./y_train_{TACTIC}_{2}.npy')
        c,c2 =  np.load(f'./x_train_{TACTIC}_{0}.npy'), np.load(f'./y_train_{TACTIC}_{0}.npy')
        d,d2 =  np.load(f'./x_train_{TACTIC}_{4}.npy'), np.load(f'./y_train_{TACTIC}_{4}.npy')
        X_train = np.concatenate((a,b,c,d),axis=0)
        y_train = np.concatenate((a2,b2,c2,d2),axis=0)
    if fold == 4:
        X_test, y_test = np.load(f'./x_train_{TACTIC}_{fold}.npy'), np.load(f'./y_train_{TACTIC}_{fold}.npy')
        a,a2 =  np.load(f'./x_train_{TACTIC}_{1}.npy'), np.load(f'./y_train_{TACTIC}_{1}.npy')
        b,b2 =  np.load(f'./x_train_{TACTIC}_{2}.npy'), np.load(f'./y_train_{TACTIC}_{2}.npy')
        c,c2 =  np.load(f'./x_train_{TACTIC}_{3}.npy'), np.load(f'./y_train_{TACTIC}_{3}.npy')
        d,d2 =  np.load(f'./x_train_{TACTIC}_{0}.npy'), np.load(f'./y_train_{TACTIC}_{0}.npy')
        X_train = np.concatenate((a,b,c,d),axis=0)
        y_train = np.concatenate((a2,b2,c2,d2),axis=0)
    print("Done")
    print(X_train.shape)
            
                
    #Work ends here
   


    logging.info("X_train.shape={}, X_test.shape={}".format(X_train.shape, X_test.shape))
    if args.model == 'XGBC':
        (b,f,a) = X_train.shape
        X_train = np.reshape(X_train, [-1,f*a])
        X_test = np.reshape(X_test, [-1,f*a])
    else:
        normalizer = Normalization(axis=(1)) # along 105 features
        normalizer.adapt(X_train)
        X_train = normalizer(X_train)
        X_test = normalizer(X_test)
        if args.model == "2dcnn":
            X_train = tf.expand_dims(X_train, axis=-1)
            X_test = tf.expand_dims(X_test, axis=-1)
        elif args.model == 'mlp':
            X_train = tf.squeeze(X_train)
            X_test = tf.squeeze(X_test)
        else: # 1dcnn
            (b,f,a) = X_train.shape
            X_train = np.reshape(X_train, [-1,f*a,1])
            X_test = np.reshape(X_test, [-1,f*a, 1])

        del normalizer
    logging.info("X_train.shape={}, X_test.shape={}".format(X_train.shape, X_test.shape))
    return X_train, y_train, X_test, y_test

def analyze_result(y_true, y_pred, prob_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, prob_pred)
    roc_auc = metrics.auc(fpr, tpr)
    pre, rec, thresholds = metrics.precision_recall_curve(y_true, prob_pred)
    prc_auc = metrics.auc(rec, pre)

    cfm = metrics.confusion_matrix(y_true, y_pred)
    TN = cfm[0,0]
    TP = cfm[1,1]
    FP = cfm[0,1]
    FN = cfm[1,0]
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*(precision*recall)/(precision+recall)
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    result = {'accuracy': accuracy, 'roc_auc': roc_auc, 'prc_auc':prc_auc,
        'FPR':FPR, 'FNR': FNR, 'precision': precision, 'recall':recall, 'F1':F1,
        'TP':int(TP), 'FP':int(FP), 'FN':int(FN), 'TN':int(TN)}
    logging.info(result)
    return result

def train_test(logger, args):
    if args.model == 'XGBC':
        train_classic(logger, args)
    elif args.model == 'resnet1d':
        # use pytorch to implement
        logging.error("Please use main_torch.py instead")
        exit()
    else:
        # use tensorflow to implement
        train_deep(logger, args)

def train_classic(logger, args):
    
    logging.info("Train classic")


    # For Training and Saving Main Model 
    X_train, y_train, X_test, y_test = load_data(-1, args)
    clf = XGBClassifier(booster="dart",
                    objective="binary:logistic",
                    random_state=1,
                    nthread=args.nthread)
    print("About to clf fit")
    clf.fit(X_train, y_train, verbose=1)
    print("About to clf predict")
    y_pred = clf.predict(X_test)
    prob_pred = clf.predict_proba(X_test)[:,1]
    results = analyze_result(y_test, y_pred, prob_pred)
    clf.save_model(f"XGBC_{TACTIC}.h5")

    accs, rocs, fprs, fnrs = [], [], [], []
    # For 5-Fold Cross Validation
    for fold in range(5):
        print("Fold line 150 increased.")
        X_train, y_train, X_test, y_test = load_data(fold, args)
        clf = XGBClassifier(booster="dart",
                        objective="binary:logistic",
                        random_state=1,
                        nthread=args.nthread,
                        tree_method ="gpu_hist") #THIS LINE IS THE ONE ENABLING GPU USAGE. CHANGE TO SEE IF ANY DIFFERENCE. 
        print("About to clf fit")
        clf.fit(X_train, y_train, verbose=1)
        print("About to clf predict")
        y_pred = clf.predict(X_test)
        prob_pred = clf.predict_proba(X_test)[:,1]
        results = analyze_result(y_test, y_pred, prob_pred)
        logging.info(f"results: {results}")
        accs.append(results['accuracy'])
        rocs.append(results['roc_auc'])
        fprs.append(results['FPR'])
        fnrs.append(results['FNR'])

    logging.info("Mean over 5 folds: Acc={}+/-{}, ROC={}+/-{}, FPR={}+/-{}, FNR={}+/-{}".format(
            np.mean(accs), np.std(accs), np.mean(rocs), np.std(rocs),
            np.mean(fprs), np.std(accs), np.mean(fnrs), np.std(fnrs)))


def train_deep(logger, args):
    accs, rocs, fprs, fnrs, recalls, precisions = [], [], [], [], [], []
    for fold in range(5):
        X_train, y_train, X_test, y_test = load_data(fold, args)

        if args.optimizer == "AdamW":
            optim = AdamW
        elif args.optimizer == "Adamax":
            optim = Adamax
        elif args.optimizer == "SGD":
            optim = SGD
        else:
            optim = Adam

        if args.decay_step > 0:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = args.lr,
                decay_steps = args.decay_step,
                decay_rate = 0.9)
        else:
            lr_schedule = args.lr

        logdir = "tflogs/scalars/{}_{}_{}".format(args.model, args.optimizer, args.epochs) + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        optimizer=optim(lr_schedule)
        model = None
        if args.model == 'mlp':
            model = model_tf.create_mlp(optimizer=optimizer, activ='relu')
        elif args.model == '2dcnn':
            model = model_tf.create_2d_cnn(optimizer, activ='relu')
        else:
            model = model_tf.create_1d_cnn(optimizer=optimizer, activ='softsign',
                                  bias=not args.non_bias)
        model.summary(print_fn=logger.info)
        model.fit(X_train, y_train,\
                  epochs=args.epochs, batch_size=args.batch_size,\
                  callbacks=[tensorboard_callback,lr_callback])
        del X_train, y_train, optimizer, lr_schedule

        prob_pred = model.predict(X_test, batch_size=args.batch_size)
        y_pred = prob_pred.round()
        results = analyze_result(y_test, y_pred, prob_pred)
        logging.info(f"results: {results}")
        accs.append(results['accuracy'])
        rocs.append(results['roc_auc'])
        fprs.append(results['FPR'])
        fnrs.append(results['FNR'])
        precisions.append(results['precision'])
        recalls.append(results['recall'])
        del X_test, y_test, y_pred, model, tensorboard_callback, lr_callback
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

    logging.info("Mean over 5 folds: Acc={}+/-{}, ROC={}+/-{}, FPR={}+/-{}, FNR={}+/-{},"\
                 "recall={}+/-{}, precision={}+/-{}".format(
            np.mean(accs), np.std(accs), np.mean(rocs), np.std(rocs),
            np.mean(fprs), np.std(accs), np.mean(fnrs), np.std(fnrs),
            np.mean(recalls), np.std(recalls), np.mean(precisions), np.std(precisions)
            ))
    logging.info(f"args={args}")
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DL for Malware Detection new features")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--no_log", action='store_true')
    parser.add_argument("--activ", type=str, default="softsign")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--decay_step", type=int, default=1000) # each epoch 93 step,
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--model", type=str, default="1dcnn")
    parser.add_argument("--model", type=str, default="XGBC")

    parser.add_argument("--non_bias", action='store_true')
    parser.add_argument("--nthread", type=int, default=1)

    args = parser.parse_args()
    # chanegd here. Some default args are set. Changed it to XGBC instead of the 1dcnn that they set it to.

    log_dir = 'logfiles/'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    FORMAT = '%(asctime)-15s %(levelname)s %(filename)s %(lineno)s:: %(message)s'
    FILENAME = '{}/train_{}_{}_{}bs_{}ep_{}.log'.format(log_dir, args.model,\
        args.optimizer, args.batch_size,\
        args.epochs, start_time)
    LOG_LVL = logging.DEBUG if args.verbose else logging.INFO

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logging.Formatter(FORMAT))

    logger = logging.getLogger('')
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(consoleHandler)
    if not args.no_log:
        fileHandler = logging.FileHandler(FILENAME, mode='w')
        fileHandler.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(fileHandler)
    logger.setLevel(LOG_LVL)
    logging.info("args ={}".format(args))

    # Creates and configs a logger.
    print(f"Args: {args}")
    train_test(logger, args)
