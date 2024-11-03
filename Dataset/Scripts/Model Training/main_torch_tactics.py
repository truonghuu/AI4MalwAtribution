import numpy as np
import logging
import os, time
# from tqdm import tqdm
from datetime import datetime
import argparse
import json
import model_torch
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from sklearn import metrics
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
import random
import pickle

TACTIC = "Impact"

#logging.getLogger('matplotlib.font_manager').disabled = True
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")

def set_seed(seed):
    """Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def my_unit_normalization(X_train, X_test):
    """To avoid using tensorflow, we write this short code to do the task.
    We get mean, std along 105 features. Then normalize input to [0,1] range.
    """
    (b_train,f,a) = X_train.shape
    (b_test,f,a) = X_test.shape
    logging.info("X_train.shape={}, X_test.shape={}".format(X_train.shape, X_test.shape))
    maxs = np.max(X_train, axis=(0,2))
    mins = np.min(X_train, axis=(0,2))
    logging.debug(f"maxs={maxs}, mins={mins}")

    maxs_e = np.expand_dims(maxs, axis=(0,2))
    maxs_er1 = np.repeat(maxs_e, a, axis= 2)
    maxs_er_train = np.repeat(maxs_er1, b_train, axis= 0)
    maxs_er_test = np.repeat(maxs_er1, b_test, axis= 0)
    mins_e = np.expand_dims(mins, axis=(0,2))
    mins_er1 = np.repeat(mins_e, a, axis= 2)
    mins_er_train = np.repeat(mins_er1, b_train, axis= 0)
    mins_er_test = np.repeat(mins_er1, b_test, axis= 0)
    logging.debug(f"maxs_er_train={maxs_er_train.shape}, mins_er_train={mins_er_train.shape}")
    X_train_n = (X_train - mins_er_train)/(maxs_er_train - mins_er_train)
    X_test_n = (X_test - mins_er_test)/(maxs_er_test - mins_er_test)
    maxs_n = np.mean(X_train_n, axis=(0,2))
    mins_n = np.std(X_train_n, axis=(0,2))
    logging.debug(f"maxs_n={maxs_n}, mins_n={mins_n}")
    return X_train_n, X_test_n

def my_normalization(X_train, X_test):
    """To avoid using tensorflow, we write this short code to do the task.
    We get mean, std along 105 features. Then normalize input to zero mean, and unit
    variance.
    """
    (b_train,f,a) = X_train.shape
    (b_test,f,a) = X_test.shape
    logging.info("X_train.shape={}, X_test.shape={}".format(X_train.shape, X_test.shape))
    means = np.mean(X_train, axis=(0,2))
    stds = np.std(X_train, axis=(0,2))
    logging.debug(f"means={means}, stds={stds}")

    means_e = np.expand_dims(means, axis=(0,2))
    means_er1 = np.repeat(means_e, a, axis= 2)
    means_er_train = np.repeat(means_er1, b_train, axis= 0)
    means_er_test = np.repeat(means_er1, b_test, axis= 0)
    stds_e = np.expand_dims(stds, axis=(0,2))
    stds_er1 = np.repeat(stds_e, a, axis= 2)
    stds_er_train = np.repeat(stds_er1, b_train, axis= 0)
    stds_er_test = np.repeat(stds_er1, b_test, axis= 0)
    logging.debug(f"mean_er_train={means_er_train.shape}, stds_er_train={stds_er_train.shape}")
    X_train_n = (X_train - means_er_train)/stds_er_train
    X_test_n = (X_test - means_er_test)/stds_er_test
    means_n = np.mean(X_train_n, axis=(0,2))
    stds_n = np.std(X_train_n, axis=(0,2))
    logging.debug(f"means_n={means_n}, stds_n={stds_n}")
    return X_train_n, X_test_n


def load_data(fold, model):
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
    # X_test, y_test = np.load(f'./x_test_{TACTIC}.npy'.format(fold)), np.load(f'./y_test_{TACTIC}.npy')
    # X_train, y_train = np.load(f'./x_train_{TACTIC}.npy'.format(fold)), np.load(f'./y_train_{TACTIC}.npy')
    
    
    if model == 'XGBC':
        (b,f,a) = X_train.shape
        X_train = np.reshape(X_train, [-1,f*a])
        X_test = np.reshape(X_test, [-1,f*a])
    else:
        X_train, X_test = my_normalization(X_train, X_test)
        if model == 'mlp':
            (b,f,a) = X_train.shape
            X_train = np.reshape(X_train, [-1,f*a])
            X_test = np.reshape(X_test, [-1,f*a])
            y_train = np.reshape(y_train, [-1,1])
            y_test = np.reshape(y_test, [-1,1])
        else:# model == "1dcnn" or resnet1d
            (b,f,a) = X_train.shape
            X_train = np.reshape(X_train, [-1,1,f*a])# pytorch default is channel first
            X_test = np.reshape(X_test, [-1,1,f*a])
            y_train = np.reshape(y_train, [-1,1])
            y_test = np.reshape(y_test, [-1,1])
    if fold == 0:
        logging.info("X_train.shape={}, X_test.shape={}".format(X_train.shape, X_test.shape))
        logging.debug(f"y_train.shape={y_train.shape}, y_test.shape={y_test.shape}")
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

def train_test(logger, args, conf):
    if args.model in ['XGBC', 'lgb']:
        train_classic(logger, args)
    else: # 'resnet1d', 'mlp'
        # use pytorch to implement
        train_deep_torch(logger, args, conf)

def train_classic(logger, args):
    accs, rocs, fprs, fnrs, recalls, precisions = [], [], [], [], [], []
    logging.info("Train classic")
    for fold in range(-1, 4):
        X_train, y_train, X_test, y_test = load_data(fold, 'XGBC')
        if args.model == 'XGBC':
            clf = XGBClassifier(booster='gbtree', #"dart",
                            max_depth=10,
                            objective="binary:logistic",
                            random_state=1,
                            nthread=args.nthread)
            clf.fit(X_train, y_train, verbose=1)
            y_pred = clf.predict(X_test)
            prob_pred = clf.predict_proba(X_test)[:,1]
        elif args.model == 'lgb':
            clf = LGBMClassifier(boosting_type="gbdt", #"dart",
                             objective="binary",
                             learning_rate=0.25,#0.2,# default is 0.1
                             num_leaves=100,#100
                             num_iterations=350,#300,#200,
                             n_estimators=170,#150,
                             #min_data_in_leaf=20,
                             #n_estimators=130,
                             #reg_alpha =1e-4,
                             reg_lambda=1e-4, # good
                             max_bin=250,#250,#200,
                             #lamba_l2=1e-4, # not good
                             random_state=1,
                             n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            prob_pred = clf.predict_proba(X_test)[:,1]
            my_pred = np.asarray((prob_pred>0.59)*1)
            logging.info(f"Equal or not={(my_pred==y_pred).all()}")
            logging.debug(f"prob_pred={prob_pred}")
            if fold == -1:
                with open(f"LGBM_{TACTIC}.h5", 'wb') as file:
                    pickle.dump(clf, file)
                    

        results = analyze_result(y_test, my_pred, prob_pred)
        #logging.info(f"results: {results}")
        accs.append(results['accuracy'])
        rocs.append(results['roc_auc'])
        fprs.append(results['FPR'])
        fnrs.append(results['FNR'])
        precisions.append(results['precision'])
        recalls.append(results['recall'])

    logging.info(f"clf={clf}")
    logging.info("Mean over 5 folds: Acc={}+/-{}, ROC={}+/-{}, FPR={}+/-{}, FNR={}+/-{},"\
                 "recall={}+/-{}, precision={}+/-{}".format(
            np.mean(accs), np.std(accs), np.mean(rocs), np.std(rocs),
            np.mean(fprs), np.std(accs), np.mean(fnrs), np.std(fnrs),
            np.mean(recalls), np.std(recalls), np.mean(precisions), np.std(precisions)
            ))

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index].astype(float)
        y = self.y[index].astype(float)
        return x, y


def train_deep_torch(logger, args, conf):
    logging.info(f"config={conf}")
    accs, rocs, fprs, fnrs, recalls, precisions = [], [], [], [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device={device}")
    for fold in range(5):
        X_train, y_train, X_test, y_test = load_data(fold, conf['arch'])

        train_set = MyDataset(X_train, y_train)
        test_set = MyDataset(X_test, y_test)
        train_loader = torch.utils.data.DataLoader(train_set,
                            batch_size=conf['batch_size'],
                            shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set,
                            batch_size=conf['batch_size'],
                            shuffle=False)
        if conf['arch'] == 'mlp':
            net = model_torch.MLP(conf)
        elif conf['arch'] == '1dcnn':
            net = model_torch.Net1D(conf)
        else: # resnet1d
            net = model_torch.Net(conf)
        net.to(device)
        model_stats = summary(net, input_size=(1, 1, 27090))
        logging.info(str(model_stats))
        criterion = nn.BCELoss()
        pg = [p for p in net.parameters() if p.requires_grad]
        if conf.get('optimizer', 'Adam') == "SGD":
            optimizer = torch.optim.SGD(pg, lr=conf['lr'],
                momentum=conf.get('momentum', 0.9),
                weight_decay=conf.get('weight_decay', 0))
        elif conf.get('optimizer', 'Adam') == 'AdamW':
            optimizer = torch.optim.AdamW(pg, lr=conf['lr'],
                weight_decay=conf.get('weight_decay', 0))
        else:
            optimizer = torch.optim.Adam(pg, lr=conf['lr'],
                weight_decay=conf.get('weight_decay', 0))

        if conf.get('scheduler', "") == "ExponentialLR":
            lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                            gamma=0.98)
        elif conf.get('scheduler', "") == "MultiStepLR":
            lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                            conf['milestones'], gamma=conf.get('gamma', 0.1))
        else:
            lr_schedule = None
        logging.info(f"scheduler = {lr_schedule}")

        ########## Start Training #####################
        set_seed(conf.get('random_seed', 1))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/train_{}_{}_{}'.format(conf['arch'], args.epochs, timestamp))
        running_loss = 0.
        last_loss = 0.
        for epoch in range(args.epochs):
            no_batches = len(train_loader)
            net.to(device)
            net.train(True)
            correct = 0
            total = 0
            #train_loader = tqdm(train_loader)
            logging.info(f"Epoch={epoch}, lr={optimizer.param_groups[0]['lr']}")
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                logging.debug(f"inputs={inputs.shape}, type={inputs.type()}")
                logging.debug(f"net={next(net.parameters()).is_cuda}, inputs={inputs.get_device()}")
                optimizer.zero_grad()
                prob_pred = net(inputs)
                y_pred = torch.round(prob_pred)
                correct += (y_pred == labels).sum()
                total += labels.size(0)
                loss = criterion(prob_pred, labels)
                loss.backward()
                optimizer.step()
                # Gather data and report
                running_loss += loss.item()
                if i % 20 == 19:
                    last_loss = running_loss / 20 # loss per batch
                    logging.info('       batch {} loss: {}'.format(i + 1, last_loss))
                    tb_x = epoch * no_batches  + i + 1
                    writer.add_scalar('Loss/train', last_loss, tb_x)
                    running_loss = 0.
            if conf.get('scheduler', "") == 'ExponentialLR':
                if epoch %  5 == 0:
                    lr_schedule.step()
            elif lr_schedule is not None:
                lr_schedule.step()
            training_accuracy = correct/total if total > 0 else 0
            logging.info(f"Epoch {epoch}, accuracy={training_accuracy}")
            writer.add_scalar("accuracy/train", training_accuracy, epoch)


        # Complete training, test the model
        del train_loader, X_train, y_train, optimizer, criterion
        net.eval() #train(False)
        prob_preds = torch.empty(0,1).to(device).float()
        actual_labels = torch.empty(0,1).to(device).float()
        pred_labels =torch.empty(0,1).to(device).float()

        for i, data in enumerate(test_loader):
            with torch.no_grad():
                inputs, labels = data
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                _prob_pred = net(inputs)
                #test_loss = criterion(prob_pred, labels)
                _pred_y = torch.round(_prob_pred)
                prob_preds = torch.cat((prob_preds, _prob_pred), dim=0)
                pred_labels = torch.cat((pred_labels, _pred_y), dim=0)
                actual_labels = torch.cat((actual_labels, labels), dim=0)

        results = analyze_result(actual_labels.cpu().detach().numpy(),
            pred_labels.cpu().detach().numpy(),
            prob_preds.cpu().detach().numpy())
        logging.info(f"results: {results}")
        accs.append(results['accuracy'])
        rocs.append(results['roc_auc'])
        fprs.append(results['FPR'])
        fnrs.append(results['FNR'])
        precisions.append(results['precision'])
        recalls.append(results['recall'])
        del X_test, y_test, y_pred, net

    logging.info("Mean over 5 folds: Acc={}+/-{}, ROC={}+/-{}, FPR={}+/-{}, FNR={}+/-{},"\
                 "recall={}+/-{}, precision={}+/-{}".format(
            np.mean(accs), np.std(accs), np.mean(rocs), np.std(rocs),
            np.mean(fprs), np.std(accs), np.mean(fnrs), np.std(fnrs),
            np.mean(recalls), np.std(recalls), np.mean(precisions), np.std(precisions)
            ))
    logging.info(f"conf={conf}")
    logging.info(f"args={args}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DL with Pytorch for Malware Detection new features")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--no_log", action='store_true')
    parser.add_argument("--epochs", type=int, default=20)
    # parser.add_argument("--model", type=str, default="1dcnn") Original
    parser.add_argument("--model", type=str, default="lgb")
    parser.add_argument("--nthread", type=int, default=-1)
    parser.add_argument("--config", type=str, default="config/resnet1d.json")

    args = parser.parse_args()
    conf = json.load(open(args.config, "r"))

    log_dir = 'logfiles/'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    FORMAT = '%(asctime)-15s %(levelname)s %(filename)s %(lineno)s:: %(message)s'
    print(args.model)
    if args.model in ['XGBC', 'lgb'] :
        FILENAME = '{}/train_{}_{}.log'.format(log_dir, args.model,\
            start_time)
    else:
        FILENAME = '{}/train_torch_{}_{}_lr{}_{}_{}bs_{}ep_{}.log'.format(log_dir, conf['arch'],\
            conf['optimizer'], conf['lr'], conf['scheduler'], conf['batch_size'],\
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

    train_test(logger, args, conf)
