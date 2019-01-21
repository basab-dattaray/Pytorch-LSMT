import os
import torch
import copy
from torch.utils.data import DataLoader
import utils.DataProcessing as DP
import utils.LSTMClassifier as LSTMC
import torch.optim as optim
import torch.nn as nn


from apps.config import *
from apps.hyper_params import *
from apps.model_maker import *

use_gpu = torch.cuda.is_available()

use_plot = True
use_save = True
if use_save:
    import pickle
    from datetime import datetime



def adjust_learning_rate(optimizer, epoch):
    lr = LEARNING_RATE * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_input_data():
    # global filenames
    train_file = os.path.join(DATA_DIR, TRAIN_FILE)
    test_file = os.path.join(DATA_DIR, TEST_FILE)
    fp_train = open(train_file, 'r')
    train_filenames = [os.path.join(TRAIN_DIR, line.strip()) for line in fp_train]
    filenames = copy.deepcopy(train_filenames)
    fp_train.close()
    fp_test = open(test_file, 'r')
    test_filenames = [os.path.join(TEST_DIR, line.strip()) for line in fp_test]
    fp_test.close()
    corpus = DP.Corpus(DATA_DIR, filenames)
    return train_filenames, test_filenames, corpus


def process_all():
    ### parameter setting
    embedding_dim = 100
    hidden_dim = 50
    sentence_len = 64
    train_filenames, test_filenames, corpus = get_input_data()
    # filenames.extend(test_filenames)
    nlabel = 8
    ### create model
    model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                 vocab_size=len(corpus.dictionary), label_size=nlabel, batch_size=BATCH_SIZE,
                                 use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()
    ### data processing
    dtrain_set = DP.TxtDatasetProcessing(DATA_DIR, TRAIN_DIR, TRAIN_FILE, TRAIN_LABEL, sentence_len, corpus)
    train_loader = DataLoader(dtrain_set,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4
                              )
    dtest_set = DP.TxtDatasetProcessing(DATA_DIR, TEST_DIR, TEST_FILE, TEST_LABEL, sentence_len, corpus)
    test_loader = DataLoader(dtest_set,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4
                             )
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    ### training procedure
    for epoch in range(EPOCHS):
        optimizer = adjust_learning_rate(optimizer, epoch)

        total, total_loss, total_acc = process_in_epoch(loss_function, model, optimizer, train_loader, use_gpu)
        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / total)


        total, total_loss, total_acc = process_in_epoch(loss_function, model, optimizer, test_loader, use_gpu)
        test_loss_.append(total_loss / total)
        test_acc_.append(total_acc / total)

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch, EPOCHS, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))

        print()

    param = {}
    param['lr'] = LEARNING_RATE
    param['batch size'] = BATCH_SIZE
    param['embedding dim'] = embedding_dim
    param['hidden dim'] = hidden_dim
    param['sentence len'] = sentence_len
    result = {}
    result['train loss'] = train_loss_
    result['test loss'] = test_loss_
    result['train acc'] = train_acc_
    result['test acc'] = test_acc_
    result['param'] = param
    if use_plot:
        import PlotFigure as PF
        PF.PlotFigure(result, use_save)
    if use_save:
        filename = 'log/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
        result['filename'] = filename

        fp = open(filename, 'wb')
        pickle.dump(result, fp)
        fp.close()
        print('File %s is saved.' % filename)


    # loss_fraction.append(total_loss / total)
    # accuracy_fraction.append(total_acc / total)


if __name__=='__main__':
    process_all()
