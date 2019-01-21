import os
import torch
from torch.autograd import Variable

def process_in_epoch(loss_function, model, optimizer, data_loader, use_gpu):

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    for iter, traindata in enumerate(data_loader):
        train_inputs, train_labels = traindata
        train_labels = torch.squeeze(train_labels)

        if use_gpu:
            train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()
        else:
            train_inputs = Variable(train_inputs)

        model.zero_grad()
        model.batch_size = len(train_labels)
        model.hidden = model.init_hidden()
        output = model(train_inputs.t())

        loss = loss_function(output, Variable(train_labels))
        loss.backward()
        optimizer.step()

        # calc training acc
        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == train_labels).sum()
        total += len(train_labels)
        total_loss += loss.data

    return total, total_loss, total_acc
