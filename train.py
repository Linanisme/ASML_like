from tqdm import tqdm
from pyprobar import bar, probar
import numpy as np
import torch

def train(model, n_epochs, train_loader, valid_loader, optimizer, criterion):
    train_acc_his, valid_acc_his = [], []
    train_losses_his, valid_losses_his = [], []
    for epoch in range(1, n_epochs + 1):
        # keep track of training and validation loss
        train_loss, valid_loss = 0.0, 0.0
        train_losses, valid_losses = [], []
        train_correct, val_correct, train_total, val_total = 0, 0, 0, 0
        train_pred, train_target = torch.zeros(8, 1), torch.zeros(8, 1)
        val_pred, val_target = torch.zeros(8, 1), torch.zeros(8, 1)
        count = 0
        count2 = 0
        print('running epoch: {}'.format(epoch))
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in probar(train_loader):
            # move tensors to GPU if CUDA is available

            data, target = data.cuda(), target.cuda()
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # calculate accuracy
            pred = output.data.max(dim=1, keepdim=True)[1]
            train_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            train_total += data.size(0)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_losses.append(loss.item() * data.size(0))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            if count == 0:
                train_pred = pred
                train_target = target.data.view_as(pred)
                count = count + 1
            else:
                train_pred = torch.cat((train_pred, pred), 0)
                train_target = torch.cat((train_target, target.data.view_as(pred)), 0)
        train_pred = train_pred.cpu().view(-1).numpy().tolist()
        train_target = train_target.cpu().view(-1).numpy().tolist()
        ######################
        # validate the model #
        ######################
        model.eval()
        for data, target in probar(valid_loader):
            # move tensors to GPU if CUDA is available

            data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # calculate accuracy
            pred = output.data.max(dim=1, keepdim=True)[1]
            val_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            val_total += data.size(0)
            valid_losses.append(loss.item() * data.size(0))
            if count2 == 0:
                val_pred = pred
                val_target = target.data.view_as(pred)
                count2 = count + 1
            else:
                val_pred = torch.cat((val_pred, pred), 0)
                val_target = torch.cat((val_target, target.data.view_as(pred)), 0)
        val_pred = val_pred.cpu().view(-1).numpy().tolist()
        val_target = val_target.cpu().view(-1).numpy().tolist()

        # calculate average losses
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        # calculate average accuracy
        train_acc = train_correct / train_total
        valid_acc = val_correct / val_total

        if epoch % 10 == 0:
            print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                train_loss, valid_loss))
            print('\tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(
                train_acc, valid_acc))
            torch.save(model.state_dict(),'savemodel/{}_{}.pth'.format(epoch,valid_acc))

    train_acc_his.append(train_acc)
    valid_acc_his.append(valid_acc)
    train_losses_his.append(train_loss)
    valid_losses_his.append(valid_loss)
    # print training/validation statistics
    print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        train_loss, valid_loss))
    print('\tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(
        train_acc, valid_acc))



    return train_acc_his, valid_acc_his, train_losses_his, valid_losses_his, model