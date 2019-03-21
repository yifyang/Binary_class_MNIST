import torch
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def myfit(train_loader, val_loader, model, loss_fn, optimizer, n_epochs, cuda, log_interval,
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    """
    plt_train_loss = []
    plt_val_loss = []

    for epoch in range(start_epoch, n_epochs):

        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)

        val_loss, accuracy = test_epoch(val_loader, model, loss_fn, cuda)
        val_loss /= len(val_loader)

        plt_train_loss.append(train_loss)
        plt_val_loss.append(val_loss)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}. accuracy: {:.2f}'.format(epoch + 1, n_epochs, val_loss, accuracy)

        print(message)

    # make the plot of loss
    train_time = np.arange(len(plt_train_loss))
    val_time = np.arange(len(plt_val_loss))
    cx = plt.plot(train_time, plt_train_loss, 'b', val_time, plt_val_loss, 'r')
    plt.title("loss curve(red for test, blue for train)")
    plt.savefig("loss.jpg")
    plt.show()


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval):
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        loss_inputs = outputs

        loss = loss_fn(loss_inputs, target)
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), np.mean(losses))

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss


def test_epoch(val_loader, model, loss_fn, cuda):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            loss_inputs = outputs

            loss = loss_fn(loss_inputs, target)
            val_loss += loss.item()

            # Get output of the model and obtain the max as preidction.
            prediction = torch.max(outputs, 1)[1]
            pred_y = prediction.data.numpy()
            accuracy = float((pred_y == target.data.numpy()).astype(int).sum()) / float(
                len(target.data.numpy()))

            # make the plot of the architecture of the network
            # with SummaryWriter(comment='Problem5') as w:
            #     w.add_graph(model, (*data,))

    return val_loss, accuracy
