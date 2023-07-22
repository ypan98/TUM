"""trainer utility for 3DCNN"""

from pathlib import Path

import torch

from exercise_2.data.shapenet import ShapeNetVox
from exercise_2.model.cnn3d import ThreeDeeCNN


def main(config):
    """
    Driver function for training 3DCNN on ShapeNet
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "exercise_2/runs/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'batch_size': batch size for training and validation dataloaders
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate': learning rate for optimizer
                   'max_epochs': total number of epochs after which training should stop
                   'print_every_n': print train loss every n iterations
                   'validate_every_n': print validation loss and validation accuracy every n iterations
                   'is_overfit': if the training is done on a small subset of data specified in exercise_2/split/overfit.txt,
                                 train and validation done on the same set, so error close to 0 means a good overfit. Useful for debugging.
    """

    # declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # create dataloaders
    trainset = ShapeNetVox('train' if not config['is_overfit'] else 'overfit')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    valset = ShapeNetVox('val' if not config['is_overfit'] else 'overfit')
    valloader = torch.utils.data.DataLoader(valset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    # instantiate model
    model = ThreeDeeCNN(ShapeNetVox.num_classes)

    # load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # move model to specified device
    model.to(device)

    # create folder for saving checkpoints
    Path(f'exercise_2/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # start training
    train(model, trainloader, valloader, device, config)


def train(model, trainloader, valloader, device, config):

    # declare loss and move to specified device
    # TODO: Define loss
    loss_criterion = torch.nn.CrossEntropyLoss()

    loss_criterion.to(device)

    # TODO: declare optimizer (learning rate provided in config)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rate'])

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()

    # keep track of best validation accuracy achieved so that we can save the weights
    best_accuracy = 0.

    # keep track of running average of train loss for printing
    train_loss_running = 0.

    for epoch in range(config['max_epochs']):
        for i, batch in enumerate(trainloader):
            # move batch to device
            ShapeNetVox.move_batch_to_device(batch, device)

            # TODO: zero out previously accumulated gradients
            model.zero_grad()

            # TODO: forward pass
            prediction = model(batch['voxel'])

            # TODO: compute total loss = sum of loss for whole prediction + losses for partial predictions
            loss_total = torch.zeros([1], dtype=batch['voxel'].dtype, requires_grad=True).to(device)
            for output_idx in range(prediction.shape[1]):
                loss_total = loss_total + loss_criterion(prediction[:, output_idx, :], batch['label'])  # TODO: Loss due to prediction[:, output_idx, :] (output_idx=0 for global prediction, 1-8 local)

            # TODO: compute gradients on loss_total (backward pass)
            loss_total.backward()

            # TODO: update network params
            optimizer.step()

            # loss logging
            train_loss_running += loss_total.item()
            iteration = epoch * len(trainloader) + i

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                train_loss_running = 0.

            # validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):

                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                model.eval()

                loss_total_val = 0
                total, correct = 0, 0
                # forward pass and evaluation for entire validation set
                for batch_val in valloader:
                    ShapeNetVox.move_batch_to_device(batch_val, device)

                    with torch.no_grad():
                        # TODO: Get prediction scores
                        prediction = model(batch_val["voxel"])

                    # TODO: Get predicted labels from scores
                    _, pred_label = torch.max(prediction[:, 0, :], dim=1)   # gloabl max
                    predicted_label = pred_label

                    # TODO: keep track of total / correct / loss_total_val
                    total += predicted_label.shape[0]
                    correct += torch.sum(predicted_label == batch_val['label']).item()

                    for i in range(prediction.shape[1]):
                        loss_total_val += loss_criterion(prediction[:, i, :], batch_val['label']).item()

                accuracy = 100 * correct / total

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_total_val / len(valloader):.3f}, val_accuracy: {accuracy:.3f}%')

                if accuracy > best_accuracy:
                    torch.save(model.state_dict(), f'exercise_2/runs/{config["experiment_name"]}/model_best.ckpt')
                    best_accuracy = accuracy

                # set model back to train
                model.train()
