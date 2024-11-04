from pathlib import Path

import torch

from exercise_2.data.shapenet_parts import ShapeNetParts
from exercise_2.model.pointnet import PointNetSegmentation


def train(model, trainloader, valloader, device, config):

    # TODO Declare loss and move to specified device
    loss_criterion = torch.nn.CrossEntropyLoss()

    # TODO Declare optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rate'])

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()

    # keep track of best validation accuracy achieved so that we can save the weights
    best_accuracy = 0.

    # keep track of running average of train loss for printing
    train_loss_running = 0.

    for epoch in range(config['max_epochs']):
        for i, batch in enumerate(trainloader):
            # TODO Add missing pieces, as in the exercise parts before
            model.zero_grad()
            ShapeNetParts.move_batch_to_device(batch, device)
            prediction = model(batch['points'].float())
            loss = loss_criterion(prediction.transpose(1,2), batch['segmentation_labels'].type(torch.LongTensor).to(device))
            loss.backward()
            optimizer.step()
            train_loss_running += loss.item()
            iteration = epoch * len(trainloader) + i
            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                train_loss_running = 0.

            # validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                # TODO Add missing pieces, as in the exercise parts before
                model.eval()
                total, correct = 0, 0
                ious = []

                # forward pass and evaluation for entire validation set
                loss_val = 0.
                for batch_val in valloader:
                    # TODO Add missing pieces, as in the exercise parts before
                    ShapeNetParts.move_batch_to_device(batch_val, device)
                    prediction = model(batch_val['points'].float())
                    predicted_label = prediction.max(axis=2)[1]

                    total += predicted_label.numel()
                    correct += (predicted_label == batch_val['segmentation_labels']).sum().item()

                    part_ious = []
                    for part in range(ShapeNetParts.num_classes):
                        I = torch.sum(torch.logical_and(predicted_label == part, batch_val['segmentation_labels'] == part))
                        U = torch.sum(torch.logical_or(predicted_label == part, batch_val['segmentation_labels'] == part))
                        if U == 0:
                            iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
                        else:
                            iou = I / float(U)
                        part_ious.append(iou)
                    ious.append(torch.mean(torch.stack([torch.tensor(elem, device=device) if type(elem) == int else elem for elem in part_ious])))

                    loss_val += loss_criterion(prediction.transpose(2, 1), batch_val['segmentation_labels'].type(torch.LongTensor).to(device)).item()

                accuracy = 100 * correct / total
                iou = torch.mean(torch.stack(ious)).item()
                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_val / len(valloader):.3f}, val_accuracy: {accuracy:.3f}%, val_iou: {iou:.3f}')

                if accuracy > best_accuracy:
                    torch.save(model.state_dict(), f'exercise_2/runs/{config["experiment_name"]}/model_best.ckpt')
                    best_accuracy = accuracy

                # TODO Add missing pieces, as in the exercise parts before
                model.train()


def main(config):
    """
    Function for training PointNet on ShapeNet
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

    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # Create Dataloaders
    train_dataset = ShapeNetParts('train' if not config['is_overfit'] else 'overfit')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=0,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    val_dataset = ShapeNetParts('val' if not config['is_overfit'] else 'overfit')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=False,   # During validation, shuffling is not necessary anymore
        num_workers=0,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    # Instantiate model
    model = PointNetSegmentation(ShapeNetParts.num_classes)

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'exercise_2/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader, device, config)
