from pathlib import Path

import numpy as np
import torch
from exercise_2.util.toy_data import generate_toy_data


class SimpleDataset(torch.utils.data.Dataset):
    # TODO: Implement __init__, __getitem__, __len__
    def __init__(self, split):
        n = 0
        if split == "train":
            n = 4096
        elif split == "val":
            n = 1024
        samples, lables = generate_toy_data(n)
        self.samples = samples
        self.lables = lables
        pass

    def __getitem__(self, idx):
        return (np.expand_dims(self.samples[idx], axis=0), self.lables[idx])
        pass

    def __len__(self):
        return len(self.samples)
        pass


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=4, kernel_size=4, stride=3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(4)
        # TODO: Add conv2 and conv3 with the same parameters as conv1; also, add bn2 and bn3
        self.conv2 = torch.nn.Conv3d(in_channels=4, out_channels=8, kernel_size=4, stride=3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(8)
        self.conv3 = torch.nn.Conv3d(in_channels=8, out_channels=16, kernel_size=4, stride=3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(16)
        # TODO: Add Linear layer for classification which reduces the number of features from 16 to 2
        self.fc = torch.nn.Linear(in_features=16, out_features=2)
        # TODO: Add a ReLU
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        # TODO: Move tensor through layers 2 and 3
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # TODO: Apply the classification layer. Use .view() to reshape the output of layer 3 into the correct format
        x = self.fc(x.view(-1, 16))
        return x


def train(model, train_dataloader, val_dataloader, device, config):
    # TODO Declare Loss function; Use CrossEntropyLoss
    loss_criterion = torch.nn.CrossEntropyLoss()

    # TODO Declare optimizer; Use ADAM with learning rate from config['learning_rate']
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rate'])

    # Set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()

    # Keep track of best validation accuracy achieved so that we can save the weights
    best_accuracy = 0.

    for epoch in range(config['max_epochs']):
        # Keep track of running average of train loss for printing
        train_loss_running = 0.

        for i, batch in enumerate(train_dataloader):
            input_data, target_labels = batch
            # TODO Move input_data and target_labels to device
            input_data, target_labels = input_data.to(device), target_labels.to(device)
            # This is where the actual training happens:
            # 1 Zero out gradients from last iteration
            optimizer.zero_grad()
            # 2 Perform forward pass
            prediction = model(input_data)
            # 3 Calculate loss
            loss = loss_criterion(prediction, target_labels)
            # 4 Compute gradients
            loss.backward()
            # 5 Adjust weights using the optimizer
            optimizer.step()

            # Loss logging
            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + i
            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                train_loss_running = 0.

            # Validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                # Set model to eval, important if your network has e.g. dropout or batchnorm layers
                model.eval()
                # Forward pass and evaluation for entire validation set
                # Here, we calculate the loss and accuracy values over the val set
                total, correct = 0, 0
                loss_val = 0.
                for batch_val in val_dataloader:
                    input_data, target_labels = batch_val
                    # TODO Move input_data and target_labels to device
                    input_data, target_labels = input_data.to(device), target_labels.to(device)
                    with torch.no_grad():
                        prediction = model(input_data)

                    _, predicted_labels = torch.max(prediction, dim=1)

                    total += predicted_labels.shape[0]
                    correct += (predicted_labels == target_labels).sum().item()

                    loss_val += loss_criterion(prediction, target_labels).item()
                accuracy = 100 * correct / total
                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_val / len(val_dataloader):.3f}, val_accuracy: {accuracy:.3f}%')

                # Saving the best checkpoints
                if accuracy > best_accuracy:
                    torch.save(model.state_dict(), f'exercise_2/runs/{config["experiment_name"]}/model_best.ckpt')
                    best_accuracy = accuracy

                # Set model back to train
                model.train()


def main(config):
    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # Create Dataloaders
    train_dataset = SimpleDataset('train')  # TODO Instantiate Dataset in train split
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    val_dataset = SimpleDataset('val')  # TODO Instantiate Dataset in val split
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=False,   # During validation, shuffling is not necessary anymore
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    # TODO Instantiate model and move to device
    model = SimpleModel().to(device)

    # Create folder for saving checkpoints
    Path(f'exercise_2/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader, device, config)


if __name__ == '__main__':
    main(config={
        'experiment_name': 'simple_nn',
        'device': 'cuda:0',
        'batch_size': 32,
        'resume_ckpt': None,
        'learning_rate': 0.001,
        'max_epochs': 5,
        'print_every_n': 10,
        'validate_every_n': 100
    })
