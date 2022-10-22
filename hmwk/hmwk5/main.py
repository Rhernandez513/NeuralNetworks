import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torch
# import torchvision

class_labels_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
class_labels = ['Pentagon', 'Hexagon', 'Star', 'Circle', 'Nonagon', 'Triangle', 'Heptagon', 'Octagon', 'Square']
class_map = {
    'Pentagon': 0,
    'Hexagon': 1,
    'Star': 2,
    'Circle': 3,
    'Nonagon': 4,
    'Triangle': 5,
    'Heptagon': 6,
    'Octagon': 7,
    'Square': 8
}


def get_filenames(path):
    filenames = []
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            filenames.append(filename)
    return filenames


def load_images_from_tensorfiles():
    loaded_training_tensor = torch.load("training_tensor.file")
    loaded_test_tensor = torch.load("test_tensor.file")
    pass

def read_images_into_tensors(path: str, save: bool, read_image_files: bool) -> tuple:
    file_names = get_filenames(path)

    class_names = set()
    for file_name in file_names:
        class_name = file_name[:file_name.find('_')]
        class_names.add(class_name)

    file_names_by_class = {
        class_name: [] for class_name in class_names
    }

    for file_name in file_names:
        for class_name in class_names:
            if file_name.startswith(class_name):
                file_names_by_class[class_name].append(file_name)

    # seperate images into training and testing
    # first 80% for training, last 20% for testing
    training_files_by_class = {
        class_name: file_names_by_class[class_name][:8000] for class_name in class_names
    }

    testing_files_by_class = {
        class_name: file_names_by_class[class_name][8000:] for class_name in class_names
    }

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []

    for key, value in testing_files_by_class.items():
        for file_name in value:
            testing_labels.append(key)
            if read_image_files:
                testing_images.append(read_image(path + file_name))
    for key, value in training_files_by_class.items():
        for file_name in value:
            training_labels.append(key)
            if read_image_files:
                training_images.append(read_image(path + file_name))

    if not read_image_files:
        training_tensor = torch.load("training_tensor.file")
        test_tensor = torch.load("test_tensor.file")
        return training_tensor, training_labels, test_tensor, testing_labels


    if read_image_files:
        training_tensor = torch.stack(training_images)
        test_tensor = torch.stack(testing_images)
        if save:
            torch.save(training_tensor, "training_tensor.file")
            torch.save(test_tensor, "test_tensor.file")
    else:
        training_tensor = torch.load("training_tensor.file")
        test_tensor = torch.load("test_tensor.file")
    
    return training_tensor, training_labels, test_tensor, testing_labels



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1,
        self.conv1 = nn.Conv2d(3, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(76832, 100)
        self.fc2 = nn.Linear(100, 9)
        # self.fc1 = nn.Linear(4802, 128)
        # self.fc1 = nn.Linear(307328, 128)

        # self.conv1 = nn.Conv2d(3, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    tot_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()

        tot_loss = tot_loss + loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), tot_loss/(batch_idx+1), 100.0*correct/((batch_idx+1)*args.batch_size)))

    print('End of Epoch: {}'.format(epoch))
    print('Training Loss: {:.6f}, Training Accuracy: {:.2f}%'.format(
        tot_loss/(len(train_loader)), 100.0*correct/(len(train_loader)*args.batch_size)))

def test(args, model, device, test_loader):
    model.eval()
    tot_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            tot_loss += torch.nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('Test Loss: {:.6f}, Test Accuracy: {:.2f}%'.format(
        tot_loss/(len(test_loader)), 100.0*correct/(len(test_loader)*args.test_batch_size)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # section 1, read images into tensor files
    path = "geometry_dataset/"
    training_tensor, training_labels, test_tensor, test_labels = read_images_into_tensors(path, save=True, read_image_files=False)
    training_tensor = training_tensor.float()
    test_tensor = test_tensor.float()

    # section 2, train the model
    # section 3, test the model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    training_labels = torch.tensor([class_map[label] for label in training_labels])
    test_labels = torch.tensor([class_map[label] for label in test_labels])

    dataset1 = torch.utils.data.TensorDataset(training_tensor, training_labels)
    dataset2 = torch.utils.data.TensorDataset(test_tensor, test_labels)

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.test_batch_size)

    model = Net().to(device)
    # model.load_state_dict(torch.load("shapes_cnn.pt"))
    # model.eval()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args,model, device, test_loader)
        scheduler.step()

    if args.save_model:
        # torch.save(model.state_dict(), "modified_model_shapes_Adam_SGD_cnn.pt")
        torch.save(model.state_dict(), "shapes_cnn.pt")

    print("done")

if __name__ == "__main__":
    main()