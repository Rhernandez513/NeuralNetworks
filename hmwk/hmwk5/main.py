import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torch
# import torchvision

# small_geometry_dataset_path = "small_geometry_dataset/"
geometry_dataset_path = "geometry_dataset/"

def get_filenames(path):
    filenames = []
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            filenames.append(filename)
    return filenames

def read_images_into_tensor_files():
    pass

def main():
    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 64)')

    # section 1, read images into tensor files
    file_names = get_filenames(geometry_dataset_path)

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
    testing_images = []
    for key, value in training_files_by_class.items():
        for file_name in value:
            training_images.append(read_image(geometry_dataset_path + file_name))
    for key, value in testing_files_by_class.items():
        for file_name in value:
            testing_images.append(read_image(geometry_dataset_path + file_name))
    training_tensor = torch.stack(training_images)
    test_tensor = torch.stack(testing_images)

    torch.save(training_tensor, "training_tensor.file")
    torch.save(test_tensor, "test_tensor.file")

    loaded_training_tensor = torch.load("training_tensor.file")
    loaded_test_tensor = torch.load("test_tensor.file")

    # section 2, train the model
    # section 3, test the model

    print("done")

if __name__ == "__main__":
    main()