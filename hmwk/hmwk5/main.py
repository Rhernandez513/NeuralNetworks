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
    # file_names.sort()

    images = []
    for file_name in file_names:
        img = read_image(geometry_dataset_path + file_name)
        images.append(img)

    count = len(images)

    images = torch.stack(images)

    # seperate images into training and testing
    # first 80% for training, last 20% for testing
    divider = int(count * 0.8)
    training_tensor = images[:divider]
    test_tensor = images[divider:]

    # torch.save(training_tensor, "training_tensor.file")
    # torch.save(test_tensor, "test_tensor.file")

    loaded_training_tensor = torch.load("training_tensor.file")
    loaded_test_tensor = torch.load("test_tensor.file")

    loaded_training_tensor = torch.stack(loaded_training_tensor)
    loaded_test_tensor = torch.stack(loaded_test_tensor)

    # print(torch.eq(loaded_test_tensor, test_tensor))
    # print(torch.equal(loaded_test_tensor, test_tensor))
    # print(torch.eq(loaded_training_tensor, training_tensor))
    # print(torch.equal(loaded_training_tensor, training_tensor))

    print(torch.all(torch.eq(loaded_test_tensor, test_tensor)))
    print(torch.all(torch.eq(loaded_training_tensor, training_tensor)))

    print("done")

if __name__ == "__main__":
    main()