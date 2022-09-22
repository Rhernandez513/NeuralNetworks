import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST


def foo():
    pass

def main():

    mndata = MNIST('resources')

    # each image of the images list is a python list of unsigned bytes
    # In the images: "0" is white, "255" is black, we see in the python-mnist source code library that >200 is dark enough to be considered writing
    # each image is 28x28 pixels

    # labels is a python array of insigned bytes, luckily python translates these to ints no problem
    images, labels = mndata.load_training()

    # testing_images, testing_labels = mndata.load_testing()

    # should render a 2 
    print(mndata.display(images[5]))
    # should print 2 to console
    print(labels[5])


if __name__ == '__main__':
    main()


# EOF
