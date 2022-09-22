import numpy as np
import matplotlib.pyplot as plt


def plot_and_show_sine_wave():
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()


# def read_bytes_from_file(filename) -> np.array:
#     with open(filename, 'rb') as f:
#         return np.fromfile(f, dtype=np.dtype('B'))

def read_image(filename):
    image_size = 28
    num_images = 60000
    res = []
    with open(filename, 'rb') as f:
        header = f.read(16)
        number_of_items = f.read(16)
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, image_size, image_size, 1)
        res.append((header, data))
    return res



    # with open(filename, 'rb') as f:
    #     return f.read()

def read_bytes_from_file(filename):
    with open(filename, 'rb') as f:
        return f.read()


def main():
    # raw_train_images = read_bytes_from_file('resources/train-images-idx3-ubyte')
    # print(raw_train_images)
    # print(type(raw_train_images))
    # print(raw)
    # for i in range(31):
    #     print(raw_train_images[i], end='')
    a = read_image('resources/train-images-idx3-ubyte')


    raw_train_labels = read_bytes_from_file('resources/train-labels-idx1-ubyte')

    raw_test_images = read_bytes_from_file('resources/t10k-images-idx3-ubyte')
    raw_test_labels = read_bytes_from_file('resources/t10k-labels-idx1-ubyte')

    print("Hello World!")

if __name__ == '__main__':
    main()


# EOF
