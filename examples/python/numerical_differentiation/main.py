'''
Based on the video: https://www.youtube.com/watch?v=GiCJS9sUQts
'''
import matplotlib.pyplot as plt
import numpy as np


def plot_function():
    # fine
    x_start = -2.0
    x_stop = 2.1
    increment = 0.1
    x = np.arange(start=x_start, stop=x_stop, step=increment)
    y = x**2
    plt.plot(x, y)

    # course
    x_start = -2
    x_stop = 3
    increment = 1
    x = np.arange(start=x_start, stop=x_stop, step=increment)
    y = x**2
    plt.plot(x, y, '-o')

    plt.title('y(x)')
    plt.show()


def main():
    plot_function()


if __name__ == "__main__":
    main()
