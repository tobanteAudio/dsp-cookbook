'''
Based on the video: https://www.youtube.com/watch?v=GiCJS9sUQts
'''
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly


def plot_graphs():
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


def plot_solution():
    # course
    x_start = -2
    x_stop = 2.1
    increment = 0.1
    x = np.arange(start=x_start, stop=x_stop, step=increment)
    y = x**2

    # exact/analytical solution
    dydx = 2*x
    print(f'dydx={dydx}')
    plt.plot(x, dydx, 'o-')

    # numeric solution
    dydx_num = np.diff(y) / np.diff(x)
    print(f'dydx_num={dydx_num}')
    x_start = -2
    x_stop = 2
    x = np.arange(start=x_start, stop=x_stop, step=increment)
    plt.plot(x, dydx_num, 'o-')

    plt.title('dy/dx')
    plt.legend(['Exact', 'Numeric'])
    plt.show()


def polynomial_differentiation():
    # p(x) = x^3 + 2x^2 - x + 3
    # p(x) = 3 - x + 2x^2 + x^3
    p = [3, -1, 2, 1]
    dpdx = poly.polyder(p)
    print(f'dpdx = {dpdx}')

    # p(x) = x^3 + 2
    # p(x) = 2 + x^3
    # p(x) = 2 + 0x + 0x^2 + x^3
    p = [2, 0, 0, 1]
    dpdx = poly.polyder(p)
    print(f'dpdx = {dpdx}')


def main():
    # plot_graphs()
    # plot_solution()
    polynomial_differentiation()


if __name__ == "__main__":
    main()
