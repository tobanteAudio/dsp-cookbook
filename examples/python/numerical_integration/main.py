'''
Based on the video: https://www.youtube.com/watch?v=LJYLxEpH2vA
'''
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy import integrate


def plot_function():
    x_start = -1.0
    x_stop = 1.1
    increment = 0.1

    x = np.arange(start=x_start, stop=x_stop, step=increment)
    y = x**2

    plt.plot(x, y)
    plt.xlabel('x')
    plt.xlabel('x')
    plt.axis([0, 1, 0, 1])
    plt.fill_between(x, y)
    plt.show()


def trapezoid_rule():
    a = 0
    b = 1
    N = 100

    x = np.linspace(a, b, N+1)
    y = x**2

    y_right = y[1:]
    y_left = y[:-1]

    dx = (b-a)/N
    A = (dx/2)*np.sum(y_right+y_left)

    print(A)
    print(np.trapz(y, x, dx))
    print(integrate.quad(lambda x: x**2, a, b))

    plt.plot(x, y)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()


def polynomial_integration():
    # p(x) = x^3 + 2x^2 - x + 3
    # p(x) = 3 - x + 2x^2 + x^3
    p = [3, -1, 2, 1]

    # Find indefinite integral
    I = poly.polyint(p)
    print(f'I = {I}')

    # Find definite integral
    a = -1
    b = 1
    A = poly.polyval(b, I) - poly.polyval(a, I)
    I = poly.polyint(p)
    print(f'A = {A}')


def main():
    # plot_function()
    # trapezoid_rule()
    polynomial_integration()


if __name__ == "__main__":
    main()
