'''
Based on the video: https://www.youtube.com/watch?v=8XeGDpZZYeU
'''
import numpy as np
import numpy.linalg as la


def square_mat2():
    # x + 2y = 5
    # 3x + 4y = 6
    A = np.array([[1, 2], [3, 4]])
    b = np.array([[5], [6]])

    Ainv = la.inv(A)
    x = Ainv.dot(b)
    print(x)

    x = la.solve(A, b)
    print(x)


def non_square_mat23():
    # x + 2y = 5
    # 3x + 4y = 6
    # 7x + 8y = 9
    A = np.array([[1, 2], [3, 4], [7, 8]])
    b = np.array([[5], [6], [9]])

    x = la.inv(A.transpose() * np.mat(A)) * A.transpose() * b
    print(x)

    x = la.lstsq(A, b, rcond=None)[0]
    print(x)


def square_mat3():
    #  4x + 3y + 2z =  25
    # -2x + 2y + 3z = -10
    #  3x + 5y + 2z =  -4
    A = np.array([[4, 3, 2], [-2, 2, 3], [3, -5, 2]])
    b = np.array([[25], [-10], [-4]])

    x = la.inv(A.transpose() * np.mat(A)) * A.transpose() * b
    print(x)

    x = la.lstsq(A, b, rcond=None)[0]
    print(x)


def main():
    # square_mat2()
    # non_square_mat23()
    square_mat3()


if __name__ == "__main__":
    main()
