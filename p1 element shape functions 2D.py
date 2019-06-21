import numpy as np


def tri_gauss_quad(k):
    """
    this rotuine gives the quadrature points and weights of quadrature to
    integrate numerically a polynomial of order "k".
    :param k: polynomial order to approximate
    :return: xi :  coordinates of quad points
             w  :  value of quad weight
    """

    if k == 1:
        # 1st order ---> 1 quad point
        xi = np.array([[1 / 3, 1 / 3]])
        w = np.array([0.5])
    elif k == 2:
        # 2nd order ---> 3 quad points
        xi = np.array([[2 / 3., 1. / 6.], [1. / 6., 2. / 3.], [1. / 6., 1. / 6.]])
        w = np.array([1. / 6., 1. / 6., 1. / 6.])
    elif k == 3:
        # 3rd order ---> 4 quad points
        xi = np.array([[1. / 3., 1. / 3.], [3. / 5., 1. / 5.], [1. / 5., 3. / 5.], [1. / 5., 1. / 5.]])
        w = np.array([-27. / 96., 25. / 96., 25. / 96., 25. / 96.])
    return xi, w


def p1_shape_function(x, n):
    """
    this routine evaluates the shape functions and their spatial gradients
    at the quadrature points in the iso-parametric configuration for P1-elements
    :param x: nodes of the of the elements in non-iso parametric configuration
    :param n: order of quadrature
    :return: Np : values ​​of the function as assessed at the quad point.
             DN : gradient of Np
             J  : Jacobian
    """

    xi = tri_gauss_quad(n)[0][0]
    xi1 = xi[0]
    xi2 = xi[1]
    xi3 = 1 - xi1 - xi2
    N1hat = xi1
    N2hat = xi2
    N3hat = xi3
    N = np.vstack((N1hat, N2hat, N3hat))
    Bhat = np.array([[1., 0., -1.], [0., 1., -1.]])
    J = np.dot(x, Bhat.T)
    DN = np.dot(np.linalg.inv(J.T), Bhat)
    return N, DN, J


"""
Execution
"""

x = np.array([[0, 0], [0, 3], [5, 0]]).T
n = 2
print(p1_shape_function(x, n))
