import numpy as np


def gauss_quadrature_triangle(k):
    """
    this routine gives the quadrature points and weights of quadrature to
    integrate numerically a polynomial of order "k".
    :param k: polynomial order to approximate
    :return: xi :  coordinates of quad points
             w  :  value of quad weight
    """

    if k == 1:
        xi = np.array([[1 / 3, 1 / 3]])
        w = np.array([0.5])
    elif k == 2:
        xi = np.array([[2 / 3., 1. / 6.], [1. / 6., 2. / 3.], [1. / 6., 1. / 6.]])
        w = np.array([1. / 6., 1. / 6., 1. / 6.])
    elif k == 3:
        xi = np.array([[1. / 3., 1. / 3.], [3. / 5., 1. / 5.], [1. / 5., 3. / 5.], [1. / 5., 1. / 5.]])
        w = np.array([-27. / 96., 25. / 96., 25. / 96., 25. / 96.])
    return xi, w


def p1_shape_function(triangle, xi):
    """
    this routine evaluates the shape functions and their spatial gradients
    at the quadrature points in the iso-parametric configuration for P1-elements
    :param triangle: nodes of the of the elements in non-iso parametric configuration
    :param xi: point of quadrature
    :return: N : values ​​of the function as assessed at the quad point.
             DN : gradient of N
             J  : Jacobian
    """

    xi1, xi2 = xi
    xi3 = 1 - xi1 - xi2
    N1hat = xi1
    N2hat = xi2
    N3hat = xi3
    N = np.vstack((N1hat, N2hat, N3hat))
    nabla_Nhat = np.array([[1., 0., -1.], [0., 1., -1.]])
    J = np.dot(triangle, nabla_Nhat.T)
    nabla_N = np.dot(np.linalg.inv(J.T), nabla_Nhat)
    return N, nabla_N, J


def p2_shape_function(triangle, xi):
    """
    this routine evaluates the shape functions and their spatial gradients
    at the quadrature points in the iso-parametric configuration for P2-elements
    :param triangle: nodes of the of the elements in non-iso parametric configuration
    :param xi: point of quadrature
    :return: N : values of the function as assessed at the quad point.
             nabla_N : gradient of N
             J  : Jacobian
    """
    xi1, xi2 = xi
    xi3 = 1 - xi1 - xi2
    n1hat = xi1 * (2. * xi1 - 1.)
    n2hat = xi2 * (2. * xi2 - 1.)
    n3hat = xi3 * (2. * xi3 - 1.)
    n4hat = 4. * xi1 * xi2
    n5hat = 4. * xi2 * xi3
    n6hat = 4. * xi1 * xi3
    N = np.vstack((n1hat, n2hat, n3hat, n4hat, n5hat, n6hat))
    nabla_Nhat = np.array([[4. * xi1 - 1., 0., -(4. * xi3 - 1.), 4. * xi2, - 4. * xi2, 4. * (xi3 - xi1)],
                           [0., 4. * xi2 - 1., -(4. * xi3 - 1.), 4. * xi1, 4. * (xi3 - xi2), - 4. * xi1]])
    J = np.dot(triangle, nabla_Nhat.T)
    nabla_N = np.dot(np.linalg.inv(J.T), nabla_Nhat)
    return N, nabla_N, J



"""
Execution
"""
n = 2
xi = gauss_quadrature_triangle(n)[0][0]
x_p1 = np.array([[0, 0], [0, 3], [5, 0]]).T
x_p2 = np.array([[0, 0], [0, 1.5], [2.5, 0], [2.5, 1.5], [0, 3], [5, 0]]).T

print(p1_shape_function(triangle=x_p1, xi=xi))
print(p2_shape_function(triangle=x_p2, xi=xi))
