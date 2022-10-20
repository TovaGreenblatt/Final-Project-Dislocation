import numpy as np


# Enumerators
X_INDEX = 0
Y_INDEX = 1
Z_INDEX = 2


def getNormalizedMatrix(x, y, z):
    # normalization the axis
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)

    return np.matrix([x, y, z]).T

def getNormalizedInvMatrix(x, y, z):
    return np.linalg.inv(getNormalizedMatrix(x,y,z))


def transformation(new_matrix_inv, old_point):
    return np.matmul(new_matrix_inv, old_point).getA1()

# change points matrix - by reference
def matrixTransformation(new_matrix_inv, points):
    for i in range(len(points)):
        points[i] = transformation(new_matrix_inv, points[i])


# This function returns an angle ranging from 0 to 2*pi.
def aPositiveAngle(a_opposite_side, a_adjacent_side):
    angle = np.arctan2(a_opposite_side, a_adjacent_side)
    angle[angle < 0] += 2 * np.pi
    return angle


# Dividing the vector to the part in the direction of the dislocation
def proj(a, b):
    return np.abs(np.dot(a, b) / np.dot(b, b)) * b



def getNewCoordinateSystem(a_dislocation_vector, a_burgers):
    # we wanted to get rid of one of the dimensions of the burgers vector, because of the assumption
    # in the Theory_of_dislocation book (John Price Hirth, Jens Lothe), that it has only two dimensions.
    # that's why we made a new Coordinate system in which the Z axis is the dislocation's direction,
    # the X axis is the part of the burgers vector which is vertical to the dislocation's direction.
    # the Y axis is the vector that is vertical to the others.
    z = proj(a_burgers, a_dislocation_vector)
    # if the burgers vector and the dislocation vector are perpendicular. In other words, it's just an edge dislocation
    if (z == np.zeros(3)).all():
        z = np.copy(a_dislocation_vector)
        x = np.copy(a_burgers)

    # if the burgers vector and the dislocation vector are parallel. In other words, it's just a screw dislocation
    elif (np.abs(a_burgers / np.linalg.norm(a_burgers)) == np.abs(z / np.linalg.norm(z))).all():
        if a_burgers[X_INDEX] != 0:
            value = -(a_burgers[Z_INDEX] + a_burgers[Y_INDEX]) / a_burgers[X_INDEX]
            x = (value, 1, 1)
        elif a_burgers[Y_INDEX] != 0:
            value = -(a_burgers[Z_INDEX] + a_burgers[X_INDEX]) / a_burgers[Y_INDEX]
            x = (1, value, 1)
        else:
            value = -(a_burgers[X_INDEX] + a_burgers[Y_INDEX]) / a_burgers[Z_INDEX]
            x = (1, 1, value)
    else:
        x = a_burgers - z

    y = np.cross(z, x)

    return getNormalizedMatrix(x, y, z)

