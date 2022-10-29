import numpy as np
import os
import sys


# A is the fcc cell dimension.
FCC_LATTICE_CONSTANT = 3.615
POISSONS_RATIO = 0.33
# Enumerators
X_INDEX = 0
Y_INDEX = 1
Z_INDEX = 2


def do(a_n, aa_dislocations, lattice_constant, wrap):
    # Calculate the constants.
    burgers = lattice_constant / np.sqrt(2)
    # Build one cell.
    aa_unit_cell = lattice_constant * np.array(((0, 0, 0),
                                               (0.5, 0.5, 0),
                                               (0.5, 0, 0.5),
                                               (0, 0.5, 0.5)))
                                                #metrix of 3*4

    num_of_atoms_in_cell = aa_unit_cell.shape[0] #=4
    # Create an array skeleton with nx*ny*nz cells. The number of cells in each dimension must be even.
    # The cells will be indexed from in each axis i from -ni/2 to ni/2 - 1.
    # The values of nx, ny, and nz are taken from an ini file, whose name must be specified in the command line call to the program.
    aa_box_edges = np.column_stack((-a_n / 2, a_n / 2))
    aa_cell_ranges = [np.arange(i[0], i[1]) for i in aa_box_edges]
    aa_cell_indices = np.array([(x, y, z) for x in aa_cell_ranges[X_INDEX] for y in aa_cell_ranges[Y_INDEX] for z in aa_cell_ranges[Z_INDEX]])
    # Calculate the coordinates of each atom.
    a_cell_vectors = lattice_constant * np.array((1, 1, 1))
    aa_cell_locations = aa_cell_indices * a_cell_vectors
    # Convolute to get the whole FCC array. Each line in aa_atom_locations represents on atom. Each column represents the coordinate in one axis.
    aa_atom_locations = np.array([i + j for i in aa_cell_locations for j in aa_unit_cell])
    # Calculate the dimensions of the simulation box.
    aa_dimensions = aa_box_edges.T * a_cell_vectors
    # Dump the initial configuration to a file.
    dumpToFile(aa_atom_locations, aa_dimensions, 'atoms_perfect_crystal')

    # Shift the atoms to get the array with the dislocations.
    # Each line in the dislocation file describes one dislocation.
    # The first three coordinates are the location of the dislocation line, in \AA (the x coordinate is don't care).
    # The other three coordinates are the Burgers vector, in units of the Burgers vector length of one Burgers vector.

    for a_dislocation in aa_dislocations:
        a_dislocation_line_coordinates = a_dislocation[0:3]
        a_burgers = a_dislocation[3:6]
        a_dislocation_vector = a_dislocation[6:9]

        a_burgers = lattice_constant * a_burgers
        a_dislocation_vector = a_dislocation_vector / np.linalg.norm(a_dislocation_vector)

        aa_new_coordinate_system = getNewCoordinateSystem(a_dislocation_vector, a_burgers)
        aa_new_coordinate_system_inv = np.linalg.inv(aa_new_coordinate_system)

        # transformation
        for i in range(len(aa_atom_locations)):
            aa_atom_locations[i] = transformation(aa_new_coordinate_system_inv, aa_atom_locations[i])

        a_dislocation_vector = transformation(aa_new_coordinate_system_inv, a_dislocation_vector)
        a_dislocation_line_coordinates = transformation(aa_new_coordinate_system_inv, a_dislocation_line_coordinates)
        a_burgers = transformation(aa_new_coordinate_system_inv, a_burgers)

        # movements
        aa_atom_locations = aaDislocationByStrain(aa_atom_locations, a_dislocation_line_coordinates, a_burgers)

        # transformation
        for i in range(len(aa_atom_locations)):
            aa_atom_locations[i] = transformation(aa_new_coordinate_system, aa_atom_locations[i])

    # After the atoms have been shifted, some of them might have moved out of the box.
    # If the box is periodic, we have to get them back in.
    if wrap == True:
        aa_atoms_with_negative_overshoot = np.where(aa_atom_locations < aa_dimensions[0])
        aa_atoms_with_positive_overshoot = np.where(aa_atom_locations >= aa_dimensions[1])
        a_size_of_box = aa_dimensions[1] - aa_dimensions[0]
        aa_atom_locations[aa_atoms_with_negative_overshoot] += a_size_of_box[aa_atoms_with_negative_overshoot[1]]
        aa_atom_locations[aa_atoms_with_positive_overshoot] -= a_size_of_box[aa_atoms_with_positive_overshoot[1]]
    # Write the results in the format that Lammps wants.
    dumpToFile(aa_atom_locations, aa_dimensions, 'atoms_with_dislocations')




# This function creates a dislocation line in the positive x direction, on the Thompson tetrahedron ABC plane.
# The x component of the Burgers vector is the screw component. The z component is the edge component.
def aaDislocationByStrain(aa_atom_locations, a_dislocation_line_coordinates, a_burgers):
    # Find the location of each atom relative to the dislocation line.
    # a_y is the distance of each atom from the dislocation line in the y axis, and a_z is the distance of each atom from the dislocation line in the z axis.
    # Their short names are for the sake of brevity in the equations.
    a_y = aa_atom_locations[:, Y_INDEX] - a_dislocation_line_coordinates[Y_INDEX]
    a_x = aa_atom_locations[:, X_INDEX] - a_dislocation_line_coordinates[X_INDEX]
    # Calculate the displacement of each atom.
    aa_displacements = np.zeros(aa_atom_locations.shape)

    # Calculate the displacement due to the edge and screw components.
    # The formula for the displacement of each atom in an edge dislocation is taken from Hirth p. 78.
    if(a_burgers[X_INDEX] != 0):
        aa_displacements[:, X_INDEX] = a_burgers[X_INDEX] / 2 / np.pi * (aPositiveAngle(a_y, a_x) + a_x * a_y / 2 / (1 - POISSONS_RATIO) / (a_x**2 + a_y**2))
        aa_displacements[:, Y_INDEX] = -a_burgers[X_INDEX] / 2 / np.pi \
                                       * ((1 - 2 * POISSONS_RATIO) / 4 / (1 - POISSONS_RATIO) * np.log(a_x**2 + a_y**2) + (a_x**2 - a_y**2) / 4 / (1 - POISSONS_RATIO) / (a_x**2 + a_y**2))
    # The formula for the displacement of each atom in a screw dislocation is taken from Hirth p. 60.
    if(a_burgers[Z_INDEX] != 0):
        aa_displacements[:, Z_INDEX] = a_burgers[Z_INDEX] / 2 / np.pi * aPositiveAngle(a_y, a_x)
    # Add the displacements back to the atom locations.
    aa_atom_locations += aa_displacements
    return aa_atom_locations




def dumpToFile(aa_atom_locations, aa_dimensions, s_name):

    # The number of atoms must be written to the file.
    num_of_atoms = aa_atom_locations.shape[0]
    # Write into the file.
    f = open(s_name, 'w')
    f.write('Position data for Ni File\n')
    f.write('\n')
    f.write('%u atoms\n' % num_of_atoms)
    f.write('1 atom types\n')
    f.write('%f\t%f\t xlo xhi\n' % (aa_dimensions[0, X_INDEX], aa_dimensions[1, X_INDEX]))
    f.write('%f\t%f\t ylo yhi\n' % (aa_dimensions[0, Y_INDEX], aa_dimensions[1, Y_INDEX]))
    f.write('%f\t%f\t zlo zhi\n' % (aa_dimensions[0, Z_INDEX], aa_dimensions[1, Z_INDEX]))
    f.write('0.0 0.0 0.0 xy xz yz\n')
    f.write('\n')
    f.write('Atoms\n')
    f.write('\n')
    for i, a_atom in enumerate(aa_atom_locations, 1):
        f.write('%u\t 1\t %f\t%f\t%f\n' % (i, a_atom[X_INDEX], a_atom[Y_INDEX], a_atom[Z_INDEX]))
    f.close()
    print('Atom file created.')




def aGetDimensions():
    f = open('dimensions.ini', 'r')
    lines = f.readlines()
    f.close()
    ns = {'nx':0, 'ny':0, 'nz':0}
    for line in lines:
        exec(line, globals(), ns)
    return np.array([ns['nx'], ns['ny'], ns['nz']])




def aaGetDislocations():
    aa_dislocations = np.loadtxt('dislocations.ini')
    return aa_dislocations


# This function returns an angle ranging from 0 to 2*pi.
def aPositiveAngle(a_opposite_side, a_adjacent_side):
    angle = np.arctan2(a_opposite_side, a_adjacent_side)
    angle[angle < 0] += 2 * np.pi
    return angle


# Dividing the vector to the part in the direction of the dislocation
def proj(a, b):
    return np.abs(np.dot(a, b) / np.dot(b, b)) * b


def transformation(new_matrix_inv, old_point):
    return np.matmul(new_matrix_inv, old_point).getA1()

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

    # normalization the axis
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    z /= np.linalg.norm(z)

    return np.matrix([x, y, z]).T

if __name__ == '__main__':

    num_of_arguments = len(sys.argv)

    if num_of_arguments > 3:
        print('create_one_dislocation.py usage: create_one_dislocation.py <folder> [lattice_constant] [wrap]')
        print('<folder> should contain one file called dimensions.ini, and one file called dislocations.ini.')
        print('[lattice_constant] is an optional parameter and should be a number.')
        print('[wrap] is an optional parameter.')
        print('It should be "wrap" if the simulation box is to be periodic.')
        exit()
    folder = sys.argv[1]
    os.chdir(folder)
    a_n = aGetDimensions()
    aa_dislocations = aaGetDislocations()

    if(num_of_arguments > 2):
        lattice_constant = float(sys.argv[2])
    else:
        lattice_constant = FCC_LATTICE_CONSTANT

    # if(num_of_arguments > 3 and argv[3] == 'wrap'):
    if(num_of_arguments > 3):
        wrap = True
    else:
        wrap = False
    do(a_n, aa_dislocations, lattice_constant, wrap)