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
    burgers = lattice_constant / np.sqrt(2)  #
    # Build one cell.
    # This unit cell contains six atoms, and has the following unit vectors directions (specified in the following in the fcc coordinate system):
    # (-1, 1, 0), (1, 1, 1), (1, 1, -2).
    # This is the same idea as, but slightly different in detail, from the axes in ashkenazy03.
    # This means that, on the Thompson tetrahedron, the x axis is AB, the y axis is D\delta, and the z axis is \delta{}C.
    # Note that this is when the Thompson tetrahedron is defined with D at the origin, as in Hirth p. 296.
    # (Other definitions, such as that by H. Foll, have it differently.)
    # We'll work with the ABC slip plane, with the dislocation lines along the x axis.
    # This means that for any dislocation, the screw component of the Burgers vector is the x component, and the edge component of the Burgers vector is the z component.
    # The Burgers vector will not have a y component. There can be perfect screw dislocations, but no perfect edge dislocations, in the slip system.
    aa_unit_cell = lattice_constant * np.array(((1 / 2 / np.sqrt(2), 0, 1 / 2 / np.sqrt(6) * 3), 
                                                    (1 / 2 / np.sqrt(2), 1 / np.sqrt(3), 5 * np.sqrt(6) / 12), 
                                                    (0, 0, 0), 
                                                    (0, 1 / np.sqrt(3), 1 / np.sqrt(6)), 
                                                    (1 / 2 / np.sqrt(2), 2 / np.sqrt(3), 1 / 2 / np.sqrt(6)), 
                                                    (0, 2 / np.sqrt(3), 2 / np.sqrt(6)))) 
                                                    #metrix of 3*6

    num_of_atoms_in_cell = aa_unit_cell.shape[0] #=6
    # Create an array skeleton with nx*ny*nz cells. The number of cells in each dimension must be even.
    # The cells will be indexed from in each axis i from -ni/2 to ni/2 - 1.
    # The values of nx, ny, and nz are taken from an ini file, whose name must be specified in the command line call to the program.
    aa_box_edges = np.column_stack((-a_n / 2, a_n / 2))
    aa_cell_ranges = [np.arange(i[0], i[1]) for i in aa_box_edges]
    aa_cell_indices = np.array([(x, y, z) for x in aa_cell_ranges[X_INDEX] for y in aa_cell_ranges[Y_INDEX] for z in aa_cell_ranges[Z_INDEX]])
    # Calculate the coordinates of each atom.
    a_cell_vectors = lattice_constant * np.array((1 / np.sqrt(2), np.sqrt(3), 3 / np.sqrt(6)))
    aa_cell_locations = aa_cell_indices * a_cell_vectors
    # Convolute to get the whole FCC array. Each line in aa_atom_locations represents on atom. Each column represents the coordinate in one axis.
    aa_atom_locations = np.array([i + j for i in aa_cell_locations for j in aa_unit_cell])
    # Calculate the dimensions of the simulation box.
    aa_dimensions = aa_box_edges.T * a_cell_vectors
    # Dump the initial configuration to a file.
    dumpToFile(aa_atom_locations, aa_dimensions, 'atoms_perfect_crystal')
    # Shift the atoms to get the array with the dislocations.
    # Each line in the dislocation file describes one dislocation.
    # The first three coordinates are the location of the dislcation line, in \AA (the x coordinate is don't care).
    # The other three coordinates are the Burgers vector, in units of the Burgers vector length of one Burgers vector.
    for a_dislocation in aa_dislocations:
        a_dislocation_line_coordinates = a_dislocation[0:3]
        a_burgers = burgers * a_dislocation[3:6]
        print(a_dislocation_line_coordinates)
        print(a_burgers)
        aa_atom_locations = aaDislocationByStrain(aa_atom_locations, a_dislocation_line_coordinates, a_burgers)
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
    a_z = aa_atom_locations[:, Z_INDEX] - a_dislocation_line_coordinates[Z_INDEX]
    # Calculate the displacement of each atom.
    aa_displacements = np.zeros(aa_atom_locations.shape)
    # Calculate the displacement due to the edge and screw components.
    # The formula for the displacement of each atom in an edge dislocation is taken from Hirth p. 78.
    aa_displacements[:, Z_INDEX] = a_burgers[Z_INDEX] / 2 / np.pi * (aPositiveAngle(a_y, a_z) + a_z * a_y / 2 / (1 - POISSONS_RATIO) / (a_z**2 + a_y**2))
    aa_displacements[:, Y_INDEX] = -a_burgers[Z_INDEX] / 2 / np.pi \
                                   * ((1 - 2 * POISSONS_RATIO) / 4 / (1 - POISSONS_RATIO) * np.log(a_z**2 + a_y**2) + (a_z**2 - a_y**2) / 4 / (1 - POISSONS_RATIO) / (a_z**2 + a_y**2))
    # The formula for the displacement of each atom in a screw dislocation is taken from Hirth p. 60.
    aa_displacements[:, X_INDEX] = a_burgers[X_INDEX] / 2 / np.pi * aPositiveAngle(a_y, a_z)
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
              
    if(num_of_arguments > 3 and argv[3] == 'wrap'):
        wrap = True
    else:
        wrap = False    
    do(a_n, aa_dislocations, lattice_constant, wrap)
