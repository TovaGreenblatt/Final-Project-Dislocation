import numpy as np
import os
import sys
import files_handle
import mathematical_functions as mf
import crystals

# A is the fcc cell dimension.
FCC_LATTICE_CONSTANT = 3.615
POISSONS_RATIO = 0.33
# Enumerators
X_INDEX = 0
Y_INDEX = 1
Z_INDEX = 2


def do(a_n, aa_dislocations, lattice_constant, wrap, folder):
    crystal = crystals.FCC_crystal(lattice_constant, a_n)
    # Dump the initial configuration to a file.
    files_handle.dumpToFile(crystal.aa_atom_locations, crystal.aa_dimensions, folder, 'atoms_perfect_crystal')

    # Shift the atoms to get the array with the dislocations.
    # Each line in the dislocation file describes one dislocation.
    # The first three coordinates are the location of the dislocation line, in \AA (the x coordinate is don't care).
    # The other three coordinates are the Burgers vector, in units of the Burgers vector length of one Burgers vector.

    for a_dislocation in aa_dislocations:
        a_dislocation_line_coordinates = a_dislocation[0:3]
        a_burgers = a_dislocation[3:6]
        a_dislocation_vector = a_dislocation[6:9]
        crystal.add_dislocation(a_dislocation_line_coordinates, a_burgers, a_dislocation_vector)

    # After the atoms have been shifted, some of them might have moved out of the box.
    # If the box is periodic, we have to get them back in.
    # if wrap == True:
    #     aa_atoms_with_negative_overshoot = np.where(aa_atom_locations < aa_dimensions[0])
    #     aa_atoms_with_positive_overshoot = np.where(aa_atom_locations >= aa_dimensions[1])
    #     a_size_of_box = aa_dimensions[1] - aa_dimensions[0]
    #     aa_atom_locations[aa_atoms_with_negative_overshoot] += a_size_of_box[aa_atoms_with_negative_overshoot[1]]
    #     aa_atom_locations[aa_atoms_with_positive_overshoot] -= a_size_of_box[aa_atoms_with_positive_overshoot[1]]
        crystal.add_fcc_dislocation(a_dislocation_line_coordinates, a_burgers, a_dislocation_vector)

    # Write the results in the format that Lammps wants.
    files_handle.dumpToFile(crystal.aa_atom_locations, crystal.aa_dimensions, folder, 'atoms_with_dislocations')


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

    a_n = files_handle.aGetDimensions(folder, 'dimensions.ini')
    aa_dislocations = files_handle.aaGetDislocations(folder, 'dislocations.ini')

    if(num_of_arguments > 2):
        lattice_constant = float(sys.argv[2])
    else:
        lattice_constant = FCC_LATTICE_CONSTANT

    # if(num_of_arguments > 3 and argv[3] == 'wrap'):
    if(num_of_arguments > 3):
        wrap = True
    else:
        wrap = False

    folder = '..//output_files'
    do(a_n, aa_dislocations, lattice_constant, wrap, folder)