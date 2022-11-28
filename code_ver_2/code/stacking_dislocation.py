import sys
import files_handle
import crystals

# A is the fcc cell dimension.
FCC_LATTICE_CONSTANT = 3.615



def do(a_dimensions, aa_dislocations, lattice_constant, wrap, folder):
    # Create fcc crystal
    crystal = crystals.FCC_crystal(lattice_constant, a_dimensions)
    # Dump the initial configuration to a file.
    files_handle.dumpToFile(crystal.get_aa_atom_locations(), crystal.get_aa_dimensions(), folder, 'atoms_perfect_crystal')

    # Shift the atoms to get the array with the dislocations.
    # Each line in the dislocation file describes one dislocation.
    # The first three coordinates are the location of the dislocation line, in \AA (the x coordinate is don't care).
    # The other three coordinates are the Burgers vector, in units of the Burgers vector length of one Burgers vector.
    crystal.add_fcc_dislocations(aa_dislocations)

    if wrap:
        crystal.wrap()

    # Write the results in the format that Lammps wants.
    files_handle.dumpToFile(crystal.get_aa_atom_locations(), crystal.get_aa_dimensions(), folder, 'atoms_with_dislocations')


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

    if num_of_arguments > 2:
        lattice_constant = float(sys.argv[2])
    else:
        lattice_constant = FCC_LATTICE_CONSTANT

    # if(num_of_arguments > 3 and argv[3] == 'wrap'):
    if num_of_arguments > 3:
        wrap = True
    else:
        wrap = False

    folder = '..//output_files'
    do(a_n, aa_dislocations, lattice_constant, True, folder)