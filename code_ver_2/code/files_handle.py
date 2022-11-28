import numpy as np
import os
import constant as const

def dumpToFile(aa_atom_locations, aa_dimensions, folder, file_name):
    os.chdir(folder)
    # The number of atoms must be written to the file.
    num_of_atoms = aa_atom_locations.shape[0]
    # Write into the file.
    f = open(file_name, 'w')
    f.write('Position data for Ni File\n')
    f.write('\n')
    f.write('%u atoms\n' % num_of_atoms)
    f.write('1 atom types\n')
    f.write('%f\t%f\t xlo xhi\n' % (aa_dimensions[0, const.X_INDEX], aa_dimensions[1, const.X_INDEX]))
    f.write('%f\t%f\t ylo yhi\n' % (aa_dimensions[0, const.Y_INDEX], aa_dimensions[1, const.Y_INDEX]))
    f.write('%f\t%f\t zlo zhi\n' % (aa_dimensions[0, const.Z_INDEX], aa_dimensions[1, const.Z_INDEX]))
    f.write('0.0 0.0 0.0 xy xz yz\n')
    f.write('\n')
    f.write('Atoms\n')
    f.write('\n')
    for i, a_atom in enumerate(aa_atom_locations, 1):
        f.write('%u\t 1\t %f\t%f\t%f\n' % (i, a_atom[const.X_INDEX], a_atom[const.Y_INDEX], a_atom[const.Z_INDEX]))
    f.close()
    print('Atom file created.')


# folder ='' --> ??
def aGetDimensions(folder, file_name):
    os.chdir(folder)
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    ns = {'nx':0, 'ny':0, 'nz':0}
    for line in lines:
        exec(line, globals(), ns)
    return np.array([ns['nx'], ns['ny'], ns['nz']])




def aaGetDislocations(folder, file_name):
    os.chdir(folder)
    aa_dislocations = np.loadtxt(file_name)
    return aa_dislocations



#
# if __name__ == '__main__':
#     a = aGetDimensions('..\\init_files', "dimensions.ini")
#     print(a)
#     b = aaGetDislocations("C:\\Users\\tovag\\Desktop\\tester", "dislocations.ini")
#     print(b)
#

