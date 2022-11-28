from abc import ABC, abstractmethod
import numpy as np
import mathematical_functions as mf
import constant as const


class crystal(ABC):
    _aa_dimensions = None
    _aa_atom_locations = None
    lattice_constant = None

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def add_dislocation(self):
        pass

    # After the atoms have been shifted, some of them might have moved out of the box.
    # If the box is periodic, we have to get them back in.
    def wrap(self):
        aa_atoms_with_negative_overshoot = np.where(self._aa_atom_locations < self._aa_dimensions[0])
        aa_atoms_with_positive_overshoot = np.where(self._aa_atom_locations >= self._aa_dimensions[1])
        a_size_of_box = self._aa_dimensions[1] - self._aa_dimensions[0]
        self._aa_atom_locations[aa_atoms_with_negative_overshoot] += a_size_of_box[aa_atoms_with_negative_overshoot[1]]
        self._aa_atom_locations[aa_atoms_with_positive_overshoot] -= a_size_of_box[aa_atoms_with_positive_overshoot[1]]

    # getter method
    def get_aa_atom_locations(self):
        return self._aa_atom_locations

    def get_aa_dimensions(self):
        return self._aa_dimensions


class FCC_crystal(crystal):
    def __init__(self, lattice_constant, a_dimensions):
        super().__init__()
        self.lattice_constant = lattice_constant
        self._aa_dimensions = a_dimensions
        self.build()

    # overriding abstract method
    def build(self):
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
        aa_unit_cell = self.lattice_constant * np.array(
            ((1 / 2 / np.sqrt(2), 0, 1 / 2 / np.sqrt(6) * 3),  # (0,0.5,-0.5)
             (1 / 2 / np.sqrt(2), 1 / np.sqrt(3), 5 * np.sqrt(6) / 12),
             # (0.5,1,-0.5)
             (0, 0, 0),  # (0,0,0)
             (0, 1 / np.sqrt(3), 1 / np.sqrt(6)),  # (0.5,0.5,0)
             (1 / 2 / np.sqrt(2), 2 / np.sqrt(3), 1 / 2 / np.sqrt(6)),
             # (0.5,1,0.5)
             (0, 2 / np.sqrt(3), 2 / np.sqrt(6))))  # (1,1,0)
        # metrix of 3*6

        num_of_atoms_in_cell = aa_unit_cell.shape[0]  # =6
        # Create an array skeleton with nx*ny*nz cells. The number of cells in each dimension must be even.
        # The cells will be indexed from in each axis i from -ni/2 to ni/2 - 1.
        # The values of nx, ny, and nz are taken from an ini file, whose name must be specified in the command line call to the program.
        aa_box_edges = np.column_stack((-self._aa_dimensions / 2, self._aa_dimensions / 2))
        aa_cell_ranges = [np.arange(i[0], i[1]) for i in aa_box_edges]
        aa_cell_indices = np.array([(x, y, z) for x in aa_cell_ranges[const.X_INDEX] for y in aa_cell_ranges[const.Y_INDEX] for z in
                                    aa_cell_ranges[const.Z_INDEX]])
        # Calculate the coordinates of each atom.
        a_cell_vectors = self.lattice_constant * np.array((1 / np.sqrt(2), np.sqrt(3), 3 / np.sqrt(6)))
        aa_cell_locations = aa_cell_indices * a_cell_vectors
        # Convolute to get the whole FCC array. Each line in aa_atom_locations represents on atom. Each column represents the coordinate in one axis.
        self._aa_atom_locations = np.array([i + j for i in aa_cell_locations for j in aa_unit_cell])
        # Calculate the dimensions of the simulation box.
        self._aa_dimensions = aa_box_edges.T * a_cell_vectors

    def add_fcc_dislocation(self, a_dislocation_line_coordinates, a_dislocation_vector, a_burgers):
        aa_new_coordinate_system_inv = self.getStackingCoordinateSystemInv()
        # a_dislocation_line_coordinates = mf.transformation(aa_new_coordinate_system_inv, a_dislocation_line_coordinates)
        a_dislocation_vector = mf.transformation(aa_new_coordinate_system_inv, a_dislocation_vector)
        a_burgers = mf.transformation(aa_new_coordinate_system_inv, a_burgers)
        self.add_dislocation(a_dislocation_line_coordinates, a_dislocation_vector, a_burgers)

    def add_dislocation(self, a_dislocation_line_coordinates, a_dislocation_vector, a_burgers):
        print("add_dislocation")
        a_burgers = self.lattice_constant * a_burgers

        aa_new_coordinate_system = mf.getNewCoordinateSystem(a_dislocation_vector, a_burgers)
        aa_new_coordinate_system_inv = np.linalg.inv(aa_new_coordinate_system)

        # transformation
        mf.matrixTransformation(aa_new_coordinate_system_inv, self._aa_atom_locations)

        a_dislocation_line_coordinates = mf.transformation(aa_new_coordinate_system_inv, a_dislocation_line_coordinates)
        a_burgers = mf.transformation(aa_new_coordinate_system_inv, a_burgers)

        # movements
        self.aaDislocationByStrain(a_dislocation_line_coordinates, a_burgers, )

        # transformation
        mf.matrixTransformation(aa_new_coordinate_system, self._aa_atom_locations)

    def add_fcc_dislocations(self, aa_dislocations):
        for a_dislocation in aa_dislocations:
            self.add_fcc_dislocation(a_dislocation[0:3], a_dislocation[3:6], a_dislocation[6:9])

    def add_dislocations(self, aa_dislocations):
        for a_dislocation in aa_dislocations:
            self.add_dislocation(a_dislocation[0:3], a_dislocation[3:6], a_dislocation[6:9])

    # This function creates a dislocation line in the positive z direction, on the Thompson tetrahedron ABC plane.
    # The x component of the Burgers vector is the screw component. The z component is the edge component.
    def aaDislocationByStrain(self, a_dislocation_line_coordinates, a_burgers, poissons_ratio=const.COPPER_POISSONS_RATIO):
        # Find the location of each atom relative to the dislocation line.
        # a_y is the distance of each atom from the dislocation line in the y axis, and a_z is the distance of each atom from the dislocation line in the z axis.
        # Their short names are for the sake of brevity in the equations.
        a_y = self._aa_atom_locations[:, const.Y_INDEX] - a_dislocation_line_coordinates[const.Y_INDEX]
        a_x = self._aa_atom_locations[:, const.X_INDEX] - a_dislocation_line_coordinates[const.X_INDEX]
        # Calculate the displacement of each atom.
        aa_displacements = np.zeros(self._aa_atom_locations.shape)

        # Calculate the displacement due to the edge and screw components.
        # The formula for the displacement of each atom in an edge dislocation is taken from Hirth p. 78.
        if (a_burgers[const.X_INDEX] != 0):
            aa_displacements[:, const.X_INDEX] = a_burgers[const.X_INDEX] / 2 / np.pi * (
                    mf.aPositiveAngle(a_y, a_x) + a_x * a_y / 2 / (1 - poissons_ratio) / (a_x ** 2 + a_y ** 2))
            aa_displacements[:, const.Y_INDEX] = -a_burgers[const.X_INDEX] / 2 / np.pi \
                                           * ((1 - 2 * poissons_ratio) / 4 / (1 - poissons_ratio) * np.log(
                a_x ** 2 + a_y ** 2) + (a_x ** 2 - a_y ** 2) / 4 / (1 - poissons_ratio) / (a_x ** 2 + a_y ** 2))
        # The formula for the displacement of each atom in a screw dislocation is taken from Hirth p. 60.
        if (a_burgers[const.Z_INDEX] != 0):
            aa_displacements[:, const.Z_INDEX] = a_burgers[const.Z_INDEX] / 2 / np.pi * mf.aPositiveAngle(a_y, a_x)
        # Add the displacements back to the atom locations.
        self._aa_atom_locations += aa_displacements

    def getStackingCoordinateSystemInv(self):
        x = np.array((-1, 1, 0))
        y = np.array((1, 1, 1))
        z = np.array((1, 1, -2))
        return mf.getNormalizedInvMatrix(x, y, z)
