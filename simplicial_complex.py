'''
Author: Ryan Oakley (roakley7)
'''

import itertools
import numpy as np

def f2_bases(matrix):
    '''
    puts a matrix over F_2 into normal form, removing linearly dependent columns and computing the basis for both image and kernel
    '''
    new_mat = matrix.T
    new_mat = np.array(sorted(new_mat, key=lambda x: list(x).index(1)))
    # outputs = np.array([[1 if col==idx else 0 for idx in range(new_mat.shape[0])] for col in range(new_mat.shape[0])])
    outputs = np.identity(new_mat.shape[0])
    for row_idx, row in enumerate(new_mat):
        pivot = list(row).index(1)
        for above_idx in range(0,row_idx):
            if new_mat[above_idx, pivot] == 1:
                outputs[above_idx] += outputs[row_idx]
                new_mat[above_idx] += row
                new_mat[above_idx] = new_mat[above_idx] % 2

    Z_basis = outputs[new_mat.sum(axis=1) == 0].T # basis for kernel of del
    B_basis = outputs[new_mat.sum(axis=1) != 0].T # basis for image of del
    new_mat = new_mat[new_mat.sum(axis=1) != 0].T
    
    return Z_basis, B_basis
        

class SimplicialComplex:

    def __init__(self, vertices=None):
        '''
        self.faces[dim_number] gives a list of faces of that dimension
        '''
        self.faces = dict()
        self.faces[0] = [] if vertices == None else vertices
        self.dimension = 0

    def euler_char(self):
        sign = 1
        characteristic = 0
        for dim in range(self.dimension + 1):
            characteristic += sign * len(self.get_faces_dim(dim))
            sign *= -1
        
        return characteristic
                   
    def _dangerous_add_face(self, face):
        """
        WARNING: only use if you know for sure that the subfaces already exist and that this is a new face :)
        """
        dim = len(face)-1
        self.dimension = max(self.dimension, dim)

        if dim not in self.faces:
            self.faces[dim] = []
        self.faces[dim].append(face)

    def add_face(self, face):
        '''
        add a given set of vertices as a face
        will ensure that all subsets are added to the complex
        '''
        dim = len(face)-1
        self.dimension = max(self.dimension, dim)

        for subset_dim in range(0,dim+1):
            if subset_dim not in self.faces:
                self.faces[subset_dim] = []
                
            for subset_elems in itertools.combinations(face, subset_dim+1):
                subset = set(subset_elems)
            
                if subset not in self.faces[subset_dim]:
                    self.faces[subset_dim].append(subset)

    def get_faces_dim(self, dim):
        '''
        returns the faces of a requested dimension
        '''
        if not (0 <= dim <= self.dimension):
            return []
        return self.faces[dim]

    def get_vertices(self):
        '''
        return the vertices in as singleton sets
        '''
        return self.get_faces_dim(0)
    
    def get_faces_vert(self, vert):
        '''
        returns a dict of faces that include the given vertex
        '''
        return self.get_faces_face(vert)

    def get_faces_face(self, face):
        '''
        returns a dict of faces that include the given face
        '''
        faces_of_interest = dict()
        base_dim = len(face) - 1
        if face in self.get_faces_dim(base_dim):
            for dim in range(base_dim+1, self.dimension+1):
                faces_of_interest[dim] = [superface for superface in self.get_faces_dim(dim) if face.issubset(superface)]
        return faces_of_interest

    def remove_vertex(self, vert):
        '''
        removes a vertex while preseving the ASC structure
        vert: the vertex to remove, passed as a singleton set
        '''
        if vert in self.get_vertices():
            for dim in range(1, self.dimension+1):
                self.faces[dim] = [face - vert for face in self.faces[dim]]


    def get_boundary_matrix(self, dim):
        assert 0 <= dim
        if dim == 0:
            return np.zeros((1,len(self.get_vertices())))
        elif dim > self.dimension:
            return np.zeros((1,1))
        rows = self.get_faces_dim(dim-1)
        cols = self.get_faces_dim(dim)
        bnd_mat = []
        for sub_face in rows:
            bnd_mat.append([1 if sub_face.issubset(super_face) else 0 for super_face in cols])
            
        return np.array(bnd_mat)

    def get_cycles(self, dim):
        if dim == 0:
            return np.identity(len(self.get_vertices()))
        bnd_mat = self.get_boundary_matrix(dim)
        return f2_bases(bnd_mat)[0]

    def display_cycles(self, dim):
        kernel_basis = self.get_cycles(dim)
        print(f"Basis of Ker(d_{dim}), i.e. cycles in C_{dim}:")
        for i, col in enumerate(kernel_basis.T):
            vect = [self.get_faces_dim(dim)[i] for i in range(len(col)) if col[i] == 1]
            print(f"\t{i+1}:\t{vect}")

    def get_boundaries(self, dim):
        if dim > self.dimension:
            return np.array([[]])
        bnd_mat = self.get_boundary_matrix(dim)
        return np.matmul(bnd_mat, f2_bases(bnd_mat)[1])
    
    def display_boundaries(self, dim):
        img_basis = self.get_boundaries(dim)
        print(f"Basis of Img(d_{dim}), i.e. boundaries in C_{dim-1}:")
        for i, col in enumerate(img_basis.T):
            vect = [self.get_faces_dim(dim-1)[i] for i in range(len(col)) if col[i] == 1]
            print(f"\t{i+1}:\t{vect}")

    def get_homologies(self, dim):
        if dim == self.dimension:
            return self.get_cycles(dim)
        else:
            basis = []
            base_boundaries = self.get_boundaries(dim+1).T
            for cycle in self.get_cycles(dim).T:
                if f2_bases(np.array([r for r in base_boundaries] + [cycle]))[0].shape[1] == 0:
                    basis.append(cycle)
            return np.array(basis).T
                    

    def display_homologies(self, dim):
        hom_basis = self.get_homologies(dim)
        print(f"Basis of H_{dim}, i.e. cycles that aren't boundaries in C_{dim}:")
        for i, col in enumerate(hom_basis.T):
            vect = [self.get_faces_dim(dim-1)[i] for i in range(len(col)) if col[i] == 1]
            print(f"\t{i+1}:\t{vect}")
            
    def get_homology_rank(self, dim):
        if dim == self.dimension:
            return self.get_cycles(dim).shape[1] - 0 
        else:
            return self.get_cycles(dim).shape[1] - self.get_boundaries(dim+1).shape[1]

    def display_faces(self):
        '''
        prints the faces of the simplex in a reasonably readable way
        '''
        for dim in range(self.dimension+1):
            print(f'Dimension {dim}:')
            for face in self.get_faces_dim(dim):
                print(f'\t{face}')
        print()

    def display_facets(self):
        '''
        prints the facets (i.e. maximal faces) of the simplex in the same was as `display faces`
        '''
        facets = {self.dimension : self.get_faces_dim(self.dimension)}
        for src_dim in range(self.dimension):
            facets[src_dim] = []
            dst_faces = self.get_faces_dim(src_dim + 1)
            for src_face in self.get_faces_dim(src_dim):
                if not any([src_face.issubset(face) for face in dst_faces]):
                    facets[src_dim].append(src_face)
        
        for dim in range(self.dimension+1):
            print(f'Dimension {dim}:')
            for face in facets[dim]:
                print(f'\t{face}')
        print()
