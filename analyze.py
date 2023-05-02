import pandas as pd
import numpy as np
from queue import PriorityQueue
from simplicial_complex import SimplicialComplex
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

def euc_metric(x, y):
    return sum(pow(x - y, 2))

def discrete_metric(x,y):
    return 1 if (x == y).all() else 0

def voronoi_neighbors(x, candidates):
    neighbors = []
    for c_test in candidates:
        # constants for eqn of line between x and c_test: l0 + l*pos = p for pos in [0,2]
        l0 = x
        l = (c_test - x) / 2
        neighbor = True
        for c_comp in candidates:
            if (c_test == c_comp).all():
                continue
            # constants for eqn of hyperplane between x and c_test: (p - p0)\cdot n
            p0 = (x + c_comp) / 2
            n = (c_comp - x) / 2

            # Where the hyperplane intersects the segment between x and c_test
            pos = np.dot(p0 - l0, n) / np.dot(l, n)
            if 0 < pos < 1:
                neighbor = False
                break
        if neighbor:
            neighbors.append(c_test)
    return neighbors

class Filtration:

    def __init__(self, data_arr, met_func=discrete_metric, alpha=False):
        self.metric = met_func
        self.data_arr = data_arr

        self.point_to_index = dict()
        for i, row in enumerate(self.data_arr):
            self.point_to_index[tuple(row)] = i

        self.index_to_point = dict()
        for i, row in enumerate(self.data_arr):
            self.index_to_point[i] = tuple(row)

        self.simp_complex = SimplicialComplex([{tuple(row)} for row in self.data_arr])
        self.__calc_dists(alpha=alpha)
        self.alpha = alpha
        
        self.face_list = [self.face_to_indices({tuple(row)}) for row in self.data_arr]
        self.thresholds = [0] # (inculsive index, exclusive index), threshold
        self.thresh_dict = dict()
        for k in range(len(self.face_list)):
            self.thresh_dict[k] = 0

    def __calc_dists(self, alpha=False):
        self.alpha = alpha
        pq = PriorityQueue()
        if alpha:
            for src_idx in range(len(self.data_arr)):
                src = self.data_arr[src_idx]
                for dst in voronoi_neighbors(src, self.data_arr[[idx for idx in range(len(self.data_arr)) if idx != src_idx]]):
                    curr_dist = self.metric(src, dst)
                    pq.put((curr_dist, {tuple(src), tuple(dst)}))
        else: #VRips
            for src, dst in itertools.combinations(self.data_arr, 2):
                curr_dist = self.metric(src, dst)
                pq.put((curr_dist, {tuple(src), tuple(dst)}))

        self.dist_list = []
        while not pq.empty():
            elem = pq.get()
            if elem not in self.dist_list:
                self.dist_list.append(elem)

    def face_to_indices(self, face):
        new_face = set()
        for point in face:
            new_face.add(self.point_to_index[tuple(point)])
        return new_face

    def set_metric(self, met_func):
        self.metric = met_func

    def recompute(self, alpha=False):
        self.__calc_dists(alpha=alpha)

    def get_bounds(self):
        return self.dist_list[0][0], self.dist_list[-1][0]

    def get_dists(self):
        return [x for x,_ in self.dist_list]


    def create_simplex(self, threshold, max_dim=None, cut_idx=None):
        if cut_idx is None:
            cut_idx = 0
            while (cut_idx < len(self.dist_list)) and ((self.dist_list[cut_idx][0] / 2) <= threshold):
                cut_idx += 1

        lines = [line for _, line in self.dist_list[:cut_idx]]
        
        curr_dim = 2
        # [f for f in lines if f not in self.simp_complex.get_faces_dim(1)]
        next_faces = [] 
        idx = -1
        while (idx + len(lines) > 0) and (lines[idx] not in self.simp_complex.get_faces_dim(1)):
            next_faces.append(lines[idx])
            idx -= 1
        
        start_idx = len(self.face_list)
        for line in next_faces:
            self.simp_complex._dangerous_add_face(line)
            
        while True:
            subfaces = self.simp_complex.get_faces_dim(curr_dim - 1)
            new_faces = next_faces
            next_faces = []
            
            new_faces_p = []
            for face in new_faces:
                new_face = self.face_to_indices(face)
                if (new_face not in self.face_list) and (new_face not in new_faces_p):
                    new_faces_p.append(new_face)
            self.face_list.extend(new_faces_p)
            
            if (len(new_faces) == 0) or ((max_dim is not None) and (curr_dim > max_dim)):
                break
                
            # if 2 faces only differ in one place and the differing vertices are connected by a 1-simplex,
            #      then the add the union of the 2 faces as a curr_dim+1 face
            for f1, f2 in itertools.product(subfaces, new_faces):
                check_pair = f1.symmetric_difference(f2)
                union_face = f1.union(f2)
                if (check_pair in lines) and (union_face not in next_faces):
                    self.simp_complex._dangerous_add_face(union_face)
                    next_faces.append(union_face)
            curr_dim += 1
        
        self.thresholds.append(threshold)
        for k in range(start_idx, len(self.face_list)):
            self.thresh_dict[k] = threshold
        


    def filter(self, max_dim=None):
        dists = self.get_dists()
        # add delta to avoid rounding errors as much as possible
        delta = np.min([(dists[i+1]/2 - dists[i]/2)/256 for i in range(len(dists)-1)])
        euler_history = [self.simp_complex.euler_char()]
        for threshold in tqdm([(x/2) + delta for x in dists]):
            # print(list(map(len, [self.simp_complex.get_faces_dim(i) for i in range(self.simp_complex.dimension+1)])))
            self.create_simplex(threshold, max_dim=max_dim)
            euler_history.append(self.simp_complex.euler_char())
            
        return euler_history


    def compute_persistence(self):           
            
        # Create bnd matrix across filtration, taking advantage of python's arbitrary integer sizes and bitops
        #       Lowest 1 in a col in col_list is the highest bit in the int
        print(f"Creating {len(self.face_list)}x{len(self.face_list)} boundary matrix...")
        full_dim = len(self.face_list)
        pivot_cols = dict()
        for pivot in range(len(self.face_list)):
            pivot_cols[pivot] = []
            
        col_list = []
        for col_idx, col_face in tqdm(list(enumerate(self.face_list))):
            res = 0
            for row_idx in range(col_idx):
                res += 1 << row_idx if (self.face_list[row_idx].issubset(col_face) and ((len(self.face_list[row_idx]) + 1) == len(col_face))) else 0
            pivot = len(bin(res)) - 2 - 1 # row index (0-based) of pivot
            if res != 0:
                pivot_cols[pivot].append(col_idx)
            col_list.append(res)
        
        # Reduce bnd mat with col operations
        print("Reducing boundary matrix...")
        face_tracker = [1 << i for i in range(full_dim)] # tracks which columns are combined
        for col_idx in tqdm(range(1,full_dim)):
            col_dim = len(self.face_list[col_idx])
            if col_list[col_idx] != 0:
                assert col_idx in pivot_cols[len(bin(col_list[col_idx])) - 2 - 1]
            # While there are previous shared pivot positions
            while True:
                done = True
                curr_pivot = len(bin(col_list[col_idx])) - 2 - 1
                if bin(col_list[col_idx])[2] == '0': # this means the whole column is 0
                    break
                
                for pivot_idx in pivot_cols[curr_pivot]:
                    # Make sure addition only happens within same-dim chain complexes to the left of the current column
                    if (len(self.face_list[pivot_idx]) != col_dim) or (pivot_idx >= col_idx):
                        continue
                    
                    col_list[col_idx] = col_list[col_idx] ^ col_list[pivot_idx]
                    face_tracker[col_idx] = face_tracker[col_idx] ^ face_tracker[pivot_idx]
                    if col_idx in pivot_cols[curr_pivot]:
                        pivot_cols[curr_pivot].remove(col_idx)
                    done = False
                    break
                
                if done and (col_list[col_idx] != 0) and (col_idx not in pivot_cols[curr_pivot]):
                    pivot_cols[curr_pivot].append(col_idx)
                    break
                    
                if done or (col_list[col_idx] == 0):
                    break
                
        print("Extracting persistences from reduced boundary matrix...")
        # Read off persistences
        lifetimes = []
        for col_idx, col in tqdm(list(enumerate(col_list))):
            if col != 0:
                # column doesn't represent a feature
                continue
            
            # Translate back into faces
            offset = -1
            col_bin = bin(face_tracker[col_idx])[2:]
            base_idx = len(col_bin) - 1
            faces = []
            while True:
                try:
                    offset = col_bin.index('1', offset + 1)
                    row_idx = base_idx - offset
                    faces.append(self.face_list[row_idx])
                except ValueError as e:
                    break
            #faces = [self.face_list[row_idx] for row_idx in range(len(bin(face_tracker[col_idx])) - 2) if (face_tracker[col_idx] & (1 << row_idx)) != 0]
            
            # Since column is a feature, find its birth and death time
            birth = self.thresh_dict[col_idx]
                
            # find pivot (which may not exist) in same row as col of feature
            if len(pivot_cols[col_idx]) != 0:
                # assert len(pivot_cols[col_idx]) == 1
                death = self.thresh_dict[pivot_cols[col_idx][0]]
            else:
                death = self.thresholds[-1]
                
            lifetimes.append(((birth, death), faces))
            
        return lifetimes


def characteristic(homology_ranks):
        sign = 1
        characteristic = 0
        for rank in homology_ranks:
            characteristic += sign * rank
            sign *= -1
        
        return characteristic

def main():
    max_dim = 2
    for fname in [f"CDHWdata_{i+1}.csv" for i in range(5)]:
        print(f"\n{fname}:")
        data_arr = pd.read_csv(fname, index_col="partno").to_numpy()
        filt = Filtration(data_arr, met_func=euc_metric, alpha=True)

        print("Creating filtration...")
        euler_history = filt.filter(max_dim=max_dim)
        
        print("Calculating homologies...")
        all_lifetimes = filt.compute_persistence()
        
        idx = 0
        for threshold in filt.thresholds[:-1]:
            homology_ranks = []
            features = [f for l, f in all_lifetimes if l[0] <= threshold < l[1]]
            for dim in range(max_dim+1):
                homology_ranks.append(len([f for f in features if len(f[0]) == (dim + 1)]))

            b_char = characteristic(homology_ranks)
            assert euler_history[idx] == b_char
            # print(f"characteristic (Simplicial Complex) = {euler_history[idx]}")
            # print(f"characteristic (Betti Numbers) = {b_char}")
            # print()
            idx += 1

        plt.plot([threshold for threshold in filt.thresholds], euler_history)
        plt.xlim((0, filt.thresholds[-1]))
        plt.title("Euler Characteristic")
        plt.xlabel("Filtration Threshold")
        plt.savefig(f"./CDHWdata_figs/{fname[-5]}/euler_{fname[:-4]}.png")
        plt.close()

        f_max = filt.thresholds[-1] * 1.05
        
        for dim in range(2+1): 
            print(f"Calculating H{dim} persistence...")
            lifetimes = [(l, f) for l, f in all_lifetimes if (len(f[0]) - 1) == dim]
            with open(f"./CDHWdata_figs/{fname[-5]}/lifetimes_{dim}.txt", 'w') as f:
                for life, faces in sorted(lifetimes, key=lambda x: x[0][1] - x[0][0], reverse=True):
                    if not np.allclose(life[0], life[1]):
                        f.write(f"{life}:{faces}\n")

            plt.scatter([x[0][0] for x in lifetimes], [y[0][1] for y in lifetimes], s=3)
            plt.xlabel("Birth")
            plt.ylabel("Death")
            plt.plot([0,f_max], [0,f_max], 'r--')

            plt.title(f"Persistent {dim}-Homologies")
            plt.xlim((-0.05*f_max,f_max))
            plt.ylim((-0.05*f_max,f_max))

            plt.savefig(f"./CDHWdata_figs/{fname[-5]}/H{dim}_persistence_{fname[:-4]}.png")
            plt.close()
        
        exit(0)


if __name__=='__main__':
    main()