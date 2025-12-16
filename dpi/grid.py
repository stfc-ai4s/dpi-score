import os
import numpy as np
from scipy import signal
from sklearn.metrics import pairwise_distances
import pandas as pd


class Grid:
    """
    Convert the feature matrix into grid.
    features (str): Path to feature matrix or dataframe of shape (M, 9) with columns (X, Y, Z, C, N, O, S, chainID, resID),
        where (X, Y, Z) is coordinates, (C, N, O, S) are atomic labelling, ChainID, resID
    features (list[str]): List of string of features to convert to grid. If None, all features in the feature matrix will be converted to grid. 
    name: group/chain name
    grid_size: int, size of the grid
    dynamic_grid: (Bool) If True, keep the original size and reshape to grid_size.
    feats_cols: List of indexes of the features to keep from the feature matrix
    """
    def __init__(self, 
                 features,
                 feature_names=['C', 'N', 'O', 'S', 'P', 'H'],
                 name='', 
                 grid_size=32, 
                 grid_type='dynamicgrids', 
                 smooth_features = ["C", "N", "O", "S"],
                 smooth_type='wdw',
                 **kwargs):

        self.features = features
        self.feature_names = feature_names
        self.name = name
        self.grid_size = grid_size
        self.grid_type = grid_type
        self.dynamic_grid = grid_type=='dynamicgrids'
        self.smooth_features = smooth_features
        self.smooth_type = smooth_type

        # Load the feature matrix
        if isinstance(features, (str, os.PathLike)):
            self.feature_matrix = pd.read_csv(features)
        else:
            self.feature_matrix = features
            
        if feature_names is None:
            self.feature_names = self.feature_matrix.columns.to_list()

        self.feature_matrix['vdW_radius'] = self.feature_matrix.apply(self.get_vdw_radius, axis=1)

        # Ensure the feature matrix has the expected number of features
        # assert self.feature_matrix.shape[-1] == self.num_features

        self.atom_coordinates = self.feature_matrix[['x', 'y', 'z']].values
        self.grid_coordinates, self.grid_dimensions = self.get_grid_settings()
        
        self.n_atoms_grid = 0

    @property
    def num_atoms(self, ):
        return len(self.feature_matrix)
        
    @property
    def num_features(self,):
        return len(self.feature_names)

    @property
    def retention_ratio(self,):
        return self.n_atoms_grid/len(self.feature_matrix)*100
        
    # Function to get vdW radius based on atom type
    def get_vdw_radius(self, row):
        vdw_radii = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 'H': 1.20}
        for atom, radius in vdw_radii.items():
            if row[atom] == 1.0:  # If the atom type column is 1.0 (indicating presence)
                return radius
        return None  # Default if no atom type is found

        
    def process(self, interaction_dist=7):
        self.grid_features = {}
        # Process each feature listed in features_to_process
        for feature_name in self.feature_names:
            if feature_name in self.feature_matrix.columns:

                feature_grid = self.create_grid(self.feature_matrix[feature_name] )
                
                # Update number of atom before smoothing for computing compression ratios
                if feature_name in ["C", "N", "O", "S", "P", "H"]:
                    self.n_atoms_grid += (feature_grid>0).sum() # count atoms in grid
                
                if feature_name in self.smooth_features:
                    feature_grid = self.smoothen(feature_grid, feature_name)
                    
                self.grid_features[feature_name] = feature_grid

        pairwise_dist = pairwise_distances(self.atom_coordinates, self.atom_coordinates)
        # average distances to all other atoms, gives the global structure
        # self.grid_features['pairwise_distance'] = self.create_grid(pairwise_dist.mean(-1)) 
        # self.grid_features['nearest_distance'] = self.create_grid(pairwise_dist.min(-1)) 
        self.grid_features['coords_norm'] = self.create_grid(np.linalg.norm(self.atom_coordinates, axis=-1)) # distance from origin

        
        if 'interaction_energy' in self.feature_names:
            chain_feats = self.feature_matrix['chainID'].values
            chain_mask = chain_feats[:, None]!=chain_feats[None, :]
            
            interchain_distance = np.divide(
                (pairwise_dist * chain_mask).sum(-1), # sum of masked inter chain distances, doesn't include distance threshold
                chain_mask.sum(-1), # divided by total atoms from other other chain
                where = chain_mask.sum(-1) != 0,
                out = np.zeros((pairwise_dist.shape[0]))  # Set 0 in places where the denominator is 0
            )
    
            # count number of interaction with atoms from other chains within distance threshold
            interaction_energy = (chain_mask * (pairwise_dist<interaction_dist)).sum(-1) 
            
            self.grid_features['interaction_energy'] = self.create_grid(interaction_energy)
            self.grid_features['interchain_distance'] = self.create_grid(interchain_distance)

        # generate grid for steric_clash
        clash_grid = np.zeros((self.grid_dimensions[0], self.grid_dimensions[1], self.grid_dimensions[2]), dtype=np.float32)
        
        pairwise_radii = self.feature_matrix.vdW_radius.values[:, None] + self.feature_matrix.vdW_radius.values[None, :]
        clash_matrix = pairwise_dist<pairwise_radii

        N = len(clash_matrix)
        # Fill grid based on clashes
        for i in range(N):
            for j in range(i + 1, N):  # Avoid redundant pairs
                if clash_matrix[i, j]:  # Clash detected
                    # Get voxel positions
                    v1, v2 = self.grid_coordinates[i], self.grid_coordinates[j]  
                    # Mark voxels (could also interpolate for path filling)
                    clash_grid[v1[0], v1[1], v1[2]] = 1
                    clash_grid[v2[0], v2[1], v2[2]] = 1

        self.grid_features['steric_clash'] = clash_grid
        
        return self.grid_features
        
    def create_grid(self, feature_values, fill_method='nearest'):
        feature_grid = np.zeros((self.grid_dimensions[0], self.grid_dimensions[1], self.grid_dimensions[2]), dtype=np.float32)
        # Populate the grid with feature values, coord might be repeated
        coord_dists = {} # for each coord, store the distance to features position
        for idx, coord in enumerate(self.grid_coordinates):
            curr_dist = np.linalg.norm(coord - self.atom_coordinates[idx]) # quantization distances
            
            if tuple(coord) not in coord_dists:
                feature_grid[tuple(coord)] = feature_values[idx]
                coord_dists[tuple(coord)] = curr_dist
                
                continue

            if fill_method=='nearest':
                # populate with the nearest features
                if curr_dist < coord_dists[tuple(coord)]:
                    feature_grid[tuple(coord)] = feature_values[idx]
            elif fill_method=='add':
                # Add feature value 
                feature_grid[tuple(coord)] += feature_values[idx]
            else:
                # overwrite
                feature_grid[tuple(coord)] = feature_values[idx]

        return feature_grid
                
    def get_grid_settings(self, buffer=1):
        # Calculate the minimum and maximum values for each axis
        min_vals = np.min(self.atom_coordinates, axis=0)
        max_vals = np.max(self.atom_coordinates, axis=0)
        
        # Normalize coordinates to the range [0, 1]
        normalized_coords = (self.atom_coordinates - min_vals) / (max_vals - min_vals)
    
        if self.dynamic_grid:
            grid_shape = (max_vals - min_vals).astype(int) + buffer
        else:
            grid_shape = np.array([self.grid_size, self.grid_size, self.grid_size])
            
        # Scale normalized coordinates to the grid size
        grid_coords = np.around(normalized_coords * (grid_shape - 1)).astype(int)

        return grid_coords, grid_shape
    
        
    def smoothen(self, grid, atom_type):
        """
        Inputs:
            grid - array of shape (grid_size, grid_size, grid_size)
            atom_type - name of atom
        """
        # first build the smoothing kernel
        if self.smooth_type == "wdw":
            sigma = {"C":1.7, "N":1.55, "O":1.52, "S":1.8, "P":1.8, "H": 1.2}  # van der Waals radius of an atom
        elif self.smooth_type == "const":
            sigma = {"C":1, "N":1, "O":1, "S":1, "P":1, "H": 1}
        elif self.smooth_type == "calc":
            sigma = {"C":0.67, "N":0.56, "O":0.48, "S":0.88}
        
        x = np.arange(-3, 4, 1)  # coord arrays -- make sure they contain 0!
        y = np.arange(-3, 4, 1)
        z = np.arange(-3, 4, 1)
        yy, xx, zz = np.meshgrid(y, x, z)
        kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma[atom_type] ** 2))

        # apply to sample data
        grid = signal.convolve(grid, kernel, mode="same")
        
        return grid
