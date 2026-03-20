"""
    Authors: 
    Nicholas Whyatt UKRI-STFC on behalf of CCP-EM
    Modified by Niraj Bhujel on behalf of SciML
    Adapted from pi-score code (credit to Sony Malhotra)

"""

import os
import sys
import h5py
import json
import tqdm
import gemmi
import time
import shutil
import argparse
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter

from . import helper
from . import docking as dock
from . import similarity as sim
from . interface import get_interface_dict, get_residue_atoms

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
# os.chdir(dname)

class PDBProcessor:
    """
    A class to process .PDB or .CIF files and extract features from each interface in the PDB and save to csv files.

    Attributes:
        min_neighbors_chains (int, optional): Minimum number of neighboring chains for neighbor search. Default is 30.
        max_neighbors_dist (float, optional): Maximum distance to consider for neighboring search. Default is 7.0.
        min_num_residues (int, optional): Minimum number of residues for an interface. Default is 10.
        max_num_chains (int): Maximum number of chains within a file.
        author (str): denotes if author denoted sequences should be used
        ialign_file (str): Path to the ialign file containing alignment data.
        ialign_cutoff (float): Cutoff value for ialign alignment.
        file_types (list[str]): List of file extensions to be processed (e.g., ['.pdb', '.cif']).
        out_dir (str, optional): Output directory to save the HDF5 files. Default is None.

    """
    def __init__(self, 
                 min_neighbors_chains=30, 
                 max_neighbors_dist=7.0, 
                 min_num_residues=10, 
                 max_num_chains=30, 
                 author=None, 
                 ialign_file_path=None,
                 ialign_cutoff=0.7,
                 irmsd_th=3,
                 file_types=[".cif", ".pdb"],
                 out_dir='./dpi_outputs',
                 clean=False,
                ):

        self.min_neighbors_chains = min_neighbors_chains
        self.max_neighbors_dist = max_neighbors_dist
        self.min_num_residues = min_num_residues
        self.max_num_chains = max_num_chains
        self.author = author
        self.ialign_file_path = ialign_file_path
        self.ialign_cutoff = ialign_cutoff
        self.irmsd_th = irmsd_th
        self.file_types = file_types
        self.out_dir = out_dir or root_dir
        self.clean = clean
        
        if self.ialign_file_path:
            with open(ialign_file_path, "rb") as f:
                self.ialign_dict = json.load(f)
            print(f"ialign dict {self.ialign_file} loaded successfully!! ")
        else:
            self.ialign_dict = {}
        
        self.__reset__()
        
    def __reset__(self, ):
        self.intf_features = {}
        self.meta_dict = {}
        self.global_counter = 0

    def get_files_paths(self, input):
        # input as path to pdb  
        if os.path.isdir(input):
            pdb_files = helper.get_files_in_dir(root=input, file_end=tuple(self.file_types))
            pdb_files_path = [os.path.join(input, f) for f in pdb_files]

        # path to single pdb file
        elif os.path.isfile(input):
            pdb_files = [os.path.basename(input)]
            pdb_files_path = [input]
            
        else:
            raise ValueError(f"Invalid input: {input}. Must be a directory path, or a file path.")
        return pdb_files_path

    def get_chain_id(self, name: str, md: gemmi.Model) -> int:
        """
        Given a structure and a chain name, figure out the correct index
    
        Inputs:
            name (str): Chain name, i.e. "A", or "SSS"
            md (gemmi.Structure): target model
    
        Returns:
            int of chain_id
        """
        for idx, chain in enumerate(md):
            if chain.name == name:
                return idx
        raise LookupError(f"Chain {name} was not found within model {md.name}")
        
    def process_pdb(self, name, files):
        """
        Process a PDB structure.  Assumes interface
        dictionaries are next to target files.
    
        Inputs:
            name (str): name of PDB file, typically the pdb key (1B3J etc)
            files (list): list of file paths structures belonging to the PDB (xray, em)
    
        Returns:
            None
        """
        
        for file in files:
    
            try:
                res_contacts, intfs = get_interface_dict(
                    file=file,
                    len_chains = self.min_neighbors_chains,
                    dist_cut = self.max_neighbors_dist,
                    nres_cut = self.min_num_residues,
                    num_chains = self.max_num_chains,
                    author = self.author,
                    )
                        
            except Exception as e:
                print(e)
                continue
                    
            # if name not contain zdock then reduce intfs by dis_intfs file needs to be a full path
            ialign_calls = None
            if "zdock" not in file and len(intfs)>0:
                ialign_calls = sim.get_dissimilar_interfaces(
                    filename=file, 
                    intfs=intfs, 
                    ialign_dict=self.ialign_dict, 
                    ialign_cutoff=self.ialign_cutoff, 
                    rmsd_th=self.irmsd_th,
                )
            
            
            if not len(intfs)>0:
                warnings.warn(f"Could not find valid interfaces in {file}")
                continue

            st = gemmi.read_structure(file)

            for intf in intfs:
                
                if ialign_calls is not None and intf not in ialign_calls:
                    continue
                
                chs = intf.split("_")
                ch_0_id = self.get_chain_id(name=chs[0], md=st[0])
                ch_1_id = self.get_chain_id(name=chs[1], md=st[0])
    
                ca_0 = get_residue_atoms(
                    chain=st[0].find_chain(chs[0]),
                    chain_id=ch_0_id,
                    res_list=intfs[intf][0],
                )
                ca_1 = get_residue_atoms(
                    chain=st[0].find_chain(chs[1]),
                    chain_id=ch_1_id,
                    res_list=intfs[intf][1],
                )
                
                feature_names = list(ca_0[0].keys())
                chain0_feats, chain1_feats = [], []
                for item in ca_0:
                    chain0_feats.append([item[k] for k in feature_names])
    
                for item in ca_1:
                    chain1_feats.append([item[k] for k in feature_names])
            
                feature_matrix = np.concatenate([chain0_feats, chain1_feats], axis=0)
                
                dbset = os.path.splitext(file)[0].split(os.sep)[-2]
                file_id = f"{os.path.basename(file).split('.')[0]}_{intf}" 
                    
                
                # Save feature matrix
                df = pd.DataFrame(feature_matrix, columns=feature_names)
                
                # Save meta info
                meta_labels = {
                    "pdb": name,
                    "dir": dbset,
                    "file_id": file_id,
                    "label": -1,
                    "interface": intf,
                    "interface_residues": intfs[intf], 
                }

                # Put the meta labels in a global container
                # del meta_labels["interface_residues"]
                self.meta_dict[file_id] = meta_labels
                self.intf_features[file_id] = df
                
                self.global_counter += 1 

    
    def process(self, inputs):
        self.__reset__()

        pdb_files_path = self.get_files_paths(inputs)
        print(f"Found {len(pdb_files_path)} PDB file(s).")

        if not pdb_files_path:
            raise Exception("No PDB files found. Please check your INPUT path.")
            
        for pdb_path in tqdm(pdb_files_path, desc="Processing PDBs"):
            pdb_name = os.path.basename(pdb_path).split('.')[0]
            print(f"  Processing {pdb_path} ...")
            self.process_pdb(pdb_name, [pdb_path])

        if not self.intf_features:
            raise RuntimeError("No valid interfaces found. Check your filtering parameters.")

        print(f"\n{len(self.intf_features)} interface(s) ready for inference.")
        
        print("Done!")
        

