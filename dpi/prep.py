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

class PreparePDBDataset:
    """
    A class to process .PDB or .CIF files and extract features from each interface in the PDB and save to csv files.

    Attributes:
        root_dir (str): The root directory containing the dataset.
        dataset (str): The name of the dataset to be processed.
        pdb_dirs (list[str]): List of directories containing PDB files e.g dock_pos, dock_neg.
        file_types (list[str]): List of file extensions to be processed (e.g., ['.pdb', '.cif']).
        min_neighbors_chains (int, optional): Minimum number of neighboring chains for neighbor search. Default is 30.
        max_neighbors_dist (float, optional): Maximum distance to consider for neighboring search. Default is 7.0.
        min_num_residues (int, optional): Minimum number of residues for an interface. Default is 10.
        max_num_chains (int): Maximum number of chains within a file.
        author (str): denotes if author denoted sequences should be used
        ialign_file (str): Path to the ialign file containing alignment data.
        ialign_cutoff (float): Cutoff value for ialign alignment.
        out_dir (str, optional): Output directory to save the HDF5 files. Default is None.

    Methods:

    """
    def __init__(self, 
                 root_dir, 
                 dataset, 
                 pdb_dirs, 
                 file_types=[".cif", ".pdb"],  
                 min_neighbors_chains=30, 
                 max_neighbors_dist=7.0, 
                 min_num_residues=10, 
                 max_num_chains=30, 
                 author=None, 
                 ialign_file=None,
                 ialign_cutoff=0.7,
                 irmsd_th=3,
                 out_dir=None,
                 feature_dir_name='features',
                 clean=False,
                ):


        self.root_dir = root_dir
        self.dataset = dataset
        self.pdb_dirs = pdb_dirs
        self.file_types = file_types
        self.min_neighbors_chains = min_neighbors_chains
        self.max_neighbors_dist = max_neighbors_dist
        self.min_num_residues = min_num_residues
        self.max_num_chains = max_num_chains
        self.author = author
        self.ialign_file = ialign_file
        self.ialign_cutoff = ialign_cutoff
        self.irmsd_th = irmsd_th
        self.out_dir = out_dir or root_dir
        self.clean = clean
        
        self.feature_dir = f"{self.out_dir}/{self.dataset}/{feature_dir_name}"
        self.meta_file_path = f"{self.out_dir}/{self.dataset}/meta/meta_labels.json"

        if self.ialign_file:
            with open(os.path.join(self.root_dir, self.dataset, self.ialign_file), "rb") as f:
                self.ialign_dict = json.load(f)
            print(f"ialign dict {self.ialign_file} loaded successfully!! ")
        else:
            self.ialign_dict = {}
        
        self.__reset__()
        
    def __reset__(self, ):
        self.intf_features = {}
        self.meta_dict = {}
        self.global_counter = 0
        self.c0_counter = 0
        self.c1_counter = 0

    def check_scr_rmsd(self, is_scr, rmsd):
        if is_scr is None or rmsd is None:
            return -1
        
        if (float(is_scr)>=self.ialign_cutoff) & (float(rmsd)<=self.irmsd_th):
            return 1
        # else:
        #     return 0
        elif (float(is_scr)<=0.3) & (float(rmsd)>=4):
            return 0

        else:
            return -1
                
    def lookup_luc_scores(self, pdb_file: str):
    
        file_dir = os.path.dirname(pdb_file)
        pdb_intf_num = os.path.basename(pdb_file).split('.')[0] # contain pdb and intf e.g 5FTJ_B_A_3
        pdb_intf = pdb_intf_num[:8]
        try:
            intf_file_path = f"{file_dir}/{pdb_intf}_interface_residues_dict.json"
            with open(intf_file_path, 'r') as f:
                ialign_dict = json.load(f)
            
            is_scr = float(ialign_dict[pdb_intf]["ground_truth"][pdb_intf_num]['is_scr'])
            rmsd = float(ialign_dict[pdb_intf]["ground_truth"][pdb_intf_num]['rmsd'])

            return is_scr, rmsd
            
        except FileNotFoundError:
            warnings.warn(f"Missing file: {intf_file_path}", category=UserWarning)
        except KeyError as e:
            warnings.warn(f"Missing key in JSON: {e}", category=UserWarning)
        except Exception as e:
            warnings.warn(f"Unexpected error: {e}", category=UserWarning)

        return None, None
            
    def lookup_casp_scores(self, file: str, gname: str, cname: str) -> int:
        """
        Relabels CASP data based on criteria from the pi-score paper PD2/ND elements.
    
        casp_lookup is based on the meta/casp_nat_preds.json file which is the iAlign of every
        prediction against the native file, but the native file will also pass through here.
        If the file isn't found, return 1 (assuming native is class 1).
        """
        
        file_id = os.path.basename(file).split('.')[0]
        # print(gname, file_id, cname)
        try:
            is_scr = float(
                self.ialign_dict[gname][file_id][cname.replace('_', '')]["is_scr"]
            )
            rmsd = float(
                self.ialign_dict[gname][file_id][cname.replace('_', '')]["rmsd"]
            )
            return is_scr, rmsd
            
        except KeyError as e:
            warnings.warn(f"Missing key in {e} align dict", category=UserWarning)
        except Exception as e:
            warnings.warn(f"Unexpected error: {e}", category=UserWarning)
        
        return None, None

    def get_class_label(self, file, name=None, intf=None):
        dataset = self.dataset.lower()
        is_scr, irmsd = None, None
    
        if dataset == 'pdb':
            return 0 if "dock_neg" in file else 1
    
        elif dataset in ('many', 'dc'):
            return 1 if 'bio' in file else 0
    
        elif 'casp' in dataset:
            is_scr, irmsd = self.lookup_casp_scores(file, name, intf)
    
        elif 'luc' in dataset:
            is_scr, irmsd = self.lookup_luc_scores(file)
    
        else:
            raise Exception(f'Label for dataset {self.dataset} not supported yet')
    
        return self.check_scr_rmsd(is_scr, irmsd)
    
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
    
    def prune_empty_h5s(self, ) -> None:
        """
        Removes empty h5s in directory.
    
        Inputs:
            h5_dir (str): target directory
    
        Returns:
            None
        """
        h5_files = helper.get_files_in_dir(root=self.h5_dir, file_end=".h5")

        for h5_file in h5_files:
            with h5py.File(os.path.join(self.h5_dir, h5_file), "r") as f:
                num_intfs = len(f.keys())
            if num_intfs == 0:
                os.remove(os.path.join(self.h5_dir, h5_file))
        return None
        
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
                
                # Add class label 
                is_scr, irmsd = None, None
                if self.dataset.lower()=='pdb':
                    class_label = 0 if "dock_neg" in file else 1
                elif self.dataset.lower() in ('many', 'dc'):
                    class_label = 1 if 'bio' in file else 0
                elif 'casp' in self.dataset.lower():
                    is_scr, irmsd = self.lookup_casp_scores(file, name, intf)
                    class_label = self.check_scr_rmsd(is_scr, irmsd)
                elif 'luc' in self.dataset.lower():
                    is_scr, irmsd = self.lookup_luc_scores(file)
                    class_label = self.check_scr_rmsd(is_scr, irmsd)
                else:
                    print(f"Label for {self.dataset} not supported.Setting class label to -1")
                    class_label= -1
                    
                
                # Save feature matrix
                df = pd.DataFrame(feature_matrix, columns=feature_names)
                
                # Save meta info
                meta_labels = {
                    "pdb": name,
                    "dir": dbset,
                    "file_id": file_id,
                    "label": int(class_label),
                    "interface": intf,
                    "interface_residues": intfs[intf], 
                    "is_scr": is_scr,
                    "irmsd": irmsd,
                }

                # Put the meta labels in a global container
                # del meta_labels["interface_residues"]
                self.meta_dict[file_id] = meta_labels
                self.intf_features[file_id] = df
                
                self.global_counter += 1 
                self.c1_counter += class_label

    
    def prepare(self, ):
        
        """
        This function parses directories for files with specified extensions, checks the 
        interface against given criteria, and saves results in HDF5 formats to the specified output directory. 
        """
        
        helper.create_dir(self.feature_dir, clean=self.clean)

        if self.dataset=='CASP' or self.dataset=='CASP16':
            # casp has an unconventional file structure and non-pdb naming conventions
            st_dict = helper.get_structure_file_dict_casp(
                preds_path = f"{self.root_dir}/{self.dataset}/predictions/oligo",
                targets_path = f"{self.root_dir}/{self.dataset}/targets"
            )
        else:
            # pdb_dirs as multiple dirs
            if isinstance(self.pdb_dirs, list):
                file_lists = [helper.get_files_in_dir(root=os.path.join(self.root_dir, self.dataset, dir), 
                                                    file_end=tuple(self.file_types)) for dir in self.pdb_dirs]
                st_dict = helper.get_structure_file_dict(roots=[os.path.join(self.root_dir, self.dataset, dir) for dir in self.pdb_dirs],
                                                       file_lists=tuple(file_lists))


        # print(f"{[len(file_lists[i]) for i, _ in enumerate(self.pdb_dirs)]} files in {self.pdb_dirs} with format {self.file_types}")

        if not (k := len(st_dict)) > 0:
            raise Exception(f"{k} structures. Nothing to process")

        print(f"{k} structures found ...")

        start, etc = time.time(), 0
        for i, structure in enumerate(st_dict):
            
            print(f"Processing structure {structure}:{i}/{len(st_dict)} etc:{etc*(len(st_dict)-i)/60:.3f}mins")
            # print(f"Processing structure {structure}:{i}/{len(st_dict)} etc:{etc*(len(st_dict)-i)/60:.3f}mins, files:{st_dict[structure]}")

            # Generate h5 file from pdb structure and write to h5_dir
            self.process_pdb(
                name=structure, 
                files=st_dict[structure],
            )
            
            etc = (time.time()-start)/(i+1)
        
        # save features
        for file_id, df in self.intf_features.items():
            df.to_csv(os.path.join(self.feature_dir, f"{file_id}.csv"), index=False)

        # save meta file
        helper.create_dir(os.path.dirname(self.meta_file_path))
        with open(self.meta_file_path, "w") as fp:
            json.dump(self.meta_dict, fp)

        label_counts = Counter(label['label'] for label in self.meta_dict.values())
        for k, v in label_counts.items():
            print(f"Class: {k}: {v}")
            
        print("Total Interfaces:", self.global_counter)
        
        print("Done!")
        

