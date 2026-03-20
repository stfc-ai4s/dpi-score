"""
Authors: 
    Niraj Bhujel AI4S, STFC, UK (niraj.bhujel@stfc.ac.uk)
    Nick Whyatt, CCP-EM, STFC, UK (nicholas.whyatt@stfc.ac.uk)
    Sony Malhotra CCP-EM, STFC, UK (sony.malhotra@stfc.ac.uk)
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.nn import functional as F

abspath = os.path.abspath(__file__)
print(abspath)

from dpi.model import DPIScore
from dpi.grid import Grid
from dpi.processor import PDBProcessor
from dpi.helper import get_files_in_dir


def run(args):
    print(args)

    print("Starting ...")
    start = time.time()
    
    if not os.path.exists(args.model_dir):
        raise Exception(f"{args.model_dir} doesn't exists!")
    

    # Create a processor and process PDB files
    processor = PDBProcessor(
        min_neighbors_chains=args.min_neighbors_chains, 
        max_neighbors_dist=args.max_neighbors_dist, 
        min_num_residues=args.min_num_residues, 
        max_num_chains=args.max_num_chains, 
        author=args.author, 
        ialign_file_path=args.ialign_file_path,
        )
    processor.process(args.input)
    
    
    # Setup device
    if args.gpu!=-1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    
    # load config
    cfg = OmegaConf.load(args.model_dir + '/cfg.yaml')
    # print(OmegaConf.to_yaml(cfg.data))

    # Load model 
    model_cfg = OmegaConf.load(f"{args.model_dir}/cfg.yaml")

    model = DPIScore(model_cfg)
    model.load_checkpoint(f"{args.model_dir}/{args.checkpoint}", map_location=device)
    model.to(device)
    model.eval()

    predictions = []
    start = time.time()

    for intf, intf_feats in processor.intf_features.items():
        t0 = time.time()

        dpi_score = model.predict(intf_feats, device)

        print(
            f"Interface: {intf} | "
            f"dpi_score: {min(dpi_score, 0.999):.3f} | "
            f"time: {time.time() - t0:.3f}s"
        )

        intf_meta_info = processor.meta_dict.get(intf, {})
        predictions.append({
            'pdb':       intf_meta_info.get('pdb', None),
            'interface': intf_meta_info.get('interface', None),
            'dpi_score': dpi_score,
        })

    print(f"\nInference complete in {time.time() - start:.2f}s")

    output_dir = f"{args.output_dir}/{os.path.basename(os.path.splitext(args.input)[0])}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")
    
    df_results_path = f"{output_dir}/inference_results.csv"
    df_results = pd.DataFrame(predictions)
    df_results.to_csv(df_results_path, index=False)
    print(f"Predictions are saved to {df_results_path}")

    meta_file_path = f"{output_dir}/meta_labels.json"
    with open(meta_file_path, "w") as f:
        json.dump(processor.meta_dict, f)
    print(f"Meta info is saved to {meta_file_path}")
    
    print(f"Done in {time.time()-start}secs")

def main():
    parser = argparse.ArgumentParser(description="Run DPI Score inference on single PDB file or PDB directory containing multiple pdb files")

    # PDB to HDF5
    parser.add_argument("--input", type=str, default='./examples',
                        help="Path to PDB directory, where pdb files are located (relative to root dir)."
                       )
    parser.add_argument('--model_dir', type=str, default='./models/dynamicgrids_aug',
                        help='Directory/Path where the trained models are located'
                       )
    parser.add_argument("--checkpoint", type=str, default='model_k2.pth',
                        help="Name of model checkpoint to load (default to fold 2)."
                       )
    # Ialign
    parser.add_argument("--ialign_file_path", type=str,
                        help="File path to ialign."
                       )
    parser.add_argument("--ialign_cutoff", type=float, default=0.7,
                        help="is_score cutoff for class threshold(irrelevant if ialign_dict is empty)."
                       )
    parser.add_argument("--irmsd", type=float, default=3,
                        help="Min rmsd_th for class threshold (irrelevant if ialign_dict is empty)."
                       )

    # Filtering parameters
    parser.add_argument("--min_num_residues", type=int, default=10,
                        help="Minimum number of residues required in a chain for interface. Higher value will increase the number of samples. ")
    parser.add_argument("--max_neighbors_dist", type=float, default=7,
                        help="Maximum distance for neighbor search when calculating interface residues.")
    parser.add_argument("--min_neighbors_chains", type=int, default=30,
                        help="Minimum length of chains for neighbor search when calculating interface residues.")
    parser.add_argument("--max_num_chains", type=int, default=30,
                        help="Maximum number of chains per file to consider.")
    parser.add_argument("--author", action="store_true",
                        help="If set, include author sequences should be used.")

    # Dataloader
    parser.add_argument('--gpu', type=int, default=0,
                        help='The GPU id to use for inference'
                       )
    
    # Output direcotry
    parser.add_argument("--output_dir", type=str, default='./results',
                        help="Output directory to save the output/results (relative to root dir)."
                       )
    
    args = parser.parse_args()

    run(args)
    
if __name__ == '__main__':
    
    # python inference.py --input ./examples
    # python inference.py --input ./examples/H1129.pdb
    
    main()
    
    