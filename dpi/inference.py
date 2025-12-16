import os
import sys
import time
import json
import random
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict
from skimage.transform import resize

import torch
from torch.nn import functional as F

abspath = os.path.abspath(__file__)
print(abspath)

# if "./src" not in sys.path:
#     sys.path.insert(0, './src')

# if "./prepare" not in sys.path:
#     sys.path.insert(0, './prepare')
    
from .network import Network
from .grid import Grid
from .prep import PreparePDBDataset
from .helper import get_files_in_dir

def absolute_path(path_string):
    """Convert path to absolute path"""
    return str(Path(path_string).resolve())

def set_random_seed(seed):
    #setup seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def load_checkpoint(model, ckpt_path, map_location='cuda', strict=True):
    
    print(f"Loading pretrained model from {ckpt_path}...")
    
    ckpt = torch.load(ckpt_path, map_location=map_location)
    
    if 'model_states' in os.path.basename(ckpt_path) :
        ckpt = ckpt['model_state']
        
    model.load_state_dict(ckpt, strict=strict)
    
    return model
        
def get_files_paths(input, file_types):
    # input as path to pdb  
    if os.path.isdir(input):
        pdb_files = get_files_in_dir(root=input, file_end=tuple(file_types))
        pdb_files_path = [os.path.join(input, f) for f in pdb_files]

    # path to single pdb file
    elif os.path.isfile(input):
        pdb_files = [os.path.basename(input)]
        pdb_files_path = [input]
        
    else:
        raise ValueError(f"Invalid input: {input}. Must be a directory path, or a file path.")
    return pdb_files_path

def create_grid(df_features, grid_size, grid_type, grid_features):
    # Create grids from feature matrix
    grid = Grid(df_features, 
                grid_size=grid_size,
                grid_type=grid_type, 
                )
    
    grid_dict = grid.process()
    if grid_type == 'dynamicgrids':
        grid_dict = {
            f: resize(
            grid, 
            (grid_size, grid_size, grid_size),
            order=1, 
            mode='reflect', 
            anti_aliasing=True) for f, grid in grid_dict.items()
        }
    grid_features = np.stack([grid_dict[k] for k in grid_features]).astype(np.float32)
    return grid_features

def run(args):
    print(args)

    print("Starting ...")
    start = time.time()
    
    if not os.path.exists(args.model_dir):
        raise Exception(f"{args.model_dir} doesn't exists!")
    

    # Create a processor with empty inputs as we are manually handling pdb files
    processor = PreparePDBDataset(
        root_dir='.', 
        dataset='inference', 
        pdb_dirs=None, 
        min_neighbors_chains=args.min_neighbors_chains, 
        max_neighbors_dist=args.max_neighbors_dist, 
        min_num_residues=args.min_num_residues, 
        max_num_chains=args.max_num_chains, 
        author=args.author, 
        ialign_file=args.ialign_file,
        )

    pdb_files_path = get_files_paths(args.input, [".cif", ".pdb", ".ent"])
    
    if not len(pdb_files_path)>0:
        print(f"{len(pdb_files_path)} pdbs found! Terminating...")
        return 0
    
    for pdb_path in pdb_files_path:
        print(f"Processing {pdb_path} ... ")
        pdb_name = os.path.basename(pdb_path).split('.')[0]
        processor.process_pdb(pdb_name, [pdb_path])
    
    # print(f"{processor.global_counter} interfaces processed.")
    if not len(processor.intf_features):
        print(f"Processor could not find any valid interfaces. Terminating...")
        return 0
        
    # sys.exit()
    
    if args.gpu!=-1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    
    # load config
    cfg = OmegaConf.load(args.model_dir + '/cfg.yaml')
    # print(OmegaConf.to_yaml(cfg.data))

    # Set seed
    set_random_seed(cfg.rng.seed)

    # Load model 
    model = Network(cfg).to(device)
    load_checkpoint(model, f"{args.model_dir}/{args.checkpoint}.pth", map_location=device)
    print("Model loaded sucessfully.")
    
    model.eval()

    predictions = []
    print("Running inference ... ")
    for pdb_intf, pdb_intf_feats in processor.intf_features.items():
        
        inference_start = time.time()
        
        grid_features = create_grid(pdb_intf_feats, cfg.data.grid_size, cfg.data.grid_type, cfg.data.grid_features)
        
        model_inputs= {'grids': torch.from_numpy(grid_features).unsqueeze(0).to(device)}
        with torch.no_grad():
            model_preds = model(model_inputs)
            
        y_probs = F.softmax(model_preds, dim=-1).squeeze().cpu().numpy()
        dpi_score = y_probs[1]
        print(f"Interface: {pdb_intf}, dpi_score: {min(dpi_score, 0.999):.3f}, inference time: {time.time()-inference_start:.3f} secs")

        pdb_intf_info = processor.meta_dict.get(pdb_intf, {})
        
        predictions.append(dict(
            pdb = pdb_intf_info.get('pdb', None),
            interface=pdb_intf_info.get('interface', None),
            dpi_score = dpi_score,
            )
        )

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
    parser = argparse.ArgumentParser(description="Run inference on single PDB file or PDB directory containing multiple pdb files")

    # PDB to HDF5
    parser.add_argument("--input", type=absolute_path, default='./examples',
                        help="Path to PDB directory, where pdb files are located (relative to root dir)."
                       )
    parser.add_argument('--model_dir', type=absolute_path, default='./models/dynamicgrids_aug',
                        help='Directory/Path where the trained models are located'
                       )
    parser.add_argument("--checkpoint", type=str, default='model_k2',
                        help="Name of model checkpoint to load (default to fold 2)."
                       )
    # Ialign
    parser.add_argument("--ialign_file", type=str,
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
    parser.add_argument("--output_dir", type=absolute_path, default='./results',
                        help="Output directory to save the output/results (relative to root dir)."
                       )
    
    args = parser.parse_args()

    run(args)
    
if __name__ == '__main__':
    
    # python inference.py --input ./examples
    # python inference.py --input ./examples/H1129.pdb
    
    main()
    
    