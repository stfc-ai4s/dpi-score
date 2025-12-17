## Installation
To avoid issues with python packages, we suggest to use conda environment
```
conda create -n dpi python=3.12
conda activate dpi
```

We ran our experiments in Nvidia V100 using CUDA11.8 and Pytorch v2.2.2. If your CUDA version is different, please see [Previous Pytorch Version](https://pytorch.org/get-started/previous-versions/) for Pytorch installations. 
```
# CUDA 11.8
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
# For CPU only
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
```

Now, install the DPI Package
```
git clone https://gitlab.com/ccpem/dpi.git
cd dpi
pip install -e .
```

## Download DPI Models
Download one of the model variant from [DPI Models](https://zenodo.org/records/15268284?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjU5OWUzZDg3LWJkODAtNGRkZS05NzZlLTgzNjExZDIyYTNiOSIsImRhdGEiOnt9LCJyYW5kb20iOiJjYmYyN2RjNTlhOGY5MmM3NDRmMGVhNDIxNDEzNjk2MyJ9.EEDGLGU7UxzhyHLdvYEz_zIsIGKRDfVuLNBPowaQXnZzb_xJ0o9Dz1oBOGcF8uLLSyDUs9ZKXz4g7eZxLnkxXA) from Zenodo and save it to the `model` directory. 

## Usage
You can specify a single file either in .pdb format, or specify a directory containing .pdb files. Optionally, it also support .cif files. 

Single PDB File
```
dpi-inference --input </path/to/pdb> --model </path/to/model>
```
Directory of PDB Files
```
dpi-inference --input </path/to/pdb_directory/> --model </path/to/model>
```
### Advance usage 
```
dpi-inference \
    --input </path/to/pdb> \
    --model_dir </path/to/model> \
    --checkpoint model_k2 \
    --output_dir ./results \
    --min_num_residues 10 \
    --max_neighbors_dist 7 \
    --gpu 0
```

## Command Line Arguments

### Required Arguments
- `--input`: Path to PDB file or directory containing PDB files

### Model Arguments
- `--model_dir`: Directory containing the trained model and config(default: `./models/dynamicgrids_aug/`)
- `--checkpoint`: Name of model checkpoint to load (default to K=2 model `model_k2`)

### Output Arguments
- `--output_dir`: Directory to save results (default: `./results`)

### Filtering Parameters
- `--min_num_residues`: Minimum residues required in a chain (default: 10)
- `--max_neighbors_dist`: Maximum distance for neighbor search (default: 7.0 Å)
- `--min_neighbors_chains`: Minimum chain length for neighbor search (default: 30)
- `--max_num_chains`: Maximum number of chains per file (default: 30)
- `--author`: Include author sequences if set

### iAlign Parameters (Optional)
- `--ialign_file`: Path to iAlign executable
- `--ialign_cutoff`: IS-score cutoff for classification (default: 0.7)
- `--irmsd`: Minimum RMSD threshold (default: 3.0)

### GPU Acceleration
- `--gpu`: GPU device ID to use for inference, if multi GPUs (default: 0)

## Example Usage

### Single PDB File
```bash
dpi-inference \
    --input ./examples/H1157.pdb \
    --model_dir ./models/dynamicgrids_aug \
    --checkpoint model_k2 \
    --output_dir ./results
```

### Directory Processing
```bash
dpi-inference \
    --input ./examples/ \
    --model_dir ./models/dynamicgrids_aug \
    --checkpoint model_k2 \
    --output_dir ./results
```

### Custom Model and Parameters
```bash
dpi-inference \
    --input ./test_dir/ \
    --model_dir ./models/custom_model/ \
    --checkpoint model.pth \
    --output_dir ./custom_results \
    --max_neighbors_dist 8.0 \
    --author
```

## Output Files

The tool generates two main output files in the specified output (default `results`) directory:

### 1. `inference_results.csv`

This CSV file contains the prediction results for all interface that exist in each PDB with the following columns:

| Column | Description |
|--------|-------------|
| `pdb` | PDB identifier/filename |
| `interface` | Interface identifier (e.g., "A_B" for chains A and B) |
| `dpi_score` | Our model score (0-1, <0.5 for negative interface otherwise positive) |

**Example CSV content:**
```csv
pdb,interface,dpi_score
H1157,A_B,0.99
```

### Interpretation of DPI Score

- **Above 0.5**: Positive (good) interface
- **Below 0.5**: Negative (bad) interface


### 2. `meta_labels.json`

This JSON file contains detailed metadata for each processed interface:

```json
{
  "H1157_A_B": {
    "pdb": "H1157",
    "interface": "A_B",
    "dir": "examples",
    "label": -1,
    "interface_residues":[
     ["HIS663", "ILE461", ...],
     ["GLU220", "GLU327" ... ]
    ],
    "is_scr": null,
    "irmsd": null,
  }
}
```

#### Meta Labels Fields:
- **pdb**: Original PDB filename/identifier
- **interface**: Chain pair identifier (e.g., "A_B)
- **dir**: Directory name of the pdb file
- **label**: Ground truth class label (0 for negative interfaces or 1 for positive interface) obtained from is_scr and irmsd threshold if provided. 
- **interface_residues**: List of residue sequence from each chain in the interface participating in the interaction.
  - **sequence1**: List of amino acid sequence in the chain 1
  - **sequence2**: List of amino acid sequence in the chain 2
- **is_scr**: is_scr score from the ialgin file if provided
- **irmsd**: iRMSD if ialign file if provided

## Output Directory Structure
If input is a directory e.g. `./examples`;
```
results/
└── examples/
    ├── inference_results.csv
    └── meta_labels.json
```

If processing single pdb, e.g `H1157.pdb`:
```
results/
└── H1157/
    ├── inference_results.csv
    └── meta_labels.json
```


## Troubleshooting

### Common Issues:

1. **No valid interfaces found**: 
   - Check if PDB file contains multiple chains
   - Reduce `--min_num_residues` or `--min_neighbors_chains` parameter
   - Increase `--max_neighbors_dist` parameter

2. **CUDA out of memory**:
   - Use `--gpu -1` to run on CPU
   - Process files individually instead of in batches

3. **Model not found**:
   - Ensure `--model_dir` points to correct model directory
   - Check that the specified `--checkpoint` exists

### Performance Notes:
- GPU processing is significantly faster than CPU
- Processing time depends on protein size and number of interfaces
- Large complexes with many chains will take longer to process


## Model Information

The default model uses:
- Dynamic grid-based representation with rotation augmentations (DPI-IV)


For any other questions and issues, please mail to (niraj.bhujel@stfc.ac.uk).
