"""
Authors: 
    Niraj Bhujel AI4S, STFC, UK (niraj.bhujel@stfc.ac.uk)
    Sony Malhotra CCP-EM, STFC, UK (sony.malhotra@stfc.ac.uk)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize

from .network import CNN3D, ClassHead
from .grid import Grid

        
class DPIScore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        
        self.net = CNN3D(
            len(cfg.data.grid_features), 
            cfg.net.hidden_dim,
            drop_prob=cfg.net.drop,
        )
        
        # Pooling and Flattening
        if cfg.net.pool=='avg':
            self.pool3d = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
            
        elif cfg.net.pool=='max':
            self.pool3d = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))
        else:
            raise Exception(f"cfg.net.pool can be either 'avg' or 'max', but{cfg.net.pool} given!")
            
        self.flat = nn.Flatten(start_dim=1)
        
        # Classification Head
        self.class_head = ClassHead(
            cfg.net.hidden_dim, 
            cfg.data.num_class,
            dropout=cfg.net.drop_head,
        )
    
    def forward(self, batch, **kwargs):

        x = self.net(batch['grids'], **kwargs)
        x = self.pool3d(x)
        
        x = self.class_head(self.flat(x))

        return x

    @torch.no_grad()
    def predict(self, interface_features, device='cpu'):
        """
        Perform interface assessments based on the coordinates of interface atoms
        Args:
            interface_features (pd.Dataframe): Dataframe with columns ['x', 'y', 'z', 'C', 'N', 'O', 'S', 'P', 'H']
            Where,['x', 'y', 'z'] is the coordinates of one of the atoms present at that location, 
            Atom 'C', 'N', 'O', 'S', 'P', 'H' are binary indicating the presence/absence of the atom. 

        """
        grid_size = self.cfg.data.grid_size
        grid_type = self.cfg.data.grid_type

        grid = Grid(interface_features, 
                    grid_size=grid_size,
                    grid_type=grid_type, 
                    )
        
        grid_dict = grid.process()
        if grid_type == 'dynamicgrids':
            grid_dict = {
                f: resize(
                grid, 
                [grid_size]*3,
                order=1, 
                mode='reflect', 
                anti_aliasing=True) for f, grid in grid_dict.items()
            }

        # Prepare model inputs
        model_inputs = np.stack([grid_dict[k] for k in self.cfg.data.grid_features]).astype(np.float32)
        model_inputs = {'grids': torch.from_numpy(model_inputs).unsqueeze(0).to(device)}

        with torch.no_grad():
            model_preds = self.forward(model_inputs)

        probs     = F.softmax(model_preds, dim=-1).squeeze().cpu().numpy() # [2, ] logits
        dpi_score = float(probs[1]) # probablity of being good quality interface
        
        return dpi_score


    def load_checkpoint(self, ckpt_path, map_location='cuda', strict=True):
        
        print(f"Loading pretrained model from {ckpt_path}...")
        
        state_dict = torch.load(ckpt_path, map_location=map_location)
        
        # if 'model_states' in os.path.basename(ckpt_path) :
        #     state_dict = ckpt['model_state']
            
        self.load_state_dict(state_dict, strict=strict)

        print("Model loaded successfully.")