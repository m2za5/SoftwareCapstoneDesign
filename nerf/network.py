import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize

# TODO: not sure about the details...
class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.norm = nn.LayerNorm(self.dim_out)
        self.activation = nn.SiLU(inplace=True)

        if self.dim_in != self.dim_out:
            self.skip = nn.Linear(self.dim_in, self.dim_out, bias=False)
        else:
            self.skip = None

    def forward(self, x):
        # x: [B, C]
        identity = x

        out = self.dense(x)
        out = self.norm(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.activation(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C]

        out = self.dense(x)
        out = self.activation(out)

        return out    

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, block=BasicBlock):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            if l == 0:
                net.append(BasicBlock(self.dim_in, self.dim_hidden, bias=bias))
            elif l != num_layers - 1:
                net.append(block(self.dim_hidden, self.dim_hidden, bias=bias))
            else:
                net.append(nn.Linear(self.dim_hidden, self.dim_out, bias=bias))

        self.net = nn.ModuleList(net)
        
    
    def forward(self, x):

        for l in range(self.num_layers):
            x = self.net[l](x)
            
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=5,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=32,
                 encoding='frequency_torch',
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder, self.in_dim = get_encoder(encoding, input_dim=3, multires=12)
        self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True, block=ResBlock)

        self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else F.softplus

        self.mesh_prior_scale = getattr(opt, "mesh_prior_scale", 0.2)
        self.mesh_prior_max_sigma = getattr(opt, "mesh_prior_max_sigma", 20.0)
        self._prior_debug_printed = 0

        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding, input_dim=3, multires=4)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    def common_forward(self, x):
        enc = self.encoder(x, bound=self.bound, max_level=self.max_level)
        h = self.sigma_net(enc)

        sigma_raw = torch.zeros_like(h[..., 0])
        if getattr(self, "density_init", None) is not None:
            R = self.density_init.shape[0]
            coords = (x / (2 * self.bound) + 0.5) * (R - 1)
            idx = coords.round().long().clamp(0, R - 1)
            ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
            prior = self.density_init[ix, iy, iz]   # [N]
            t = 0.2
            prior_sigma = F.relu(prior - t) * self.mesh_prior_scale

            sigma_raw = sigma_raw + prior_sigma

        sigma_raw = sigma_raw + self.density_blob(x)

        sigma = self.density_activation(sigma_raw)
        sigma = sigma.clamp(min=0.0, max=self.mesh_prior_max_sigma)

        albedo = torch.sigmoid(h[..., 1:])
        return sigma, albedo

    def finite_difference_normal(self, x, epsilon=1e-2):
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal
    
    def normal(self, x):
    
        with torch.enable_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x.requires_grad_(True)
                sigma, albedo = self.common_forward(x)
                normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]

        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)

        return normal
        
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        if shading == 'albedo':
            sigma, color = self.common_forward(x)
            normal = None
        
        else:     
            with torch.enable_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x.requires_grad_(True)
                    sigma, albedo = self.common_forward(x)
                    normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]
            normal = safe_normalize(normal)
            normal = torch.nan_to_num(normal)
            lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0)

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else:
                color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

      
    def density(self, x):
        sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):

        h = self.encoder_bg(d)
        
        h = self.bg_net(h)
        rgbs = torch.sigmoid(h)

        return rgbs

    def get_params(self, lr):

        params = [
            # {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
        ]        

        if self.opt.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        if self.opt.dmtet and not self.opt.lock_geo:
            params.append({'params': self.sdf, 'lr': lr})
            params.append({'params': self.deform, 'lr': lr})

        return params