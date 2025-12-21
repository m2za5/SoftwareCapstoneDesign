import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp, biased_softplus
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize
import torch
import numpy as np
import mcubes
import trimesh

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers


        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=32,
                 ):
        
        super().__init__(opt)

        self.mesh_prior_scale = getattr(opt, "mesh_prior_scale", 0.0)
        self.mesh_prior_max_sigma = getattr(opt, "mesh_prior_max_sigma", 25.0)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3, log2_hashmap_size=19, desired_resolution=2048 * self.bound, interpolation='smoothstep')

        self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        # self.normal_net = MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True)

        self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else biased_softplus
        self.density_init_scale = getattr(opt, "density_init_scale", 1.0)
        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=6)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    def common_forward(self, x):
        enc = self.encoder(x, bound=self.bound, max_level=self.max_level)
        h = self.sigma_net(enc)
        raw = h[..., 0]
        albedo = torch.sigmoid(h[..., 1:])

        sigma = self.density_activation(raw - 5.0)

        if getattr(self, "density_init", None) is not None:
            vol = self.density_init.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
            grid = (x / self.bound).view(1, 1, 1, -1, 3)
            prior = F.grid_sample(vol, grid, align_corners=True, mode='bilinear', padding_mode='border')
            prior = prior.view(x.shape[:-1])
            sigma = sigma + self.mesh_prior_scale * prior
        sigma = sigma + self.density_blob(x)
        sigma = sigma.clamp(min=0.0, max=self.mesh_prior_max_sigma)

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
        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)
        return normal
    
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        sigma, albedo = self.common_forward(x)

        if shading == 'albedo':
            normal = None
            color = albedo
        
        else:
            normal = self.normal(x)
            lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0) # [N,]
            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else:
                color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

      
    def density(self, x):
        sigma, albedo = self.common_forward(x)

        
        """
        # === CHECK PRIOR LOG (first few calls only) ===
        if not hasattr(self, "_prior_debug_printed"):
            self._prior_debug_printed = 0

        # === [ADD] Shap-E mesh 기반 prior 적용 ===
        if getattr(self, "density_init", None) is not None:
            R = self.density_init.shape[0]

            coords = (x / (2 * self.bound) + 0.5) * (R - 1)
            idx = coords.round().long().clamp(0, R - 1)

            px, py, pz = idx[:, 0], idx[:, 1], idx[:, 2]
            prior = self.density_init[px, py, pz]


        # === DEBUG LOGGING ===
        if self._prior_debug_printed < 5:  # print only first 5 times
            print("\n[PRIOR CHECK] -----")
            print(f"x sample (first 3):\n{x[:3].detach().cpu().numpy()}")
            print(f"coords (first 3):\n{coords[:3].detach().cpu().numpy()}")
            print(f"indices (first 3):\n{idx[:3].detach().cpu().numpy()}")
            print(f"prior values (first 10):\n{prior[:10].detach().cpu().numpy()}")
            print(f"sigma(before) min/max = {sigma.min().item():.4f} / {sigma.max().item():.4f}")    

            # *** 여기 핵심: prior를 약하게 더한 뒤 클램프 ***
            #sigma = sigma + PRIOR_SCALE * prior
            #sigma = sigma.clamp(min=0.0, max=SIGMA_MAX)

            t = 0.2  # 필요하면 0.1~0.3 사이에서 조정
            prior = F.relu(prior - t)

            sigma = sigma + self.mesh_prior_scale * prior
            sigma = sigma.clamp(min=0.0, max=self.mesh_prior_max_sigma)
            
            if self._prior_debug_printed < 5:
                print(f"sigma(after)  min/max = {sigma.min().item():.4f} / {sigma.max().item():.4f}")
                print("[PRIOR CHECK] applied\n")
                self._prior_debug_printed += 1
            """
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):

        h = self.encoder_bg(d)     
        h = self.bg_net(h)
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
        ]        

        if self.opt.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        if self.opt.dmtet and not self.opt.lock_geo:
            params.append({'params': self.sdf, 'lr': lr})
            params.append({'params': self.deform, 'lr': lr})

        return params
    
    @torch.no_grad()
    def export_mesh1(self, save_path=None, resolution=None, threshold=None, **kwargs):
        if resolution is None:
            resolution = 512
        S = 64

        target_faces = kwargs.get('decimate_target', 50000)
        if target_faces > 100000: target_faces = 50000

        import mcubes
        import trimesh
        import numpy as np
        import torch
        import os
        import cv2

        device = next(self.parameters()).device
        print(f"[export_mesh] Generating density grid at {resolution}^3 resolution (Block-wise)...")
        sigmas = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        xs = torch.linspace(-self.bound, self.bound, resolution).to(device)
        ys = torch.linspace(-self.bound, self.bound, resolution).to(device)
        zs = torch.linspace(-self.bound, self.bound, resolution).to(device)

        X_batches = xs.split(S)
        Y_batches = ys.split(S)
        Z_batches = zs.split(S)

        pbar_cnt = 0
        total_batches = len(X_batches) * len(Y_batches) * len(Z_batches)
        
        with torch.no_grad():
            for xi, x_batch in enumerate(X_batches):
                for yi, y_batch in enumerate(Y_batches):
                    for zi, z_batch in enumerate(Z_batches):
                        grid_x, grid_y, grid_z = torch.meshgrid(x_batch, y_batch, z_batch, indexing='ij')
                        pts = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
                        
                        out = self.density(pts)
                        val = out['sigma'].detach().cpu().numpy()
                        
                        x_start, x_end = xi * S, xi * S + len(x_batch)
                        y_start, y_end = yi * S, yi * S + len(y_batch)
                        z_start, z_end = zi * S, zi * S + len(z_batch)
                        
                        sigmas[x_start:x_end, y_start:y_end, z_start:z_end] = val.reshape(len(x_batch), len(y_batch), len(z_batch))
                        
                        pbar_cnt += 1
                        if pbar_cnt % 50 == 0:
                            print(f"\r[export_mesh] Processing chunks... {pbar_cnt}/{total_batches}", end="")
        print("")

        level = float(threshold) if threshold is not None else float(self.density_thresh)
        print(f"[export_mesh] Marching Cubes with Level={level}")


        print(f"[export_mesh] Marching Cubes with Level={level} (Skinny Mode)")
        verts, faces = mcubes.marching_cubes(sigmas, level)
        verts = verts / (resolution - 1.0) * 2 * self.bound - self.bound
        if target_faces > 0 and faces.shape[0] > target_faces:
            print(f"[export_mesh] Heavy mesh ({faces.shape[0]} faces)! Decimating to {target_faces} (CLI option)...")
            try:
                mesh_temp = trimesh.Trimesh(vertices=verts, faces=faces)
                mesh_temp = mesh_temp.simplify_quadratic_decimation(int(target_faces))
                verts, faces = mesh_temp.vertices, mesh_temp.faces
                print(f"[export_mesh] Diet success! Current faces: {faces.shape[0]}")
            except Exception as e:
                print(f"[export_mesh] Diet failed: {e}")
        print("[export_mesh] Baking texture (albedo.png) using xatlas...")
        
        try:
            import xatlas
            import nvdiffrast.torch as dr
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            # UV Unwrapping
            atlas = xatlas.Atlas()
            atlas.add_mesh(verts, faces)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
            v_torch = torch.from_numpy(verts).float().to(device)
            f_torch = torch.from_numpy(faces.astype(np.int32)).int().to(device)
            h, w = 2048, 2048
            uv = vt * 2.0 - 1.0
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)

            if self.glctx is None:
                self.glctx = dr.RasterizeCudaContext()

            rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), ft, (h, w))
            xyzs, _ = dr.interpolate(v_torch.unsqueeze(0), rast, f_torch)
            mask, _ = dr.interpolate(torch.ones_like(v_torch[:, :1]).unsqueeze(0), rast, f_torch)

            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)
            feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
            if mask.any():
                xyzs = xyzs[mask]
                all_feats = []
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 64000, xyzs.shape[0])
                    results_ = self.density(xyzs[head:tail])
                    all_feats.append(results_['albedo'].float())
                    head += 64000
                feats[mask] = torch.cat(all_feats, dim=0)

            feats = feats.view(h, w, -1).cpu().numpy()
            mask = mask.view(h, w).cpu().numpy()
            feats = (feats * 255).astype(np.uint8)
            inpaint_region = binary_dilation(mask, iterations=3)
            inpaint_region[mask] = 0
            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=2)
            search_region[not_search_region] = 0
            
            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)
            
            if len(search_coords) > 0 and len(inpaint_coords) > 0:
                knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
                _, indices = knn.kneighbors(inpaint_coords)
                feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                tex_path = os.path.join(save_path, "albedo.png")
                cv2.imwrite(tex_path, cv2.cvtColor(feats, cv2.COLOR_RGB2BGR))
                print(f"[export_mesh] Saved texture to {tex_path}")
                obj_file = os.path.join(save_path, "mesh.obj")
                mtl_file = os.path.join(save_path, "mesh.mtl")
                
                with open(obj_file, "w") as fp:
                    fp.write(f'mtllib mesh.mtl \n')
                    for v in verts: fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
                    for v in vt_np: fp.write(f'vt {v[0]} {1 - v[1]} \n') 
                    fp.write(f'usemtl mat0 \n')
                    for i in range(len(faces)):
                        fp.write(f"f {faces[i, 0] + 1}/{ft_np[i, 0] + 1} {faces[i, 1] + 1}/{ft_np[i, 1] + 1} {faces[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

                with open(mtl_file, "w") as fp:
                    fp.write(f'newmtl mat0 \nmap_Kd albedo.png \n')
                
                print(f"[export_mesh] Saved mesh to {obj_file}")

        except Exception as e:
            print(f"[export_mesh] Texture baking failed: {e}")
            print("[export_mesh] Fallback: Saving mesh with vertex colors only.")

            verts_torch = torch.from_numpy(verts).float().to(device)
            colors = []
            for i in range(0, verts_torch.shape[0], 65536):
                _, albedo = self.common_forward(verts_torch[i:i+65536])
                colors.append(albedo)
            colors = torch.cat(colors, dim=0).clamp(0, 1).cpu().numpy()
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors)
            if save_path: mesh.export(os.path.join(save_path, "mesh.obj"))
            
            return mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        return mesh