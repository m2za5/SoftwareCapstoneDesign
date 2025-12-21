import torch
import argparse
import pandas as pd
import sys

from nerf.provider import NeRFDataset
from nerf.utils import *
from nerf.utils import Trainer

import csv, os
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
import pandas as pd
import torch
import trimesh
from pytorch3d.loss import chamfer_distance

import matplotlib.pyplot as plt

import json

import re

import time

# torch.autograd.set_detect_anomaly(True)
def _sobel_gray(x):
    # x: [H,W,3] in [0,1] -> gray
    g = 0.299*x[...,0] + 0.587*x[...,1] + 0.114*x[...,2]
    gx = np.pad(g[:,1:] - g[:,:-1], ((0,0),(0,1)), mode='edge')
    gy = np.pad(g[1:,:] - g[:-1,:], ((0,1),(0,0)), mode='edge')
    return np.sqrt(gx*gx + gy*gy)

def _deltaE76(a,b):
    # a,b: [H,W,3] RGB in [0,1]
    import skimage.color as skc
    La = skc.rgb2lab(a); Lb = skc.rgb2lab(b)
    de = np.sqrt(((La-Lb)**2).sum(axis=-1))
    return de.mean()

if __name__ == '__main__':
    # See https://stackoverflow.com/questions/27433316/how-to-get-argparse-to-read-arguments-from-a-file-with-an-option-rather-than-pre
    class LoadFromFile (argparse.Action):
        def __call__ (self, parser, namespace, values, option_string = None):
            with values as f:
                # parse arguments in the file and store them in the target namespace
                parser.parse_args(f.read().split(), namespace)

    parser = argparse.ArgumentParser()
    parser.add_argument('--init_mesh', type=str, default=None, help="initialize NeRF geometry from mesh file (.obj/.ply)")
    parser.add_argument('--file', type=open, action=LoadFromFile, help="specify a file filled with more arguments")
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray")
    parser.add_argument('-O2', action='store_true', help="equals --backbone vanilla")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--six_views', action='store_true', help="six_views mode: save the images of the six views")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--test_interval', type=int, default=100, help="test on the test set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', default=None)

    parser.add_argument('--image', default=None, help="image prompt")
    parser.add_argument('--image_config', default=None, help="image config csv")

    parser.add_argument('--known_view_interval', type=int, default=4, help="train default view with RGB loss every & iters, only valid if --image is not None.")

    parser.add_argument('--IF', action='store_true', help="experimental: use DeepFloyd IF as the guidance model for nerf stage")

    parser.add_argument('--guidance', type=str, nargs='*', default=['SD'], help='guidance model')
    parser.add_argument('--guidance_scale', type=float, default=100, help="diffusion model classifier-free guidance scale")

    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--mcubes_resolution', type=int, default=256, help="mcubes resolution for extracting mesh")
    parser.add_argument('--decimate_target', type=int, default=5e4, help="target face number for mesh decimation")

    parser.add_argument('--dmtet', action='store_true', help="use dmtet finetuning")
    parser.add_argument('--tet_grid_size', type=int, default=128, help="tet grid size")
    parser.add_argument('--init_with', type=str, default='', help="ckpt to init dmtet")
    parser.add_argument('--lock_geo', action='store_true', help="disable dmtet to learn geometry")

    ## Perp-Neg options
    parser.add_argument('--perpneg', action='store_true', help="use perp_neg")
    parser.add_argument('--negative_w', type=float, default=-2, help="The scale of the weights of negative prompts. A larger value will help to avoid the Janus problem, but may cause flat faces. Vary between 0 to -4, depending on the prompt")
    parser.add_argument('--front_decay_factor', type=float, default=2, help="decay factor for the front prompt")
    parser.add_argument('--side_decay_factor', type=float, default=10, help="decay factor for the side prompt")

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--ckpt', type=str, default='latest', help="possible options are ['latest', 'scratch', 'best', 'latest_model']")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--taichi_ray', action='store_true', help="use taichi raymarching")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--latent_iter_ratio', type=float, default=0.2, help="training iters that only use albedo shading")
    parser.add_argument('--albedo_iter_ratio', type=float, default=0, help="training iters that only use albedo shading")
    parser.add_argument('--min_ambient_ratio', type=float, default=0.5, help="minimum ambient ratio to use in lambertian shading")
    parser.add_argument('--textureless_ratio', type=float, default=0, help="ratio of textureless shading")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument('--jitter_center', type=float, default=0.2, help="amount of jitter to add to sampled camera pose's center (camera location)")
    parser.add_argument('--jitter_target', type=float, default=0.2, help="amount of jitter to add to sampled camera pose's target (i.e. 'look-at')")
    parser.add_argument('--jitter_up', type=float, default=0.02, help="amount of jitter to add to sampled camera pose's up-axis (i.e. 'camera roll')")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0, help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--grad_clip', type=float, default=-1, help="clip grad of all grad to this limit, negative value disables it")
    parser.add_argument('--grad_clip_rgb', type=float, default=-1, help="clip grad of rgb space grad to this limit, negative value disables it")
    # model options
    parser.add_argument('--bg_radius', type=float, default=1.4, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_activation', type=str, default='exp', choices=['softplus', 'exp'], help="density activation function")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=5, help="max (center) density for the density blob")
    parser.add_argument('--blob_radius', type=float, default=0.2, help="control the radius for the density blob")
    # network backbone
    parser.add_argument('--backbone', type=str, default='grid', choices=['grid_tcnn', 'grid', 'vanilla', 'grid_taichi'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # try this if CUDA OOM
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    # rendering resolution in training, increase these for better quality / decrease these if CUDA OOM even if --vram_O enabled.
    parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--known_view_scale', type=float, default=1.5, help="multiply --h/w by this for known view rendering")
    parser.add_argument('--known_view_noise_scale', type=float, default=2e-3, help="random camera noise added to rays_o and rays_d")
    parser.add_argument('--dmtet_reso_scale', type=float, default=8, help="multiply --h/w by this for dmtet finetuning")
    parser.add_argument('--batch_size', type=int, default=1, help="images to render per batch using NeRF")

    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.01, help="minimum near distance for camera")

    parser.add_argument('--radius_range', type=float, nargs='*', default=[3.0, 3.5], help="training camera radius range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[45, 105], help="training camera range along the polar angles (i.e. up and down). See advanced.md for details.")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="training camera range along the azimuth angles (i.e. left and right). See advanced.md for details.")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[10, 30], help="training camera fovy range")

    parser.add_argument('--default_radius', type=float, default=3.2, help="radius for the default view")
    parser.add_argument('--default_polar', type=float, default=90, help="polar for the default view")
    parser.add_argument('--default_azimuth', type=float, default=0, help="azimuth for the default view")
    parser.add_argument('--default_fovy', type=float, default=20, help="fovy for the default view")

    parser.add_argument('--progressive_view', action='store_true', help="progressively expand view sampling range from default to full")
    parser.add_argument('--progressive_view_init_ratio', type=float, default=0.2, help="initial ratio of final range, used for progressive_view")
    
    parser.add_argument('--progressive_level', action='store_true', help="progressively increase gridencoder's max_level")

    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help="stable diffusion time steps range")
    parser.add_argument('--dont_override_stuff',action='store_true', help="Don't override t_range, etc.")


    ### regularizations
    parser.add_argument('--lambda_entropy', type=float, default=1e-3, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale for total variation")
    parser.add_argument('--lambda_wd', type=float, default=0, help="loss scale")

    parser.add_argument('--lambda_mesh_normal', type=float, default=0.5, help="loss scale for mesh normal smoothness")
    parser.add_argument('--lambda_mesh_laplacian', type=float, default=0.5, help="loss scale for mesh laplacian")

    parser.add_argument('--lambda_guidance', type=float, default=1, help="loss scale for SDS")
    parser.add_argument('--lambda_rgb', type=float, default=1000, help="loss scale for RGB")
    parser.add_argument('--lambda_mask', type=float, default=500, help="loss scale for mask (alpha)")
    parser.add_argument('--lambda_normal', type=float, default=0, help="loss scale for normal map")
    parser.add_argument('--lambda_depth', type=float, default=10, help="loss scale for relative depth")
    parser.add_argument('--lambda_2d_normal_smooth', type=float, default=0, help="loss scale for 2D normal image smoothness")
    parser.add_argument('--lambda_3d_normal_smooth', type=float, default=0, help="loss scale for 3D normal image smoothness")

    ### debugging options
    parser.add_argument('--save_guidance', action='store_true', help="save images of the per-iteration NeRF renders, added noise, denoised (i.e. guidance), fully-denoised. Useful for debugging, but VERY SLOW and takes lots of memory!")
    parser.add_argument('--save_guidance_interval', type=int, default=10, help="save guidance every X step")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=20, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    parser.add_argument('--zero123_config', type=str, default='./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml', help="config file for zero123")
    parser.add_argument('--zero123_ckpt', type=str, default='pretrained/zero123/zero123-xl.ckpt', help="ckpt for zero123")
    parser.add_argument('--zero123_grad_scale', type=str, default='angle', help="whether to scale the gradients based on 'angle' or 'None'")

    parser.add_argument('--dataset_size_train', type=int, default=100, help="Length of train dataset i.e. # of iterations per epoch")
    parser.add_argument('--dataset_size_valid', type=int, default=8, help="# of frames to render in the turntable video in validation")
    parser.add_argument('--dataset_size_test', type=int, default=100, help="# of frames to render in the turntable video at test time")

    parser.add_argument('--exp_start_iter', type=int, default=None, help="start iter # for experiment, to calculate progressive_view and progressive_level")
    parser.add_argument('--exp_end_iter', type=int, default=None, help="end iter # for experiment, to calculate progressive_view and progressive_level")

    # Loss / Input weighting flags
    parser.add_argument('--lambda_seam', type=float, default=0.0, help="weight for seam-aware continuity loss")
    parser.add_argument('--lambda_perc', type=float, default=0.0, help="weight for VGG perceptual texture loss")
    parser.add_argument('--lambda_chroma', type=float, default=0.0, help="weight for chromaticity (Lab a,b) consistency")
    parser.add_argument('--lambda_norm_refine', type=float, default=0.0, help="weight for normal-guided refinement loss")

    parser.add_argument('--mask_weighted_sds', action='store_true', help="reweight SDS gradient with seam/edge mask")
    parser.add_argument('--alpha_sds', type=float, default=0.7, help="SDS weight for seam mask")
    parser.add_argument('--beta_sds', type=float, default=0.3, help="SDS weight for edge mask")
    parser.add_argument('--mesh_prior_scale', type=float, default=0.2,
                    help="weight of Shap-E density_init prior in sigma_raw")
    parser.add_argument('--mesh_prior_max_sigma', type=float, default=20.0,
                    help="max sigma after applying mesh prior")

    
    opt = parser.parse_args()
    print("[DEBUG] density_thresh =", opt.density_thresh)


    opt.workspace = os.path.abspath(opt.workspace)
    print(f"[DEBUG] CWD={os.getcwd()}  WORKSPACE={opt.workspace}")
    os.makedirs(opt.workspace, exist_ok=True)
    os.makedirs(os.path.join(opt.workspace, "results"), exist_ok=True)
    os.makedirs(os.path.join(opt.workspace, "mesh"), exist_ok=True)
    with open(os.path.join(opt.workspace, "_write_test.txt"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("[DEBUG] Workspace write test OK")

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True

    elif opt.O2:
        opt.fp16 = True
        opt.backbone = 'vanilla'
        opt.progressive_level = True

    if opt.IF:
        if 'SD' in opt.guidance:
            opt.guidance.remove('SD')
            opt.guidance.append('IF')
        opt.latent_iter_ratio = 0 # must not do as_latent

    opt.images, opt.ref_radii, opt.ref_polars, opt.ref_azimuths, opt.zero123_ws = [], [], [], [], []
    opt.default_zero123_w = 1

    opt.exp_start_iter = opt.exp_start_iter or 0
    opt.exp_end_iter = opt.exp_end_iter or opt.iters

    # parameters for image-conditioned generation
    if opt.image is not None or opt.image_config is not None:

        if opt.text is None:
            # use zero123 guidance model when only providing image
            opt.guidance = ['zero123']
            if not opt.dont_override_stuff:
                opt.fovy_range = [opt.default_fovy, opt.default_fovy] # fix fov as zero123 doesn't support changing fov
                opt.guidance_scale = 5
                opt.lambda_3d_normal_smooth = 10
        else:
            # use stable-diffusion when providing both text and image
            opt.guidance = ['SD', 'clip']
            
            if not opt.dont_override_stuff:
                opt.guidance_scale = 10
                opt.t_range = [0.2, 0.6]
                opt.known_view_interval = 2
                opt.lambda_3d_normal_smooth = 20
            opt.bg_radius = -1

        # smoothness
        opt.lambda_entropy = 1
        opt.lambda_orient = 1

        # latent warmup is not needed
        opt.latent_iter_ratio = 0
        if not opt.dont_override_stuff:
            opt.albedo_iter_ratio = 0
            
            # make shape init more stable
            opt.progressive_view = True
            opt.progressive_level = True

        if opt.image is not None:
            opt.images += [opt.image]
            opt.ref_radii += [opt.default_radius]
            opt.ref_polars += [opt.default_polar]
            opt.ref_azimuths += [opt.default_azimuth]
            opt.zero123_ws += [opt.default_zero123_w]

        if opt.image_config is not None:
            # for multiview (zero123)
            conf = pd.read_csv(opt.image_config, skipinitialspace=True)
            opt.images += list(conf.image)
            opt.ref_radii += list(conf.radius)
            opt.ref_polars += list(conf.polar)
            opt.ref_azimuths += list(conf.azimuth)
            opt.zero123_ws += list(conf.zero123_weight)
            if opt.image is None:
                opt.default_radius = opt.ref_radii[0]
                opt.default_polar = opt.ref_polars[0]
                opt.default_azimuth = opt.ref_azimuths[0]
                opt.default_zero123_w = opt.zero123_ws[0]

    # reset to None
    if len(opt.images) == 0:
        opt.images = None

    # default parameters for finetuning
    if opt.dmtet:

        opt.h = int(opt.h * opt.dmtet_reso_scale)
        opt.w = int(opt.w * opt.dmtet_reso_scale)
        opt.known_view_scale = 1

        if not opt.dont_override_stuff:            
            opt.t_range = [0.02, 0.50] # ref: magic3D

        if opt.images is not None:

            opt.lambda_normal = 0
            opt.lambda_depth = 0

            if opt.text is not None and not opt.dont_override_stuff:
                opt.t_range = [0.20, 0.50]

        # assume finetuning
        opt.latent_iter_ratio = 0
        opt.albedo_iter_ratio = 0
        opt.progressive_view = False
        # opt.progressive_level = False

    # record full range for progressive view expansion
    if opt.progressive_view:
        if not opt.dont_override_stuff:
            # disable as they disturb progressive view
            opt.jitter_pose = False
            
        opt.uniform_sphere_rate = 0
        # back up full range
        opt.full_radius_range = opt.radius_range
        opt.full_theta_range = opt.theta_range
        opt.full_phi_range = opt.phi_range
        opt.full_fovy_range = opt.fovy_range

    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == 'grid_tcnn':
        from nerf.network_grid_tcnn import NeRFNetwork
    elif opt.backbone == 'grid_taichi':
        opt.cuda_ray = False
        opt.taichi_ray = True
        import taichi as ti
        from nerf.network_grid_taichi import NeRFNetwork
        taichi_half2_opt = True
        taichi_init_args = {"arch": ti.cuda, "device_memory_GB": 4.0}
        if taichi_half2_opt:
            taichi_init_args["half2_vectorization"] = True
        ti.init(**taichi_init_args)
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    print(opt)

    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(opt).to(device)

    if opt.init_mesh is not None and opt.init_mesh != '':
        import trimesh
        print(f"Loading initial mesh from {opt.init_mesh} ...")
        mesh = trimesh.load(opt.init_mesh, force='mesh', skip_material=True, process=False)
        mesh.metadata = mesh.metadata or {}
        mesh.metadata["file_path"] = opt.init_mesh  # 캐시 키 고정
        mesh.metadata["name"] = os.path.splitext(os.path.basename(opt.init_mesh))[0]
        print("[DEBUG] mesh.metadata[file_path] =", mesh.metadata.get("file_path"))

        try:
            model.init_from_mesh(mesh)
            print("Mesh initialized successfully.")
        except Exception as e:
            print(f"init_from_mesh() not implemented ({e}), fallback to voxelization.")
            import open3d as o3d
            v = o3d.io.read_triangle_mesh(opt.init_mesh)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(v, voxel_size=0.01)
            if hasattr(model, 'voxel_init'):
                model.voxel_init(voxel_grid)
            else:
                print("model.voxel_init() not found — mesh loaded but not voxelized.")

    if opt.dmtet and opt.init_with != '':
        if opt.init_with.endswith('.pth'):
            # load pretrained weights to init dmtet
            state_dict = torch.load(opt.init_with, map_location=device)
            model.load_state_dict(state_dict['model'], strict=False)
            if opt.cuda_ray:
                model.mean_density = state_dict['mean_density']
            model.init_tet()
        else:
            # assume a mesh to init dmtet (experimental, not working well now!)
            import trimesh
            mesh = trimesh.load(opt.init_with, force='mesh', skip_material=True, process=False)
            model.init_tet(mesh=mesh)

    print(model)

    if opt.six_views:
        guidance = None # no need to load guidance model at test

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        test_loader = NeRFDataset(opt, device=device, type='six_views', H=opt.H, W=opt.W, size=6).dataloader(batch_size=1)
        trainer.test(test_loader, write_video=False)

        if opt.save_mesh:
            trainer.save_mesh()

    elif opt.test:
        guidance = None # no need to load guidance model at test

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.gui:
            from nerf.gui import NeRFGUI
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader(batch_size=1)
            trainer.test(test_loader)

            if opt.save_mesh:
                trainer.save_mesh()

    else:

        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.dataset_size_train * opt.batch_size).dataloader()

        if opt.optim == 'adan':
            from optimizer import Adan
            # Adan usually requires a larger LR
            optimizer = lambda model: Adan(model.get_params(opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else: # adam
            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        if opt.backbone == 'vanilla':
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        else:
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
            # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        guidance = nn.ModuleDict()

        if 'SD' in opt.guidance:
            from guidance.sd_utils import StableDiffusion
            guidance['SD'] = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key, opt.t_range)

        if 'IF' in opt.guidance:
            from guidance.if_utils import IF
            guidance['IF'] = IF(device, opt.vram_O, opt.t_range)

        if 'zero123' in opt.guidance:
            from guidance.zero123_utils import Zero123
            guidance['zero123'] = Zero123(device=device, fp16=opt.fp16, config=opt.zero123_config, ckpt=opt.zero123_ckpt, vram_O=opt.vram_O, t_range=opt.t_range, opt=opt)

        if 'clip' in opt.guidance:
            from guidance.clip_utils import CLIP
            guidance['clip'] = CLIP(device)

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, scheduler_update_every_step=True)

        trainer.default_view_data = train_loader._data.get_default_view_data()

        if opt.gui:
            from nerf.gui import NeRFGUI
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()

        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=opt.dataset_size_valid).dataloader(batch_size=1)
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader(batch_size=1)

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, test_loader, max_epoch)

            if opt.save_mesh:
                trainer.save_mesh()

        if opt.iters == 0 and opt.save_mesh:
            trainer.save_mesh()
        
        try:
            from skimage import measure
            import trimesh
            print("Fallback mesh export activated...")

            out_dir = os.path.join(opt.workspace, "mesh")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "mesh_fallback.obj")

            # density_grid이 없을 때도 export 가능하도록 보정
            grid = None
            if hasattr(model, "density_grid") and model.density_grid is not None:
                grid = model.density_grid.detach().cpu().numpy()
            elif hasattr(model, "density_init"):  # Shap-E 초기 mesh density
                grid = model.density_init.detach().cpu().numpy()

            if grid is not None:
                level = np.percentile(grid, 90)
                print(f"[INFO] marching cubes on grid {grid.shape}, level={level:.4f}")
                verts, faces, _, _ = measure.marching_cubes(grid, level=level)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                mesh.export(out_path)
                print(f"mesh_fallback.obj exported to: {out_path}")
            else:
                print("No usable density grid found — fallback skipped.")
        except Exception as e:
            print(f"[ERROR] mesh export failed: {e}")

        loss_log_path = os.path.join(opt.workspace, "loss_log.csv")
        os.makedirs(opt.workspace, exist_ok=True)
        if not os.path.exists(loss_log_path):
            with open(loss_log_path, "w", newline="") as f:
                csv.writer(f).writerow(["iter", "total_loss", "sds_loss", "rgb_loss"])

        # 렌더링 결과 평가
                # 렌더링 결과 평가
        pred_dir = os.path.join(opt.workspace, "results")
        gt_dir = os.path.join(opt.workspace, "gt")
        out_csv = os.path.join(opt.workspace, "results", "metrics_table.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        lpips_fn = lpips.LPIPS(net='vgg')
        rows = []
        for fname in os.listdir(pred_dir):
            if fname.endswith(".png") and os.path.exists(os.path.join(gt_dir, fname)):
                gt = np.array(Image.open(os.path.join(gt_dir, fname))).astype(np.float32)/255
                pred = np.array(Image.open(os.path.join(pred_dir, fname))).astype(np.float32)/255

                psnr_v = psnr(gt, pred, data_range=1.0)
                ssim_v = ssim(gt, pred, channel_axis=-1, data_range=1.0)
                lpips_v = lpips_fn(
                    torch.tensor(pred).permute(2,0,1).unsqueeze(0)*2-1,
                    torch.tensor(gt).permute(2,0,1).unsqueeze(0)*2-1
                ).item()

                # NEW: 색 차이, 엣지 쪽 오차
                de_v = _deltaE76(gt, pred)
                edge_mask = _sobel_gray(gt)
                edge_mse = ((gt - pred)**2).mean(axis=-1)
                edge_mse = (edge_mse * (edge_mask / (edge_mask.max()+1e-8))).mean()

                rows.append([fname, psnr_v, ssim_v, lpips_v, de_v, edge_mse])

        if rows:
            df = pd.DataFrame(rows, columns=["filename","PSNR","SSIM","LPIPS","DeltaE","EdgeMSE"])
            df.to_csv(out_csv, index=False)
            print(f"Saved image metrics to {out_csv}")
        else:
            print("No comparable PNGs found in results/gt")


        # === 강제 메시 저장 보정 ===
        mesh_dir = os.path.join(opt.workspace, "mesh")
        mesh_path = os.path.join(mesh_dir, "mesh.obj")
        os.makedirs(mesh_dir, exist_ok=True)

        try:
            if hasattr(model, "export_mesh"):
                print("⚙️ Exporting mesh directly from model...")
                mesh = model.export_mesh(resolution=opt.mcubes_resolution)
                if mesh is not None:
                    mesh.export(mesh_path)
                    print(f"Mesh exported successfully to {mesh_path}")
                else:
                    print("export_mesh() returned None (empty density field)")
            else:
                print(" model.export_mesh() not found — mesh export not supported in this build")
        except Exception as e:
            print(f"Mesh export failed: {e}")


        # Chamfer Distance 계산
        pred_mesh_path = os.path.join(opt.workspace, "mesh", "mesh.obj")
        gt_mesh_path = os.path.join(opt.workspace, "gt", "gt.obj")
        if os.path.exists(pred_mesh_path) and os.path.exists(gt_mesh_path):
            pred_mesh = trimesh.load(pred_mesh_path)
            gt_mesh = trimesh.load(gt_mesh_path)
            pred_pts = torch.tensor(pred_mesh.sample(5000)).unsqueeze(0)
            gt_pts   = torch.tensor(gt_mesh.sample(5000)).unsqueeze(0)
            cd, _ = chamfer_distance(pred_pts, gt_pts)
            print(f"Chamfer Distance: {cd.item():.6f}")
            with open(os.path.join(opt.workspace, "results", "chamfer.txt"), "w") as f:
                f.write(f"{cd.item():.6f}")
        else:
            print("Skipped Chamfer Distance — mesh.obj or gt.obj not found")


        log_path = os.path.join(opt.workspace, "log.txt")
        loss_log_path = os.path.join(opt.workspace, "loss_log.csv")
        os.makedirs(opt.workspace, exist_ok=True)

        # 헤더 만들기 (이미 있으면 건너뜀)
        if not os.path.exists(loss_log_path):
            with open(loss_log_path, "w", newline="") as f:
                csv.writer(f).writerow(["iter", "total_loss", "sds_loss", "rgb_loss"])


        # 로그 파일에서 손실 값 추출
        """
        pattern = re.compile(r"Iter\s+(\d+).*?total_loss=(\S+).*?sds_loss=(\S+).*?rgb_loss=(\S+)")
        rows = []
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    m = pattern.search(line)
                    if m:
                        iter_i, total, sds, rgb = m.groups()
                        rows.append([int(iter_i), float(total), float(sds), float(rgb)])

        if rows:
            with open(loss_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            print(f"Logged {len(rows)} training steps to loss_log.csv")
        else:
            print("No iteration-level loss logs found in log.txt")

        # 손실 곡선 시각화
        if os.path.exists(loss_log_path):
            df = pd.read_csv(loss_log_path)
            if not df.empty:
                plt.figure(figsize=(6, 3))
                plt.plot(df['iter'], df['total_loss'], label='Total Loss')
                plt.xlabel("Iteration")
                plt.ylabel("Total Loss")
                plt.title("Training Stability")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(opt.workspace, "results", "loss_curve.png"), dpi=300)
                print("Saved loss curve to results/loss_curve.png")
        """
