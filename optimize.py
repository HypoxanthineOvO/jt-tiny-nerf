import jittor as jt
import numpy as np

def posenc(x,L_embed = 6):
    rets = [x]
    for i in range(L_embed):
        for fn in [jt.sin,jt.cos]:
            rets.append(fn(2.**i * x))
    return jt.concat(rets,-1)

def get_rays(H,W,focal,c2w):
    raw_i,raw_j = jt.meshgrid(
        jt.arange(end=W,dtype= jt.float32),
        jt.arange(H,dtype= jt.float32))
    i,j = jt.transpose(raw_i),jt.transpose(raw_j)
    dirs = jt.stack([
        (i - W * 0.5)/focal,
        -(j - H * 0.5)/focal,
        - jt.ones_like(i)],
        dim = -1)
    rays_d = jt.sum(dirs[...,None,:] * c2w[:3,:3], -1)
    rays_o = jt.expand(c2w[:3,-1], rays_d.shape)
    return rays_o,rays_d
    
def cumprod_exclusive(input_tensor:jt.Var,dim = -1):
    """ 
    Equal to tf.cumprod(...,exclusive = True)
    Only support dim = -1
    """
    tensor_shape = list(input_tensor.shape)
    tensor_shape[dim] = 1
    
    exclusive_tensor = jt.concat([jt.ones(tensor_shape),input_tensor[...,:-1]],dim=dim)
    
    ans = jt.cumprod(exclusive_tensor,dim)
    
    return ans    

def render_rays(net_fn, rays_o, rays_d, near, far, N_samples):
    def batchify(fn, chunk = 1024*32):
        return lambda \
            inputs: jt.concat(
                [fn(inputs[i:i+chunk]) for i in range(0,inputs.shape[0],chunk)]
                , 0)
    
    # Compute 3D Query Points
    z_vals = jt.linspace(near,far, N_samples)
    
    pts = rays_o[..., None, :] + rays_d[...,None, :] * z_vals[...,:,None]
    
    # Run Network
    pts_flat = jt.reshape(pts,[-1,3])
    pts_flat = posenc(pts_flat)
    raw = batchify(net_fn)(pts_flat)
    raw = jt.reshape(raw, list(pts.shape[:-1])+[4])
    
    # Compute Opacities and Colors
    sigma = jt.nn.relu(raw[...,3])
    rgb = jt.sigmoid(raw[...,:3])
    
    # Volume Rendering
    dists = jt.concat(
        [z_vals[..., 1:] - z_vals[...,:-1], jt.expand(jt.float32([1e10]),z_vals[...,:1].shape)],
        -1
    )
    alpha = 1.0 - jt.exp(-sigma * dists)
    weights = alpha * cumprod_exclusive(1.0-alpha+1e-10,-1)
    
    rgb_map = jt.sum(weights[...,None] * rgb, -2)
    depth_map = jt.sum(weights * z_vals, -1)
    acc_map = jt.sum(weights,-1)
    return rgb_map,depth_map,acc_map