import os, sys
import time
import jittor as jt

import numpy as np
import matplotlib.pyplot as plt

from model import NeRF
from optimize import *

if jt.has_cuda:
    jt.flags.use_cuda = 1

data = np.load('tiny_nerf_data.npz')
images = jt.float32(data['images'])
poses = jt.float32(data['poses'])
focal = jt.float32(data['focal'])
H, W = images.shape[1:3]
print(images.shape, poses.shape, focal)

testimg, testpose = images[101], poses[101]
images = images[:100,...,:3]
poses = poses[:100]

plt.imshow(testimg)
plt.imsave("./testimg.jpg",testimg.numpy())
plt.show()

L_embed = 6
model = NeRF()
N_samples = 64
N_iters = 1000
psnrs = []
iternums = []
i_plot = 25

optimizer = jt.nn.Adam(model.parameters(),lr = 5e-4)

t = time.time()
for i in range(N_iters + 1):
    img_i = np.random.randint(images.shape[0])
    target = images[img_i]
    pose = poses[img_i]
    rays_o, rays_d = get_rays(H, W, focal, pose)
    rgb,depth,acc = render_rays(model,rays_o, rays_d, near=2., far=6., N_samples=N_samples)
    loss = jt.mean(jt.sqr(rgb - target))
    optimizer.step(loss)
    
    if i%i_plot==0:
        print(i, (time.time() - t) / i_plot, 'secs per iter')
        t = time.time()
        
        # Render the holdout view for logging
        rays_o, rays_d = get_rays(H, W, focal, testpose)
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        loss = jt.mean(jt.sqr(rgb - target))
        psnr = -10. * jt.log(loss) / jt.log(10.)

        psnrs.append(psnr.numpy())
        iternums.append(i)
        
        plt.imsave(f"./output/Iteration{i}.jpg",rgb.numpy())
        print(f"PSNR = {psnr}")
        
print("DONE!")