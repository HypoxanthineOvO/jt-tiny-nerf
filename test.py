import jittor as jt
import numpy as np

x = np.arange(4)
y = np.arange(3)

i1,j1 = np.meshgrid(x,y,indexing='xy')

print(i1,"\n",j1)

x = jt.float32(x)
y = jt.float32(y)

i2,j2 = jt.meshgrid(x,y)

print(jt.transpose(i2),"\n",jt.transpose(j2))