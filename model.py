import jittor as jt
import jittor.nn as nn

class NeRF(jt.Module):
    def __init__(self,D = 8, W = 256,input_ch = 3+3*2*6):
        super(NeRF,self).__init__()
        
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = 4
        self.skips = [4]
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch,self.W)] + [
                nn.Linear(self.W,self.W) if i not in self.skips
                else nn.Linear(self.W + self.input_ch,self.W)
                for i in range(self.D - 1)
            ]
        )
        self.output_linear = nn.Linear(W, self.output_ch)
    
    def execute(self, x) :
        h = x
        for i,l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = nn.relu(h)
            if i in self.skips:
                h = jt.concat([x,h], -1)
        outputs = self.output_linear(h)
        return outputs