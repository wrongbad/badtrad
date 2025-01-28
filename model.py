import torch



def rms_norm(x, weight, eps, dim=-1):
    x = x * torch.rsqrt(x.pow(2).mean(dim=dim, keepdim=True) + eps)
    return x * weight
    
class RmsNorm(torch.nn.Module):
    def __init__(self, n_embed, eps=1e-6, dim=-1):
        super().__init__()
        self.eps = eps
        self.dim = dim
        wshape = [n_embed] + [1] * (-1-dim)
        self.weight = torch.nn.Parameter(torch.ones(wshape))

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps, self.dim)



class Wave(torch.nn.Module):
    def __init__(self, chan, dil, drop=0.1):
        super().__init__()
        self.norm = RmsNorm(chan, dim=-2)
        self.drop = torch.nn.Dropout(p=drop)
        self.c0 = torch.nn.Conv1d(chan, chan, 2, dilation=dil)
        self.c1 = torch.nn.Conv1d(chan, chan, 2, dilation=dil)
        self.co = torch.nn.Conv1d(chan, chan, 1)

    def forward(self, x):
        xn = self.norm(x)
        a = self.c0(xn).sigmoid()
        b = self.c1(xn).tanh()
        y = self.co(a * b)
        return x[...,-y.shape[-1]:] + self.drop(y)
        


class WaveNet(torch.nn.Module):
    def __init__(self, nin, nout, chan=64, nlayer=[8,8,8]):
        super().__init__()
        self.win = torch.nn.Conv1d(nin, chan, 1)
        self.wout = torch.nn.Conv1d(chan, nout, 1)
        layers = []
        for nl in nlayer:
            for i in range(nl):
                layers += [Wave(chan, 2**i)]
        self.layers = torch.nn.Sequential(*layers)


    def forward(self, x):
        x = x.log1p()
        x = self.win(x)
        x = self.layers(x)
        x = self.wout(x)
        return x