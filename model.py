import torch as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.conv2d(in_channels, out_channels, 
                               3, stride, padding=1, bias=False)
        
    
    