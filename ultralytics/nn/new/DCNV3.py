# 改不通，要报错
from torch import nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # print(" Conv:c1="+str(c1)+" ") # 
        # print(" Conv:c2="+str(c2)+" ") # 

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DCNv3(nn.Module):
    def __init__(
            self, channels=64, kernel_size=3, stride=1,
            pad=1, dilation=1, group=4, offset_scale=1.0,
            act_layer='GELU', norm_layer='LN'):
        """
        DCNv3 Module
        :param channels     
        :param kernel_size  
        :param stride      
        :param pad     
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = 1
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale

        self.dw_conv = Conv(channels, channels, kernel_size, g=channels)
        self.offset = nn.Linear(
            channels,
            group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(
            channels,
            group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        dtype = x.dtype

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1).permute(0, 2, 3, 1)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1).type(dtype)

        x = DCNv3Function.apply(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256)
        x = self.output_proj(x)

        return x

class DCNV3_YoLo(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        
        self.conv = Conv(inc, ouc, k=1)
        self.dcnv3 = DCNv3(ouc, kernel_size=k, stride=s, group=g)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.dcnv3(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.bn(x))
        return x

# if isinstance(m, Detect):
#     s = 256  # 2x min stride
#     self.model.to(torch.device('cuda'))
#     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s).to(torch.device('cuda')))]).cpu()  # forward
#     self.model.cpu()
#     check_anchor_order(m)
#     m.anchors /= m.stride.view(-1, 1, 1)
#     self.stride = m.stride
#     self._initialize_biases()  # only run once
#     # print('Strides: %s' % m.stride.tolist())
# if isinstance(m, IDetect):
#     s = 256  # 2x min stride
#     self.model.to(torch.device('cuda'))
#     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s).to(torch.device('cuda')))]).cpu()  # forward
#     self.model.cpu()
#     check_anchor_order(m)
#     m.anchors /= m.stride.view(-1, 1, 1)
#     self.stride = m.stride
#     self._initialize_biases()  # only run once
#     # print('Strides: %s' % m.stride.tolist())
# if isinstance(m, IAuxDetect):
#     s = 256  # 2x min stride
#     self.model.to(torch.device('cuda'))
#     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s).to(torch.device('cuda')))[:4]]).cpu()  # forward
#     self.model.cpu()
#     #print(m.stride)
#     check_anchor_order(m)
#     m.anchors /= m.stride.view(-1, 1, 1)
#     self.stride = m.stride
#     self._initialize_aux_biases()  # only run once
#     # print('Strides: %s' % m.stride.tolist())