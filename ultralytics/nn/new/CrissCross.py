import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B,H,W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1).cpu() #cpu上


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1).cpu())


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x).cpu()
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1).cpu()
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1).cpu()
        proj_key = self.key_conv(x).cpu()
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).cpu()
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).cpu()
        proj_value = self.value_conv(x).cpu()
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).cpu()
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).cpu()
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3).cpu()
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width).cpu()
        concate = self.softmax(torch.cat([energy_H, energy_W], 3)).cpu()

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height).cpu()
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width).cpu()
        att_W = att_W

        m = torch.bmm(proj_value_H.to(torch.half), att_H.permute(0, 2, 1).to(torch.half))
        n = m.view(m_batchsize,width,-1,height)
        out_H = n.permute(0,2,3,1).to(torch.half).cpu()
        out_W = torch.bmm(proj_value_W.to(torch.half), att_W.permute(0, 2, 1).to(torch.half)).view(m_batchsize,height,-1,width).permute(0,2,1,3).to(torch.half).cpu()

        out = out_H + out_W
        out = nn.Parameter(torch.zeros(1).cpu()).cpu() * out.cpu()
        out = out + x.cpu()
        return out.cpu()
  
