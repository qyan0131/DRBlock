# -*- coding: utf-8 -*-
"""
------ * Supplementary Material * ------ 
AAAI23 submission: 
    DR-Block: Convolutional Dense Reparameterization 
                for CNN Free Improvement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from repUtils import *



class DR_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, nonlinear=None):
        super(DR_Block, self).__init__()
        
        self.deploy=False
        self.converted = False
        
        in_ch = in_channels
        out_ch = out_channels
      
        s = stride
        k = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.s = s
        self.k = k
        self.g = groups
        self.identity = None
        self.p = kernel_size//2
        
        
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        
        if not self.deploy:
    
            self.c0 = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch//4, 1, padding=0, bias=False, stride=1),
                    nn.BatchNorm2d(in_ch//4)
                    )
            self.c1 = nn.Sequential(
                    nn.Conv2d(in_ch+1*in_ch//4, in_ch//4, 1, padding=0, bias=False, stride=1),
                    nn.BatchNorm2d(in_ch//4)
                    )
            self.c2 = nn.Sequential(
                    nn.Conv2d(in_ch+2*in_ch//4, in_ch//4, 1, padding=0, bias=False, stride=1),
                    nn.BatchNorm2d(in_ch//4)
                    )
            self.c3 = nn.Sequential(
                    nn.Conv2d(in_ch+3*in_ch//4, in_ch//4, 1, padding=0, bias=False, stride=1),
                    nn.BatchNorm2d(in_ch//4)
                    )
            self.tran = nn.Sequential(
                    nn.Conv2d(2*in_ch, out_ch, k, padding=0, bias=False, stride=s, groups=groups),
                    nn.BatchNorm2d(out_ch)
                    )
            
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False, stride=s),
                nn.BatchNorm2d(out_ch)
                )
        else:
            self.identity = nn.BatchNorm2d(num_features=out_channels) 
    
        
    def forward(self, x):

            
        if self.deploy is False:
            if self.identity is None:
                id_out = 0
            else:
                id_out = self.identity(x)
            if self.p != 0:
                x = F.pad(x, [self.p, self.p, self.p, self.p])
            c = self.c0(x)
            x = torch.cat([c, x], 1)
            c = self.c1(x)
            x = torch.cat([c, x], 1)
            c = self.c2(x)
            x = torch.cat([c, x], 1)
            c = self.c3(x)
            x = torch.cat([c, x], 1)
            x = self.tran(x)+id_out
        else:
            if not self.converted:
                raise RuntimeError('Please run reparameterize function first.')
            x = self.conv(x)
            print("Using reparameterized conv ...")
        return self.nonlinear(x)
    
    
    def reparameterize(self):
        # rep main stream
        r_cv0 = repConv(self.in_ch, self.in_ch//4, self.c0)
        r_xv0 = repDenseConcat(self.in_ch, 5*self.in_ch//4, r_cv0, None)
        
        r_cv1_l = repConv(self.in_ch, self.in_ch//4, self.c1)
        r_cv1 = rep1x1tokxk(self.in_ch, self.in_ch//4, r_xv0, r_cv1_l)
        r_xv1 = repDenseConcat(self.in_ch, 6*self.in_ch//4, r_cv1, r_xv0)
        
        r_cv2_l = repConv(self.in_ch, self.in_ch//4, self.c2)
        r_cv2 = rep1x1tokxk(self.in_ch, self.in_ch//4, r_xv1, r_cv2_l)
        r_xv2 = repDenseConcat(self.in_ch, 7*self.in_ch//4, r_cv2, r_xv1)
        
        r_cv3_l = repConv(self.in_ch, self.in_ch//4, self.c3)
        r_cv3 = rep1x1tokxk(self.in_ch, self.in_ch//4, r_xv2, r_cv3_l)
        r_xv3 = repDenseConcat(self.in_ch, 8*self.in_ch//4, r_cv3, r_xv2)
        
        r_tv_l = repConv(self.in_ch, self.in_ch//4, self.tran, k=self.k)
        r_tv = rep1x1tokxk(self.in_ch, self.out_ch, r_xv3, r_tv_l, k=self.k, s=self.s)
        
        # rep short cut
        if isinstance(self.identity, nn.BatchNorm2d):
            r_sc = repBNkxk(self.out_ch, self.identity)
        else:
            r_sc = repConv(self.in_ch, self.out_ch, self.identity)
            r_sc.weight.data = F.pad(r_sc.weight.data, [1,1,1,1])
        
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.in_ch, self.out_ch, self.k, padding=self.k//2, bias=True, stride=self.s, groups=self.g).eval()
        self.conv.weight.data = r_tv.weight.data+r_sc.weight.data
        self.conv.bias.data = r_tv.bias.data+r_sc.bias.data
        self.conv.eval()
        self.__delattr__('c0')
        self.__delattr__('c1')
        self.__delattr__('c2')
        self.__delattr__('c3')
        self.__delattr__('tran')
        self.__delattr__('identity')
        self.converted = True
        print("[INFO] Successfully reparameterized.")
        
    def switch_deploy(self, deploy):
        self.deploy = deploy
        if not deploy:
            return
        if self.converted:
            return
        self.reparameterize()
