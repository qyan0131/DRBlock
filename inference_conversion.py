# -*- coding: utf-8 -*-
"""
------ * Supplementary Material * ------ 
AAAI23 submission: 
    DR-Block: Convolutional Dense Reparameterization 
                for CNN Free Improvement
"""
import torch
from resnet_cifar import Network, ConvBN
from dr_block import DR_Block

weight_path = '' # load pretrained weights

model = Network([2, 2, 2, 2], 'basic', 'dr_block', 100)

# if we do not have a trained weight, then just use the random init weights.
if weight_path != '':
    state = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state)


with torch.no_grad():
        I = torch.randn((1,3,32,32))
        model.eval()
        # training-time model
        out1 = model(I)
        
        # reparameterization
        for n, m in model.named_modules():
            if isinstance(m, DR_Block):
                m.switch_deploy(True)
            if isinstance(m, ConvBN):
                m.switch_to_deploy()
                
        # inference-time model
        out2 = model(I)
        print('training-time model == inference-time model ?', ((abs(out1-out2)>1e-4).sum())==0) 
