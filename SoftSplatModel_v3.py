# + small size model
# + input resize

import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

# from FastFlowNet.FastFlowNet import FastFlowNet
# from FastFlowNet.FastFlowNet import FastFlowNet
from OpticalFlow.PWCNet import PWCNet
from softsplat import Softsplat
from torch.nn.functional import interpolate, grid_sample
from einops import repeat
import warnings 
warnings.filterwarnings('ignore')

from GridNet.model_pyramid import GridNet



# convert [0, 1] to [-1, 1]
def preprocess(x):
    return x * 2 - 1


# convert [-1, 1] to [0, 1]
def postprocess(x):
    return torch.clamp((x + 1) / 2, 0, 1)


backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([ tenHorizontal, tenVertical ], 1).cuda()
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

    # return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
    # return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', align_corners=True)

    # handle ver lines of warping:  border padding
    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)
# end


class SmallMaskNet(nn.Module):
    """A three-layer network for predicting mask"""
    def __init__(self, input, output):
        super(SmallMaskNet, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, output, 3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.conv3(x)
        return x

class Metric_Photo(torch.nn.Module):
    def __init__(self, init=1.0):
        super(Metric_Photo, self).__init__()
        self.paramScale = torch.nn.Parameter(init *torch.ones(1, 1, 1, 1))
    # end

    def forward(self, tenFirst, tenSecond, tenFlow):
        with torch.no_grad():
            err = torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenSecond, tenFlow), reduction='none').mean(1, True)
        return self.paramScale * err
    # end
# end

class FeatureExt(nn.Module):
    """A three-level pyramid feature extractor"""
    def __init__(self, in_planes=3, out_planes=96):
        super(FeatureExt, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU() #num_parameters=32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.PReLU() #num_parameters=32
        # pyramid 1
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.PReLU() #num_parameters=64
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.PReLU() #num_parameters=64
        # pyramid 2
        self.conv5 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1)
        self.relu5 = nn.PReLU()  #num_parameters=96
        self.conv6 = nn.Conv2d(96, out_planes, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.PReLU() #num_parameters=96
        # pyramid 3
    # end

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        fea1 = self.relu2(x)

        x = self.conv3(fea1)
        x = self.relu3(x)
        x = self.conv4(x)
        fea2 = self.relu4(x)

        x = self.conv5(fea2)
        x = self.relu5(x)
        x = self.conv6(x)
        fea3 = self.relu6(x)

        return fea1, fea2, fea3
        # return fea1
    # end
# end

class SoftSplatBaseline_v3(nn.Module):
    def __init__(self):
        super(SoftSplatBaseline_v3, self).__init__()

        # self.flow_predictor = FastFlowNet()
        # self.flow_predictor.load_state_dict(torch.load('/DATA/wangshen_data/CODES/softmax-splatting/FastFlowNet/checkpoints/fastflownet_ft_mix.pth'))

        self.flow_predictor = PWCNet()
        self.flow_predictor.load_state_dict(torch.load('./OpticalFlow/pwc-checkpoint.pt'))

        self.fwarp = Softsplat()
   
        self.splat_photo = Metric_Photo(-1.0)

        self.masknet = SmallMaskNet(3+3+32, 1)

        self.feaext = FeatureExt(3, 96)

        self.syn =  GridNet(in_chs=32, out_chs=3, grid_chs = [32, 64, 96])

    def splat_img_map(self, input_0, input_1, flow_pyramid_01, fea_pyramid_0, t):

        z_metric = self.splat_photo(preprocess(input_0),preprocess(input_1), flow_pyramid_01[0])

        warp_img = []
        warp_fea = []

        for scale, sub_flow_01 in enumerate(flow_pyramid_01): # 0,1,2
            
            sub_flow_0t =  t *  sub_flow_01

            fea = fea_pyramid_0[scale]

            img = torch.nn.functional.interpolate(input=input_0, scale_factor=0.5**scale, mode='bilinear', align_corners=False)

            z = torch.nn.functional.interpolate(input=z_metric, scale_factor=0.5** scale, mode='bilinear', align_corners=False)

            flow_t0 = self.fwarp(-1.0 * sub_flow_0t, sub_flow_0t, z)

            img_t0 = backwarp(img, flow_t0)
            fea_t0 = backwarp(fea, flow_t0)

            warp_img.append(img_t0)
            warp_fea.append(fea_t0)

        return warp_img, warp_fea
        

    def forward(self, x, target_t, input_resize=False, scale=0.5):

        input_0 = x[:, :, 0]
        input_1 = x[:, :, 1]

        b = x.shape[0]

        feature_0 = self.feaext(input_0)  # N 32 H W, N 64 H/2 W/2, 1 32 H/4 W/4
        feature_1 = self.feaext(input_1)
        
        target_t = target_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        flow_pyramid_01 = self.flow_predictor.forward_pyramid(input_0, input_1) # N 2 H W, N 2 H/2 W/2, N 2 H/4 W/4
        flow_pyramid_10 = self.flow_predictor.forward_pyramid(input_1, input_0)

        # splat_img_map(self, input_0, input_1, flow_pyramid_01, fea_pyramid_0, t):

        warped_img_0t, warped_fea_0t = self.splat_img_map(input_0, input_1, flow_pyramid_01, feature_0, target_t)
        warped_img_1t, warped_fea_1t = self.splat_img_map(input_1, input_0, flow_pyramid_10, feature_1, 1-target_t)

        input_level_0 = torch.cat([warped_img_0t[0], warped_fea_0t[0], warped_img_1t[0], warped_fea_1t[0]], dim=1)
        input_level_1 = torch.cat([warped_fea_0t[1], warped_fea_1t[1]], dim=1)
        input_level_2 = torch.cat([warped_fea_0t[2], warped_fea_1t[2]], dim=1)

        It_syn = self.syn(input_level_0, input_level_1, input_level_2)

        return It_syn



if __name__ == '__main__':
    '''
    Example Usage
    '''
    import numpy 
    frame0frame1 = torch.randn([1, 3, 2, 448, 256]).cuda()  # batch size 1, 3 RGB 
    # channels, 2 frame input, H x W of 448 x 256
    # frame0frame1 = torch.randn([1, 3, 2, 1920, 1080]).cuda()  # batch size 1, 3 RGB 
    target_t = torch.tensor([0.5]).cuda()
    model = SoftSplatBaseline_v3().cuda()
    # model.load_state_dict(torch.load('./ckpt/SoftSplatBaseline_Vimeo.pth'))

    def count_network_parameters(model):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        N = sum([numpy.prod(p.size()) for p in parameters])
        return N

    print("Num. of model parameters is :" + str(count_network_parameters(model)))
    if hasattr(model,'flow_predictor'):
        print("Num. of flow model parameters is :" +
              str(count_network_parameters(model.flow_predictor)))

    with torch.no_grad():

        for x in range(10):
            output = model(frame0frame1, target_t, False)


        import time
        start = time.time()
        for x in range(1000):
            output = model(frame0frame1, target_t, False)
        end = time.time()
        print('Time elapsed: {:.3f} ms'.format((end-start) ))
