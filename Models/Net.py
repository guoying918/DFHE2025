
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.DyfusionNet import DeformFusion
from Models.SQHENet import *

IN_DIM = 100
OUT_DIM = 128 
####################################################
def repeat(x):
    if isinstance(x, (tuple, list)):
        return x
    return [x] * 3

def to_cuda(tensor_dict):
    return [tensor.cuda() for tensor in tensor_dict]

def main_conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
    )
    return layer

class Mapping(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dim, out_dim, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(OUT_DIM)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

class S3Conv(nn.Module):
    # deep wise then point wise
    def __init__(self, in_dim, out_dim, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.mapping = Mapping(IN_DIM, OUT_DIM)
        self.conv1 = main_conv3x3x3(in_dim, out_dim)
        self.conv2 = main_conv3x3x3(in_dim, out_dim)
        k, s, p = repeat(k), repeat(s), repeat(p)

        padding_mode = 'zeros'
        self.dw_conv = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, (1, k[1], k[2]), (1, s[1], s[2]), (0, p[1], p[2]), bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
            nn.Conv3d(out_dim, out_dim, (1, k[1], k[2]), 1, (0, p[1], p[2]), bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
            nn.Conv3d(out_dim, out_dim, (1, k[1], k[2]), 1, (0, p[1], p[2]), bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
        )
        self.pw_conv = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, (k[0], 1, 1), (s[0], 1, 1), (p[0], 0, 0), bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
            nn.Conv3d(out_dim, out_dim, (k[0], 1, 1), (s[0], 1, 1), (p[0], 0, 0), bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_2 = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        """
        x: torch.tensor(N,C,H,W)
        """
        x = self.mapping(x)
        y = x
        x = F.relu(self.conv1(x.unsqueeze(1)), inplace=False)
        x = self.conv2(x)
        
        x1 = self.dw_conv(x)
        x2 = self.pw_conv(x)
        x_s2 = x1[:,0,:,:,:] + x2[:,0,:,:,:]

        # multi-dimension feature integration module（MFI）
        x_c = self.adaptive_pool(x_s2).sigmoid() # b,c,1,1
        s1_c = x_c * x_s2 # b,c,h,w

        x_h = x_s2.permute(0,2,1,3) # b,h,c,w
        s0_3_h = self.adaptive_pool(x_h).sigmoid() # b,h,1,1 -> b,h,1,1
        s1_h = (s0_3_h * x_h).permute(0,2,1,3) # b,h,c,w -> b,c,h,w
        
        x_w = x_s2.permute(0,3,2,1) # b,w,h,c
        s0_3_w = self.adaptive_pool(x_w).sigmoid() # b,w,1,1 -> b,w,1,1
        s1_w = (s0_3_w * x_w).permute(0,3,2,1) # b,w,h,c -> b,c,h,w
        
        x_out = y + self.conv_2((s1_c + s1_h + s1_w).unsqueeze(1)).squeeze(1)
        
        return x_out

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.mapping_1 = S3Conv(1,1)
        self.mapping_2 = S3Conv(1,1)
        self.module_fusion = DeformFusion(128)

    # input_dim: dict,
    def forward(self, x):
        features_dict = {}
        mappings = [self.mapping_1, self.mapping_2]

        for i, key in enumerate(x): 
            features_dict[key] = mappings[i](x[key]) 

        out_temp_list = list(features_dict.values()) # 输入[17*17对应的特征图，9*9对应的特征图]
        out_feature = self.module_fusion(to_cuda(out_temp_list)) # (N,C)
        
        return out_feature

class feature_encode(nn.Module):
    def __init__(self, TEST_CLASS_NUM):
        super(feature_encode, self).__init__()
        self.network = Network()
        self.cross_module = Network_SQHE(TEST_CLASS_NUM)

        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if isinstance(m, S3Conv):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, support, query, support_label, query_label, domain):
        """
        support, query: dict(), key():patch_size, values:()
        """
        s_feature = self.network(support)
        q_feature = self.network(query)
        support_proto, query_Enhance, support_per, enhance_loss = self.cross_module(s_feature, support_label, q_feature, query_label, domain)

        return support_proto, query_Enhance, support_per, enhance_loss

# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     feature_shapes = [
#         (45, 100, 5, 5),
#         (45, 100, 17, 17)
#     ]
#     # feature_shapes = to_cuda(feature_shapes)
#     input = [torch.rand(*shape).to(device) for shape in feature_shapes]
    
#     fusion_module = Network().to(device)
#     out = fusion_module(input)
#     print(out.shape)