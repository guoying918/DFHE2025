from Models.ODconv2d import *
import torch

def to_cuda(tensor_dict):
    return [tensor.cuda() for tensor in tensor_dict]

def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)

def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    reduction=reduction, kernel_num=kernel_num)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
        super(BasicBlock, self).__init__()
        self.conv1 = odconv3x3(inplanes, planes, stride, reduction=reduction, kernel_num=kernel_num)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = odconv3x3(planes, planes, reduction=reduction, kernel_num=kernel_num)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        out_1 = self.relu(self.bn1(self.conv1(x)))
        out_2 = self.bn2(self.conv2(out_1))
        out = identity + out_2

        return out

class DyHeadBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels
    ):
        super(DyHeadBlock, self).__init__()
        
        self.spatial_conv_high = BasicBlock(in_channels, out_channels)
        self.spatial_conv_mid = BasicBlock(in_channels, out_channels)
        self.spatial_conv_low = BasicBlock(in_channels, out_channels)
        self.conv_high = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_low = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x, outs: list of feature maps, [(N, C, H, W), ..., (N, C, H, W)]
        """
        outs = []
        for level in range(len(x)):
            sum_feat = self.spatial_conv_mid(x[level])
            summed_levels = 1
            if level > 0:
                sum_feat += self.conv_low(F.interpolate(
                        self.spatial_conv_low(x[level - 1]),
                        size=x[level].shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    ))
                summed_levels += 1
            if level < len(x) - 1:
                sum_feat += self.conv_high(F.interpolate(
                        self.spatial_conv_high(x[level + 1]),
                        size=x[level].shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    ))
                summed_levels += 1
            outs.append(sum_feat / summed_levels)

        return outs

class CPDFP(nn.Module):
    def __init__(self, in_channels):
        super(CPDFP, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels + 1, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.caLayer = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=2 * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=2 * 2, out_channels=1, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs):
        """
        inputs: list of feature maps, [(N, C, H, W), ..., (N, C, H, W)]
        output: (N, C)
        """
        outs = []
        for out_feat_temp in inputs:
            b, c, h, w = out_feat_temp.size()
            center_pixel = out_feat_temp[:, :, h // 2, w // 2].unsqueeze(2).unsqueeze(3)  # (b, c, 1, 1)
            dot_product = (out_feat_temp * center_pixel).sum(dim=1, keepdim=True) / c  # (b, 1, h, w)
            concatenated = torch.cat([out_feat_temp, dot_product], dim=1)  # (b, c+1, h, w)
            attention_weights = torch.sigmoid(self.conv(concatenated))
            normalized_weights = attention_weights / (attention_weights.sum(dim=(2, 3), keepdim=True) + 1e-8)
            fea_weights = out_feat_temp * normalized_weights
            out_feat = fea_weights.sum(dim=2).sum(dim=2).view(b, c)
            outs.append(out_feat)
            
        pooled_features = torch.stack(outs, dim=1)  # (b, num_features, c)
        b, _, c = pooled_features.size()
        x_weights = self.caLayer(pooled_features)
        y_total = pooled_features * x_weights
        output = y_total.sum(dim=1, keepdim=True).view(b,c)

        return output

class DeformFusion(nn.Module):
    def __init__(
        self,
        in_channels, # 100
        num_blocks = 1
    ):
        super(DeformFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels  # change 128 -> 1024
        dyhead_blocks = []
        for i in range(num_blocks):
            in_channels = self.in_channels if i == 0 else self.out_channels
            dyhead_blocks.append(
                DyHeadBlock(
                    in_channels,
                    self.out_channels,
                )
            )
        self.dyhead_blocks = nn.Sequential(*dyhead_blocks)
        
        self.cpdfp = CPDFP(in_channels)

    def forward(self, inputs):
        outs = self.dyhead_blocks(inputs)
        features = self.cpdfp(outs)
        return features

# if __name__ == '__main__':
#     feature_shapes = [
#         (1800, 100, 5, 5),
#         (1800, 100, 17, 17)
#     ]
#     # 创建特征图列表
#     input = [torch.rand(*shape) for shape in feature_shapes]
    
#     fusion_module = DeformFusion(100)
    
#     outs = fusion_module.dyhead_blocks(input) # 'list' object
#     for i, out in enumerate(outs):
#         print(f"Output {i} shape: {out.shape}")