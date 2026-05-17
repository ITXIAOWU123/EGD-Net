import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .PVT_V2 import pvt_v2_b2
# from .RMT_Swin import RMT_Swin
# from .UAGLNet import create_encoder
from .resnest import ResNeSt, Bottleneck


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = pvt_v2_b2(img_size=256)
        self.encoder.load_state_dict(torch.load('/home/dengbinquan/XIAOWU/PA-Net/model/pretrain/pvt_v2_b2.pth', map_location='cpu'), strict=False)

    def forward(self, x):
        out = self.encoder(x)
        return out[::-1]



# import torch
# from torch import nn

# # 假设 Bottleneck 和 ResNeSt 已定义
# resnest_model_urls = {
#     'resnest101': 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest101-22405ba7.pth'
# }

# def resnest101(pretrained=False, **kwargs):
#     """
#     构建 ResNeSt-101 模型
#     Args:
#         pretrained (bool): 是否加载 ImageNet 预训练权重
#     """
#     # ResNeSt-101 layer 配置: [3, 4, 23, 3]
#     model = ResNeSt(Bottleneck, [3, 4, 23, 3], **kwargs)
    
#     if pretrained:
#         # 下载权重
#         state_dict = torch.hub.load_state_dict_from_url(
#             resnest_model_urls['resnest101'], progress=True, check_hash=True
#         )
#         model_dict = model.state_dict()
        
#         # 只加载匹配的 key 和大小一致的权重
#         pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
#         model_dict.update(pretrained_dict)
#         model.load_state_dict(model_dict)
#         # print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} keys from pretrained weights.")
    
#     return model

# class Encoder_t(nn.Module):
#     def __init__(self):
#         super(Encoder_t,self).__init__()

#         self.encoder = resnest101(pretrained=True, deep_stem=True, stem_width=32)
#         # self.encoder.load_state_dict(torch.load('./model/pretrain/pvt_v2_b2.pth', map_location='cpu'), strict=False)

#     def forward(self, x):
#         out = self.encoder.forward_feature(x)
#         return out