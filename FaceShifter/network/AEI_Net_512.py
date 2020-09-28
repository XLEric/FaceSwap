import torch
import torch.nn as nn
import torch.nn.functional as F
from .AADLayer import *


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


def conv4x4(in_c, out_c, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False),
        norm(out_c),
        nn.LeakyReLU(0.1, inplace=True)
    )


class deconv4x4(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d):
        super(deconv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = norm(out_c)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, skip), dim=1)


class MLAttrEncoder(nn.Module):
    def __init__(self):
        super(MLAttrEncoder, self).__init__()
        self.conv1 = conv4x4(3, 32) #128
        self.conv2 = conv4x4(32, 64) #64
        self.conv3 = conv4x4(64, 128) #32
        self.conv4 = conv4x4(128, 256)#16 
        self.conv5 = conv4x4(256, 512)#8
        self.conv6 = conv4x4(512, 1024)#4
        self.conv7 = conv4x4(1024, 1024)#2  #in_c, out_c
        self.conv8 = conv4x4(1024, 1024)

        self.deconv1 = deconv4x4(1024, 1024) #2048  #in_c, out_c   and concat
        self.deconv2 = deconv4x4(2048, 1024)
        self.deconv3 = deconv4x4(2048, 512)  # 1024

        self.deconv4 = deconv4x4(1024, 256) #512
        self.deconv5 = deconv4x4(512, 128)  #256
        self.deconv6 = deconv4x4(256, 64)  
        self.deconv7 = deconv4x4(128, 32)

        self.apply(weight_init)

    def forward(self, Xt):
        feat1 = self.conv1(Xt)
        # 32x256*256
        feat2 = self.conv2(feat1)
        # 64x128*128
        feat3 = self.conv3(feat2)
        # 128x64
        feat4 = self.conv4(feat3)
        # 256x32
        feat5 = self.conv5(feat4)  #512
        # 512x16
        feat6 = self.conv6(feat5) #1024
        # 1024x8
        feat7 = self.conv7(feat6)
        # 1024x4
        z_attr1 = self.conv8(feat7)
        # 1024x2

        z_attr2 = self.deconv1(z_attr1, feat7) #2048*4*4
        #print ("z_attr2:",z_attr2.size())
        z_attr3 = self.deconv2(z_attr2, feat6)  #2048 8*8 --- 512
        #print ("z_attr3:",z_attr3.size())
        z_attr4 = self.deconv3(z_attr3, feat5)  #1024 16
        #print ("z_attr4:",z_attr4.size())
        z_attr5 = self.deconv4(z_attr4, feat4) #512 32
        #print ("z_attr5:",z_attr5.size())
        z_attr6 = self.deconv5(z_attr5, feat3)# 256 64
        #print ("z_attr6:",z_attr6.size())
        z_attr7 = self.deconv6(z_attr6, feat2) #128 
        z_attr8 = self.deconv7(z_attr7, feat1) #64  256
        z_attr9 = F.interpolate(z_attr8, scale_factor=2, mode='bilinear', align_corners=True)
        return z_attr1, z_attr2, z_attr3, z_attr4, z_attr5, z_attr6, z_attr7, z_attr8, z_attr9


class AADGenerator(nn.Module):
    def __init__(self, c_id=256):
        super(AADGenerator, self).__init__()
        self.up1 = nn.ConvTranspose2d(c_id, 1024, kernel_size=2, stride=1, padding=0)
        self.AADBlk1 = AAD_ResBlk(1024, 1024, 1024, c_id)
        self.AADBlk2 = AAD_ResBlk(1024, 1024, 2048, c_id) #(self, cin, cout, c_attr, c_id=256):
        self.AADBlk3 = AAD_ResBlk(1024, 1024, 2048, c_id)
        self.AADBlk4 = AAD_ResBlk(1024, 1024, 1024, c_id)
        self.AADBlk5 = AAD_ResBlk(1024, 512, 512, c_id)
        self.AADBlk6 = AAD_ResBlk(512, 256, 256, c_id)
        self.AADBlk7 = AAD_ResBlk(256, 128, 128, c_id)
        self.AADBlk8 = AAD_ResBlk(128, 64, 64, c_id)
        self.AADBlk9 = AAD_ResBlk(64, 3, 64, c_id)

        self.apply(weight_init)

    def forward(self, z_attr, z_id):
        # zid 12* 512
        # z_id.reshape(z_id.shape[0], -1, 1, 1)  12*512 1*1
        # m 12*1024*2*2

        m = self.up1(z_id.reshape(z_id.shape[0], -1, 1, 1))
        m2 = F.interpolate(self.AADBlk1(m, z_attr[0], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m3 = F.interpolate(self.AADBlk2(m2, z_attr[1], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m4 = F.interpolate(self.AADBlk3(m3, z_attr[2], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m5 = F.interpolate(self.AADBlk4(m4, z_attr[3], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m6 = F.interpolate(self.AADBlk5(m5, z_attr[4], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m7 = F.interpolate(self.AADBlk6(m6, z_attr[5], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m8 = F.interpolate(self.AADBlk7(m7, z_attr[6], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m9 = F.interpolate(self.AADBlk8(m8, z_attr[7], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        #print ("forward:",m9.size(), z_attr[8].size())
        y = self.AADBlk9(m9, z_attr[8], z_id)
        return torch.tanh(y)


class AEI_Net(nn.Module):
    def __init__(self, c_id=256):
        super(AEI_Net, self).__init__()
        self.encoder = MLAttrEncoder()
        self.generator = AADGenerator(c_id)

    def forward(self, Xt, z_id):
        attr = self.encoder(Xt)
        Y = self.generator(attr, z_id)
        return Y, attr

    def get_attr(self, X):
        # with torch.no_grad():
        return self.encoder(X)



