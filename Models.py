import torch.nn as nn
import numpy as np
import torch
import utils


# Convolution operation 填充卷积激活
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_relu=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_relu = use_relu
        self.PReLU=nn.PReLU()
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.use_relu is True:
            out = self.PReLU(out)
        return out

class ConvLayer_dis(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_relu=True):
        super(ConvLayer_dis, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_relu = use_relu
        self.LeakyReLU=nn.LeakyReLU(0.2)
    def forward(self, x):

        out = self.conv2d(x)
        if self.use_relu is True:
            out = self.LeakyReLU(out)
        return out

class Inter_Att(torch.nn.Module):
    def __init__(self,channels):
        super(Inter_Att, self).__init__()

        self.sigmod = nn.Sigmoid()
        self.ca_avg = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(channels,channels//2,kernel_size=1),
                nn.PReLU(),
                nn.Conv2d(channels//2,channels,kernel_size=1),
                    )
        self.ca_max = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
        )
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.conv1 = nn.Conv2d(2 * channels, channels, 1, 1)
    def forward(self,ir,vis):
        w_ir_avg = self.ca_avg(ir)
        w_ir_max = self.ca_max(ir)
        w_ir = torch.cat([w_ir_avg,w_ir_max], dim=1)
        w_ir =self.conv1(w_ir)
        w_ir_f =self.sigmod(w_ir)

        w_vis_avg = self.ca_avg(vis)
        w_vis_max = self.ca_max(vis)
        w_vis = torch.cat([w_vis_avg,w_vis_max], dim=1)
        w_vis = self.conv1(w_vis)
        w_vis_f = self.sigmod(w_vis)


        EPSILON = 1e-10
        mask_ir = torch.exp(w_ir_f) / (torch.exp(w_ir_f) + torch.exp(w_vis_f) + EPSILON)
        mask_vis = torch.exp(w_vis_f) / (torch.exp(w_ir_f) + torch.exp(w_vis_f) + EPSILON)
        out_ir = mask_ir*ir
        out_vis = mask_vis*vis

        avgout_ir = torch.mean(out_ir, dim=1, keepdim=True)
        maxout_ir, _ = torch.max(out_ir, dim=1, keepdim=True)
        x_ir = torch.cat([avgout_ir, maxout_ir], dim=1)
        x1_ir = self.conv(x_ir)
        x2_ir = self.sigmod(x1_ir)

        avgout_vis = torch.mean(out_vis, dim=1, keepdim=True)
        maxout_vis, _ = torch.max(out_vis, dim=1, keepdim=True)
        x_vis = torch.cat([avgout_vis, maxout_vis], dim=1)
        x1_vis = self.conv(x_vis)
        x2_vis = self.sigmod(x1_vis)

        mask_ir_sa = torch.exp(x2_ir) / (torch.exp(x2_ir) + torch.exp(x2_vis) + EPSILON)
        mask_vis_sa = torch.exp(x2_vis) / (torch.exp(x2_ir) + torch.exp(x2_vis) + EPSILON)

        output_ir = mask_ir_sa*out_ir
        output_vis = mask_vis_sa*out_vis

        output = torch.cat([output_ir,output_vis], dim=1)

        return output

class Comp_Att(torch.nn.Module):
    def __init__(self,channels):
        super(Comp_Att, self).__init__()
        self.ca_avg = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(channels,channels//2,kernel_size=1),
                nn.PReLU(),
                nn.Conv2d(channels//2,channels,kernel_size=1),
                    )

        self.ca_max = nn.Sequential(
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Conv2d(channels, channels // 2, kernel_size=1),
                nn.PReLU(),
                nn.Conv2d(channels // 2, channels, kernel_size=1),
    )
        self.sigmod = nn.Sigmoid()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.conv1 = nn.Conv2d(2*channels,channels,1,1)
    def forward(self,x):
        w_avg = self.ca_avg(x)
        w_max = self.ca_max(x)
        w = torch.cat([w_avg,w_max], dim=1)
        w = self.conv1(w)
        w_f =self.sigmod(w)
        output_ca = x*w_f

        avgout = torch.mean(output_ca, dim=1, keepdim=True)
        maxout, _ = torch.max(output_ca, dim=1, keepdim=True)
        x_sa = torch.cat([avgout, maxout], dim=1)
        x1 = self.conv(x_sa)
        x2 = self.sigmod(x1)
        output = output_ca*x2

        return output
class UpsampleReshape(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, shape, x):
        x = self.up(x)

        shape = shape.size()
        shape_x = x.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape[3] != shape_x[3]:
            lef_right = shape[3] - shape_x[3]
            if lef_right % 2 is 0.0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape[2] != shape_x[2]:
            top_bot = shape[2] - shape_x[2]
            if top_bot % 2 is 0.0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x = reflection_pad(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder_pad = nn.ReflectionPad2d([1, 1, 1, 1])
        kernel_size_2 = 3

        self.encoder_pad = nn.ReflectionPad2d([1, 1, 1, 1])
        # encoder
        self.conv1 =  ConvLayer(2, 16, kernel_size=3, stride=1, use_relu=True)
        self.conv2 = ConvLayer(1, 16, kernel_size=3, stride=1, use_relu=True)
        self.conv3 = ConvLayer(16, 16, kernel_size=3, stride=1, use_relu=True)
        self.conv4 = ConvLayer(16, 16, kernel_size=3, stride=1, use_relu=True)
        self.conv5 = ConvLayer(16, 32, kernel_size=3, stride=2, use_relu=True)
        self.conv6 = ConvLayer(32, 64, kernel_size=3, stride=2, use_relu=True)

        self.inter_conv_1 = ConvLayer(64, 32, kernel_size=3, stride=2, use_relu=True)
        self.inter_conv_2 = ConvLayer(128, 64, kernel_size=3, stride=2, use_relu=True)

        self.Inter_Att1 = Inter_Att(32)
        self.Comp_Att1 =Comp_Att(16)

        self.Inter_Att2 = Inter_Att(64)
        self.Comp_Att2 = Comp_Att(32)

        self.Inter_Att3 =Inter_Att(128)
        self.Comp_Att3 = Comp_Att(64)

        # decoder
        self.UP1 = UpsampleReshape()
        self.UP2 = UpsampleReshape()

        self.conv7 = ConvLayer(384, 384, 3, stride=1, use_relu=True)
        self.conv8 = ConvLayer(384, 128, kernel_size_2, stride=1, use_relu=True)
        self.conv9 = ConvLayer(192, 192, 3, stride=1, use_relu=True)
        self.conv10 = ConvLayer(192, 64, kernel_size_2, stride=1, use_relu=True)
        self.conv11 = ConvLayer(96, 32, kernel_size_2, stride=1, use_relu=True)
        self.conv12 = ConvLayer(32, 1, kernel_size_2, stride=1, use_relu=False)

        self.tanh = nn.Tanh()

    def forward(self,input_ir,input_vis):
    #ENCODER
        concat_path = torch.cat([input_ir, input_vis], dim=1)
        concat_1=self.conv1(concat_path) #2-16

        ir_1 = self.conv2(input_ir) #1-16
        vis_1 =self.conv2(input_vis) #1-16

        ir_2 = self.conv3(ir_1)#16-16
        vis_2= self.conv3(vis_1)#16-16
        concat_2 = self.conv4(concat_1)#16-16

        ir_3 =self.conv5(ir_2) #16-32
        vis_3 = self.conv5(vis_2) #16-32

        ir_4 = self.conv6(ir_3) #32-64
        vis_4 = self.conv6(vis_3) #32-64

        inter_ir_1 =  torch.cat([concat_2, ir_2], dim=1) #32
        inter_vis_1 = torch.cat([concat_2, vis_2], dim=1) #32
        inter_att_1 = self.Inter_Att1(inter_ir_1,inter_vis_1) #32-64

        inter_out_1 = self.inter_conv_1(inter_att_1) #64-32


        inter_ir_2 =  torch.cat([inter_out_1,ir_3], dim=1) #64
        inter_vis_2 = torch.cat([inter_out_1, vis_3], dim=1) #64
        inter_att_2 = self.Inter_Att2(inter_ir_2, inter_vis_2)  # 64-128

        inter_out_2 = self.inter_conv_2(inter_att_2)  # 128-64

        inter_ir_3 = torch.cat([inter_out_2, ir_4], dim=1)  # 128
        inter_vis_3 = torch.cat([inter_out_2, vis_4], dim=1)  # 128
        inter_att_3= self.Inter_Att3(inter_ir_3, inter_vis_3)  # 128-256

        inter_out_3 = inter_att_3

    #DECODER
        ir_att_4 =self.Comp_Att3(ir_4)
        vis_att_4 = self.Comp_Att3(vis_4)

        encoder_out = torch.cat([inter_out_3, ir_att_4,vis_att_4], dim=1) #256+64+64=384
        encoder_out = self.UP1(inter_ir_2,encoder_out)
        encoder_out = self.conv7(encoder_out)
        de_1 = self.conv8(encoder_out) #384 - 128

        ir_att_3 = self.Comp_Att2(ir_3)
        vis_att_3 = self.Comp_Att2(vis_3)

        de_1_out = torch.cat([de_1,ir_att_3, vis_att_3], dim=1) #128+32+32=192

        de_1_out= self.UP2(inter_ir_1, de_1_out)
        de_1_out = self.conv9(de_1_out)
        de_2 = self.conv10(de_1_out) #192 - 64

        ir_att_2 = self.Comp_Att1(ir_2)
        vis_att_2 = self.Comp_Att1(vis_2)


        de_2_out = torch.cat([de_2, ir_att_2 , vis_att_2], dim=1) #64+16+16=96
        de_3 = self.conv11(de_2_out) #96-32
        output =self.conv12(de_3) #32-1

        output =self.tanh(output)


        return output




class  D_IR(nn.Module):
    def __init__(self):
        super( D_IR, self).__init__()
        fliter = [1,16,32,64,128]
        kernel_size = 3
        stride = 2
        self.l1 = ConvLayer_dis(fliter[0], fliter[1], kernel_size, stride, use_relu=True)
        self.l2 = ConvLayer_dis(fliter[1], fliter[2], kernel_size, stride, use_relu=True)
        self.l3 = ConvLayer_dis(fliter[2], fliter[3], kernel_size, stride, use_relu=True)
        self.l4 = ConvLayer_dis(fliter[3], fliter[4], kernel_size, stride, use_relu=True)
        self.tanh = nn.Tanh()


    def forward(self, x):

        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = out.view(out.size()[0], -1)
        linear = nn.Linear(out.size()[1], 1).cuda()
        out = self.tanh(linear(out))

        return out.squeeze()

class  D_VI(nn.Module):
    def __init__(self):
        super( D_VI, self).__init__()
        fliter = [1,16,32,64,128]
        kernel_size = 3
        stride = 2
        self.l1 = ConvLayer_dis(fliter[0], fliter[1], kernel_size, stride, use_relu=True)
        self.l2 = ConvLayer_dis(fliter[1], fliter[2], kernel_size, stride, use_relu=True)
        self.l3 = ConvLayer_dis(fliter[2], fliter[3], kernel_size, stride, use_relu=True)
        self.l4 = ConvLayer_dis(fliter[3], fliter[4], kernel_size, stride, use_relu=True)

        self.tanh = nn.Tanh()
    def forward(self, x):

        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = out.view(out.size()[0], -1)
        linear = nn.Linear(out.size()[1], 1).cuda()
        out = self.tanh(linear(out))

        return out.squeeze()



