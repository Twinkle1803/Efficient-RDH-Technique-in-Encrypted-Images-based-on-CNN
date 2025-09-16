import torch.nn.functional as F
import torch.nn as nn
import cv2
from torchvision import transforms
class Three_Three_class1(nn.Module):
    def __init__(self):
        super(Three_Three_class1, self).__init__()
        ### 镜像填充  self.pad = nn.ReflectionPad2d(1)
        #self.pad33_1 = nn.ReflectionPad2d(1)
        #self.pad33_2 = nn.ReflectionPad2d(1)
        #self.pad3_1 = nn.ReflectionPad2d((1, 0))
        #self.pad1_3 = nn.ReflectionPad2d((0, 1))
        ## 卷积核3*3
        self.Conv_second_33 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.Conv_second_33_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        ### 1*3 卷积
        self.Conv_first_13 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), stride=1, padding=(0, 0))
        ### 3*1 卷积
        self.Conv_first_31 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 1), stride=1, padding=(0, 0))
        ### 激活函数
        self.leakyrelu1 = nn.LeakyReLU(inplace=True)
        self.Conv_second_33_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)  ##宽高减2
        self.leakyrelu2 = nn.LeakyReLU(inplace=True)
        self.Conv_second_33_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)  ##宽高减2


    def forward(self, images):
        out1 = F.pad(images, pad=(1,1,1,1), mode ='reflect')
        out2 = self.Conv_second_33(out1)
        out3 = F.pad(images, pad=(1,1,0,0), mode ='reflect')
        out4 = self.Conv_first_13(out3)
        out5 = F.pad(images, pad=(0,0,1,1), mode ='reflect')
        out6 = self.Conv_first_31(out5)
        out7 = self.leakyrelu1(out2 + out4 + out6)
        out8 = self.Conv_second_33_2(out7)
        out9 = self.leakyrelu2(out8)
        out10 =  self.Conv_second_33_3(out9)
        return out10  #32
class Five_class1(nn.Module):
    def __init__(self):
        super(Five_class1, self).__init__()
        ### 镜像填充  self.pad = nn.ReflectionPad2d(1)
        #self.pad55_1 = nn.ReflectionPad2d(2)
        #self.pad55_2 = nn.ReflectionPad2d(2)
        #self.pad5_1 = nn.ReflectionPad2d((2, 0))
        #self.pad1_5 = nn.ReflectionPad2d((0, 2))
        ## 卷积核3*3
        self.Conv_first_55 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.Conv_first_55_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        ### 1*3 卷积
        self.Conv_first_15 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 5), stride=1, padding=(0, 0))
        ### 3*1 卷积
        self.Conv_first_51 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 1), stride=1, padding=(0, 0))
        ### 激活函数
        self.leakyrelu1 = nn.LeakyReLU(inplace=True)
        self.Conv_first_55_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.leakyrelu2 = nn.LeakyReLU(inplace=True)
        self.Conv_first_55_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)

    def forward(self, images):
        out1 = F.pad(images, pad=(2,2,2,2), mode ='reflect')
        out2 = self.Conv_first_55(out1)
        out3 = F.pad(images, pad=(2,2,0,0), mode ='reflect')
        out4 = self.Conv_first_15(out3)
        out5 = F.pad(images, pad=(0,0,2,2), mode ='reflect')
        out6 = self.Conv_first_51(out5)
        out7 = self.leakyrelu1(out2 + out4 + out6)
        out8 = self.Conv_first_55_2(out7)
        out9 = self.leakyrelu2(out8)
        out10 = self.Conv_first_55_3(out9)

        return out10

class Seven_class1(nn.Module):
    def __init__(self):
        super(Seven_class1, self).__init__()
        ### 镜像填充  self.pad = nn.ReflectionPad2d(1)
        #self.pad77_1 = nn.ReflectionPad2d(3)
        #self.pad77_2 = nn.ReflectionPad2d(3)
        #self.pad7_1 = nn.ReflectionPad2d((3, 0))
        #self.pad1_7 = nn.ReflectionPad2d((0, 3))
        ### 卷积核7*7
        self.Conv_first_77 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0)
        self.Conv_first_77_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3)
        ### 1* 7卷积
        self.Conv_first_17 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 7), stride=1, padding=(0, 0))
        ### 7*1 卷积
        self.Conv_first_71 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, 1), stride=1, padding=(0, 0))
        ### 激活函数
        self.leakyrelu1 = nn.LeakyReLU(inplace=True)
        self.Conv_first_77_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.leakyrelu2 = nn.LeakyReLU(inplace=True)
        self.Conv_first_77_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3)

    def forward(self, images):
        out1 = F.pad(images, pad=(3,3,3,3), mode ='reflect')
        out2 = self.Conv_first_77(out1)
        out3 = F.pad(images, pad=(3,3,0,0), mode ='reflect')
        out4 = self.Conv_first_17(out3)
        out5 = F.pad(images, pad=(0,0,3,3), mode ='reflect')
        out6 = self.Conv_first_71(out5)
        out7 = self.leakyrelu1(out2 + out4 + out6)
        out8 = self.Conv_first_77_2(out7)
        out9 = self.leakyrelu2(out8)
        out10 = self.Conv_first_77_3(out9)

        return out10




class HJY_AC_CNNP(nn.Module):
    def __init__(self):
        super(HJY_AC_CNNP, self).__init__()

        self.feature3 = nn.Sequential(
            Three_Three_class1()

        )
        self.feature5 = nn.Sequential(
            Five_class1()

        )
        self.feature7 = nn.Sequential(
            Seven_class1()

        )
        self.predicted1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True)

        )
        self.predicted2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(inplace=True)
        )
        self.predicted3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(inplace=True)
        )
        self.Conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, images):
        out1 = self.feature3(images)
        out2 = self.feature5(images)
        out3 = self.feature7(images)
        out4 = self.predicted1(out1+out2+out3)
        out5 = self.predicted2(out4 + out1 + out2 + out3)
        out6=self.predicted3(out5+out1+out2+out3)
        out = self.Conv(out6)
        ##print(out.shape)
        return out

# """
# img = cv2.imread("./test/1.png",cv2.IMREAD_GRAYSCALE)
# trans_Totensor = transforms.ToTensor()
# img_totensor = trans_Totensor(img)
# hjy =  HJY_AC_CNNP()
# gg = hjy(img_totensor)
# ""”
