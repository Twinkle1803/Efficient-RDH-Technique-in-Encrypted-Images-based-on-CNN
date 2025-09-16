import torchvision
from torch.utils.data import DataLoader
from NEW_CNN3 import HJY_AC_CNNP
from Dataset_new import MyData
import torch
from torch import nn
import numpy as np
import cv2
############## 哈夫曼编码实现 #################
## 节点类
class Node(object):
    def __init__(self,name = None, value = None):
        self._name = name
        self._value = value
        self._left = None
        self._right = None
## 哈夫曼树类
class HuffmanTree(object):
    ## 根据Huffman Tree的思想，以叶子节点为基础，反向建立Huffman树
    def __init__(self, char_weights):
        ## 根据字符的频率生成叶子节点
        self.a = [Node(part[0], part[1]) for part in char_weights]
        while len(self.a) != 1:
            self.a.sort(key=lambda node:node._value,reverse=True)
            c = Node(value=(self.a[-1]._value + self.a[-2]._value))
            c._left = self.a.pop(-1)
            c._right = self.a.pop(-1)
            self.a.append(c)
        self.root = self.a[0]
        self.b = list(range(10))
        self.huffman_code = dict()
    ## 递归思想生成编码
    def pre(self,tree,length):
        node = tree
        if (not node):
            return
        elif node._name:
            x = ""
            for i in range(length):
                x += str(self.b[i])
            #print(x)
            self.huffman_code[node._name] = x
            #print(self.huffman_code)
            return
        self.b[length] = 0
        self.pre(node._left,length+1)
        self.b[length] = 1
        self.pre(node._right,length+1)
    def get_code(self):
        hjy = self.pre(self.root,0)
        return self.huffman_code


#### 定义capacity !!!!!!!!
capacity_best = 0
capacity_average = 0
capacity_worse = 0

## 定义测试设备
# 1. 训练数据集
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
###  训练数据集
test_dir = './test'
###  自定义dataloader
test_dataset = MyData(test_dir,transforms_=dataset_transform)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
## 4. 图像预处理
dot = np.zeros((512,512), dtype=np.float32)
cross = np.zeros((512,512),dtype=np.float32)
hjy1 = np.zeros((512,512),dtype=np.uint8)
for i in range(512):
    for j in range(512):
        if (i + j)%2 == 0:
            dot[i][j] = 1
            hjy1[i][j] = 1
        if (i + j)%2 == 1:
            cross[i][j] = 1
            hjy1[i][j] = 1

cross = np.expand_dims(cross, axis=2)
dot = np.expand_dims(dot, axis=2)
dot = dataset_transform(dot)
cross = dataset_transform(cross)
### secret_img 是最终保存的含谜图像
secret_img = np.zeros((512,512),dtype=np.uint8)
### 5. 加载模型
file_name = './Train_state130.pth'

model = HJY_AC_CNNP() # Initialize the model
model.load_state_dict(torch.load(file_name, map_location='cpu'))

ii = 0
a = 0
b = 0
#torch.save(model, "test.pth")
with torch.no_grad():
    for data in test_dataloader:
        print(ii)
        img, img_self, name = data
        print(name)
        #print(img)
        #print(img_self.shape)
        #print(img_self)
        img_self = torch.squeeze(img_self)
        #print(type(img_self))
        img_self = img_self.cpu().numpy()
        #print(img_self.shape)
        #print("this is the original image")
        #print(img_self)
        img_dot = torch.mul(img, dot)
        img_cross = torch.mul(img, cross)
        #print("this is  !!!!!!!!!!!!!")
        #print(img_cross.shape)
        #print(type(img_cross))
        ## 模型预测
        predicted_image = model(img_dot)
        #print(predicted_image.shape)
        #print(predicted_image)
        ## 维度变换
        predicted_image = torch.squeeze(predicted_image)
        #print(predicted_image.shape)
        #print(predicted_image)
        ## 数据类型转换
        predicted_image = predicted_image.cpu().numpy()
        predicted_image = np.around(predicted_image)
        predicted_image[predicted_image<0] = 0
        #print(predicted_image)
        ## 转换成uint8
        predicted_image = predicted_image.astype(np.uint8)
#########################################################################################
        ## 初始化一些数据，用于数据嵌入
        label_set = {"{}".format(i): i - i for i in range(9)}
        #print(label_set)
        label_matrix = np.zeros((512, 512), dtype=np.uint8)
        #print(label_matrix)
        ## 简易哈夫曼编码
        ## 不需要 code = ["11111", "11110", "1110", "1101", "1100", "101", "100", "01", "00"]
        huff_code = dict()
        ### img_self 为原始图像 ， predicted_image 为预测的cross_iamge
        ### 基于预测的cross——image，在cross_imge 中，计算MSB， 哈夫曼编码，进行数据的嵌入
        ## 首先，统计嵌入位数，进行哈夫曼编码
        ## 默认处理的图像大小都是512*512
        for i in range(512):
            for j in range(512):
                if (i+j) % 2 == 1:
                    original_pixel = img_self[i][j]
                    predicted_pixel = predicted_image[i][j]
                    #print("this is orginal {} and this is predicted {} ".format(original_pixel, predicted_pixel))
                    #print()
                    ## 转二进制
                    original_bin = '{0:08b}'.format(original_pixel)
                    predicted_bin ='{0:08b}'.format(predicted_pixel)
                    #print("this is original {} and this is the prredicted {}".format(original_bin, predicted_bin))
                    #print(type(original_bin))
                    #print(type(predicted_bin))
                    label_index = 0
                    while label_index < 8:
                        if original_bin[label_index] == predicted_bin[label_index]:
                            label_index += 1
                        else:
                            break
                    #print(label_index)
                    label_set[str(label_index)] += 1
                    label_matrix[i][j] = label_index
        #print(label_set)
        sorted_label_set = sorted(label_set.items(), key=lambda x: x[1])
        #print(sorted_label_set)
        index = 0
        """
        for label in sorted_label_set:
            huff_code[label[0]] = code[index]
            index += 1
        """
        tree2 = HuffmanTree(sorted_label_set)
        huff_code = tree2.get_code()
        #print(huff_code)
        #print(label_matrix)
        ######## 计算容量 ###############
        cross_total_capacity = 0
        cross_aux_information = 0
        for key in label_set:
            if key == '8':
                cross_total_capacity += 8 * label_set[key]
                cross_aux_information += len(huff_code[key]) * label_set[key]
            else:
                cross_total_capacity += (int(key) + 1) * label_set[key]
                cross_aux_information += len(huff_code[key]) * label_set[key]
        capacity_cross = cross_total_capacity - cross_aux_information
        print("cross有效嵌入容量为 {} bit".format(capacity_cross))
        print("cross总的嵌入容量为 {} bit".format(cross_total_capacity))
        print("cross辅助信息的大小为 {} bit".format(cross_aux_information))
        """
        print("总的嵌入容量为 {} bit".format(total_capacity))
        print("辅助信息的大小为 {} bit".format(aux_information))
        print("有效嵌入容量为 {} bit".format(total_capacity - aux_information))
        """

        ###################  dotdotdotdotdotdotdotdot   ##############
        #print("start dot processing !!!!!!!!!!!!!!")
        # print(img_dot)
        ######## top #####################

        img_dot = img_self * hjy1

        ## 初始化一些数据
        #print("this is the top predictor")
        label_set = {"{}".format(i): i - i for i in range(9)}
        label_matrix = np.zeros((512, 512), dtype=np.uint8)
        ###huff_code = dict()

        ## 简单的top edge median predict
        ## 第一列， 第一行， 最后一列不使用
        for i in range(1, 512):
            for j in range(1, 511):
                if (i + j) % 2 == 0:
                    ## 数据会溢出
                    # a = np.uint16(img_dot[i-1][j-1])
                    temp1 = (np.uint16(img_dot[i - 1][j - 1]) + np.uint16(img_dot[i - 1][j + 1])) // 2
                    temp2 = img_dot[i][j]
                    temp1_bin = '{0:08b}'.format(temp1)
                    temp2_bin = '{0:08b}'.format(temp2)
                    label_index = 0
                    while label_index < 8:
                        if temp1_bin[label_index] == temp2_bin[label_index]:
                            label_index += 1
                        else:
                            break
                    # print(label_index)
                    label_set[str(label_index)] += 1
                    label_matrix[i][j] = label_index
        #print(label_set)
        # print(label_set)
        sorted_label_set = sorted(label_set.items(), key=lambda x: x[1])
        #print(sorted_label_set)
        index = 0
        """
        for label in sorted_label_set:
            huff_code[label[0]] = code[index]
            index += 1
        """
        tree2 = HuffmanTree(sorted_label_set)
        huff_code = tree2.get_code()
        #print(huff_code)
        # print(label_matrix)
        ######## 计算容量 ###############
        total_capacity = 0
        aux_information = 0
        for key in label_set:
            if key == '8':
                total_capacity += 8 * label_set[key]
                aux_information += len(huff_code[key]) * label_set[key]
            else:
                total_capacity += (int(key) + 1) * label_set[key]
                aux_information += len(huff_code[key]) * label_set[key]
        temp_capacity1 = total_capacity - aux_information
        print("1有效嵌入容量为 {} bit".format(temp_capacity1))
        print("1总的嵌入容量为 {} bit".format(total_capacity))
        print("1辅助信息的大小为 {} bit".format(aux_information))



        ######## bottom #####################
        #print("this is the botttom predictor")
        ## 初始化一些数据
        label_set_bottom = {"{}".format(i): i - i for i in range(9)}
        label_matrix_bottom = np.zeros((512, 512), dtype=np.uint8)
        ###huff_code = dict()

        ## 简单的top edge median predict
        ## left, right , bottom 不用
        for i in range(510, -1, -1):
            for j in range(510, 0, -1):
                if (i + j) % 2 == 0:
                    ## 数据会溢出
                    # a = np.uint16(img_dot[i-1][j-1])
                    temp1 = (np.uint16(img_dot[i + 1][j + 1]) + np.uint16(img_dot[i + 1][j - 1])) // 2
                    temp2 = img_dot[i][j]
                    temp1_bin = '{0:08b}'.format(temp1)
                    temp2_bin = '{0:08b}'.format(temp2)
                    label_index = 0
                    while label_index < 8:
                        if temp1_bin[label_index] == temp2_bin[label_index]:
                            label_index += 1
                        else:
                            break
                    # print(label_index)
                    label_set_bottom[str(label_index)] += 1
                    label_matrix_bottom[i][j] = label_index
        #print(label_set_bottom)
        # print(label_set)
        sorted_label_set_bottom = sorted(label_set_bottom.items(), key=lambda x: x[1])
        #print(sorted_label_set_bottom)
        tree3 = HuffmanTree(sorted_label_set_bottom)
        huff_code_bottom = tree3.get_code()
        #print(huff_code_bottom)
        # print(label_matrix)
        ######## 计算容量 ###############
        total_capacity1 = 0
        aux_information1 = 0
        for key in label_set_bottom:
            if key == '8':
                total_capacity1 += 8 * label_set_bottom[key]
                aux_information1 += len(huff_code_bottom[key]) * label_set_bottom[key]
            else:
                total_capacity1 += (int(key) + 1) * label_set_bottom[key]
                aux_information1 += len(huff_code_bottom[key]) * label_set_bottom[key]

        temp_capacity2 = total_capacity1 - aux_information1
        print("2有效嵌入容量为 {} bit".format(temp_capacity2))
        print("2总的嵌入容量为 {} bit".format(total_capacity1))
        print("2辅助信息的大小为 {} bit".format(aux_information1))
        ######## bottom #####################
        #print("this is the left predictor")
        ## 初始化一些数据
        label_setr = {"{}".format(i): i - i for i in range(9)}
        label_matrixr = np.zeros((512, 512), dtype=np.uint8)
        ###huff_code = dict()

        ## 简单的top edge median predict
        ##  left, top , bottom不用
        for i in range(1, 511):
            for j in range(1, 512):
                if (i + j) % 2 == 0:
                    ## 数据会溢出
                    # a = np.uint16(img_dot[i-1][j-1])
                    temp1 = (np.uint16(img_dot[i - 1][j - 1]) + np.uint16(img_dot[i + 1][j - 1])) // 2
                    temp2 = img_dot[i][j]
                    temp1_bin = '{0:08b}'.format(temp1)
                    temp2_bin = '{0:08b}'.format(temp2)
                    label_index = 0
                    while label_index < 8:
                        if temp1_bin[label_index] == temp2_bin[label_index]:
                            label_index += 1
                        else:
                            break
                    # print(label_index)
                    label_setr[str(label_index)] += 1
                    label_matrixr[i][j] = label_index

        #print(label_setr)
        # print(label_set)
        sorted_label_setr = sorted(label_setr.items(), key=lambda x: x[1])
        #print(sorted_label_setr)
        tree3 = HuffmanTree(sorted_label_setr)
        huff_coder = tree3.get_code()
        #print(huff_coder)
        # print(label_matrix)
        ######## 计算容量 ###############
        total_capacityr = 0
        aux_informationr = 0
        for key in label_setr:
            if key == '8':
                total_capacityr += 8 * label_setr[key]
                aux_informationr += len(huff_coder[key]) * label_setr[key]
            else:
                total_capacityr += (int(key) + 1) * label_setr[key]
                aux_informationr += len(huff_coder[key]) * label_setr[key]


        temp_capacity3 = total_capacityr - aux_informationr
        print("3有效嵌入容量为 {} bit".format(temp_capacity3))
        print("3总的嵌入容量为 {} bit".format(total_capacityr))
        print("3辅助信息的大小为 {} bit".format(aux_informationr))
        ######## bottom #####################
        #print("this is the right predictor")
        ## 初始化一些数据
        label_setl = {"{}".format(i): i - i for i in range(9)}
        label_matrixl = np.zeros((512, 512), dtype=np.uint8)
        ###huff_code = dict()
        ## 简单的top edge median predict
        ##  right, top , bottom不用
        for i in range(1, 511):
            for j in range(510, -1, -1):
                if (i + j) % 2 == 0:
                    ## 数据会溢出
                    # a = np.uint16(img_dot[i-1][j-1])
                    temp1 = (np.uint16(img_dot[i - 1][j - 1]) + np.uint16(img_dot[i + 1][j - 1])) // 2
                    temp2 = img_dot[i][j]
                    temp1_bin = '{0:08b}'.format(temp1)
                    temp2_bin = '{0:08b}'.format(temp2)
                    label_index = 0
                    while label_index < 8:
                        if temp1_bin[label_index] == temp2_bin[label_index]:
                            label_index += 1
                        else:
                            break
                    # print(label_index)
                    label_setl[str(label_index)] += 1
                    label_matrixl[i][j] = label_index
        #print(label_setl)
        # print(label_set)
        sorted_label_setl = sorted(label_setl.items(), key=lambda x: x[1])
        #print(sorted_label_setl)
        tree3 = HuffmanTree(sorted_label_setl)
        huff_codel = tree3.get_code()
        #print(huff_codel)
        # print(label_matrix)
        ######## 计算容量 ###############
        total_capacityl = 0
        aux_informationl = 0
        for key in label_setl:
            if key == '8':
                total_capacityl += 8 * label_setl[key]
                aux_informationl += len(huff_codel[key]) * label_setl[key]
            else:
                total_capacityl += (int(key) + 1) * label_setl[key]
                aux_informationl += len(huff_codel[key]) * label_setl[key]

        temp_capacity4 = total_capacityl - aux_informationl
        print("4有效嵌入容量为 {} bit".format(temp_capacity4))
        print("4总的嵌入容量为 {} bit".format(total_capacityl))
        print("4辅助信息的大小为 {} bit".format(aux_informationl))

        dot_capacity = max(temp_capacity1, temp_capacity2, temp_capacity3, temp_capacity4)
        print("dot有效嵌入容量为 {} bit".format(dot_capacity))
        capacity = capacity_cross + dot_capacity
        print("有效嵌入容量为 {} bit".format(capacity))
        print("有效嵌入容量为 {} bpp".format(capacity/512/512))

        if ii == 0:
            capacity_best = capacity / 512 / 512
            capacity_worse = capacity / 512 / 512


        if (capacity/512/512) >= capacity_best:
            capacity_best = capacity/512/512
        if (capacity/512/512) <= capacity_worse:
            capacity_worse = capacity/512/512
        ii += 1
        capacity_average += capacity/512/512
print("the best is {}".format(capacity_best))
print("the worse is {}".format(capacity_worse))
print("the average is {}".format(capacity_average/10000))















































