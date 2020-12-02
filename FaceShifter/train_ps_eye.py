#-*-coding:utf-8-*-
# date:2020-04-25
# Author: X.L.Eric
#function : train for mask
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from network.AEI_PS_Net import *
from network.MultiscaleDiscriminator import *
from utils.Dataset import FaceEmbed, With_Identity
from torch.utils.data import DataLoader
import torch.optim as optim
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch.nn.functional as F
import torch
import time
import torchvision
import cv2
from apex import amp
import visdom
import numpy as np
#
from model import BiSeNet
from PIL import Image
import torchvision.transforms as transforms
import cv2

from utils.timer import Timer

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.mobilenetv2 import MobileNetV2
from utils.common_utils import *
from acc_model import acc_model
from heatmap import *

def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1.-X).mean()
    else:
        return torch.relu(X+1.).mean()


def get_grid_image(X):
    X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X


def make_image(Xs, Xt, Y):
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)
    Y = get_grid_image(Y)
    return torch.cat((Xs, Xt, Y), dim=1).numpy()

def create_faceparse_model(path_model):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()

    print('model : {}'.format(path_model))
    net.load_state_dict(torch.load(path_model))
    net.eval()

    return net
# Compute gaussian kernel
def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map
def create_mask(X_,str_w,vis = False):
    width = 512
    height = 512
    cy, cx = height/2.0, width/2.0

    heatmap1 = CenterGaussianHeatMap(width, height, cx, cy, 150)
    heatmap2 = CenterGaussianHeatMap(width, height, cx, cy, 200)
    face_hair_mask_list = []
    face_mask_list = []
    hair_list = []
    face_edge_list = []
    background_list = []
    only_face_list = []
    sense5_list = []
    eye_brow_list = []
    eye_eye_list = []
    with torch.no_grad():
        Xt_0 = F.interpolate(X_, [512, 512], mode='bilinear', align_corners=True)
        Xt_0 = Xt_0.cpu().numpy()
        # print(Xt_0.shape)
        for k in range(Xt_0.shape[0]):
            d = to_tensor(Xt_0[k].transpose((1, 2, 0)))
            d = torch.unsqueeze(d, 0)
            d = d.cuda()
            out_ = model_faceparse(d)[0]
            parsing = out_.squeeze(0).cpu().numpy().argmax(0)

            parsing_s = (parsing*10).astype(np.uint8)
            if vis:
                print('parsing shape :',parsing.shape)
                cv2.namedWindow('parsing_'+str_w,0)
                cv2.imshow('parsing_'+str_w,parsing_s)
                cv2.namedWindow('img_'+str_w,0)
                cv2.imshow('img_'+str_w,Xt_0[k].transpose((1, 2, 0))* 0.5 + 0.5)

            #------------------------
            face_hair_mask = np.zeros((parsing.shape[0], parsing.shape[1],3))
            face_mask = np.zeros((parsing.shape[0], parsing.shape[1],3))
            hair = np.zeros((parsing.shape[0], parsing.shape[1],3))
            only_face = np.zeros((parsing.shape[0], parsing.shape[1],3))
            sense5 = np.zeros((parsing.shape[0], parsing.shape[1],3))
            eye_brow = np.zeros((parsing.shape[0], parsing.shape[1],3))
            eye_eye = np.zeros((parsing.shape[0], parsing.shape[1],3))
            num_of_class = np.max(parsing)

            for pi in range(1, num_of_class + 1):
                index = np.where(parsing == pi)# 获得对应分类的的像素坐标
                # vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]# 给对应的类别的掩码赋值

                if pi in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
                    face_hair_mask[index[0], index[1]] = 1.
                    face_mask[index[0], index[1]] = 1.
                elif pi in [17,18,6]:
                    hair[index[0], index[1]] = 1.
                #
                if pi in [1,6,7,8,9]:
                    only_face[index[0], index[1]] = 1.
#                 elif pi in [2,3]:#眉毛
#                     only_face[index[0], index[1]] = 0.05
#                 elif pi in [4,5]:#眼睛
#                     only_face[index[0], index[1]] = 0.05

                if pi in [2,3,4,5]:# 眉毛、眼睛
                    eye_brow[index[0], index[1]] = 1.

                if pi in [2,3,4,5,6,10,11,12,13,17]:
                    sense5[index[0], index[1]] = 1.

#                 if pi in [4,5]:# 眼睛 4 和 5
                if pi in [1,2,3,4,5,6,10,11,12,13]:# 眼睛 4 和 5
                    eye_eye[index[0], index[1]] = 1.

            face_hair_mask = hair + face_hair_mask
            #----------- 背景掩码
            background = np.array(np.logical_not(face_hair_mask)).astype(np.float32)
            background = cv2.resize(background, (256,256))
            background = cv2.erode(background, np.ones((11, 11), np.uint8)) # 执行腐蚀操作
            background = cv2.blur(background,(3,3))
            #------------

#             kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#             face_hair_mask = cv2.dilate(face_hair_mask, kernel)
#             face_hair_mask = face_hair_mask * np.expand_dims(heatmap1,axis =2)


#             kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#             face_mask = cv2.dilate(face_mask, kernel)

#             face_mask = face_mask * np.expand_dims(heatmap2,axis =2)

            face_hair_mask = np.minimum(1.,face_hair_mask[:,:,:])
            face_mask = np.minimum(1.,face_mask[:,:,:])

#             hair = cv2.dilate(hair, kernel)

            hair = np.minimum(1.,hair[:,:,:])

            kernel = np.ones((5, 5), np.uint8)

            face_hair_mask = cv2.resize(face_hair_mask, (256,256))

            only_face = cv2.resize(only_face, (256,256))
            only_face = cv2.erode(only_face, np.ones((5, 5), np.uint8)) # 执行腐蚀操作
#             face_hair_mask = cv2.erode(face_hair_mask, kernel) # 执行腐蚀操作
#             for k in range(2):
#                 face_hair_mask = cv2.blur(face_hair_mask,(7,7))


            face_mask = cv2.resize(face_mask, (256,256))
            # MORPH_ELLIPSE   MORPH_RECT  MORPH_CROSS
            sense5 = cv2.dilate(sense5, cv2.getStructuringElement(cv2.MORPH_CROSS, (13, 13)))
            sense5 = cv2.resize(sense5, (256,256))

            eye_brow = cv2.dilate(eye_brow, cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7)))
            eye_brow = cv2.resize(eye_brow, (256,256))

#             for uu in range(1):
#                 eye_eye = cv2.dilate(eye_eye, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            for uu in range(1):
                eye_eye = cv2.dilate(eye_eye, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))

            eye_eye = cv2.resize(eye_eye, (256,256))



            # ------------ face edge

            kernel_edge = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            face_mask = cv2.erode(face_mask, kernel) # 执行腐蚀操作
            for k in range(3):
                face_mask_dilate = cv2.dilate(face_mask, kernel_edge)
            kernel_edge = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))

            face_edge_dilate = np.array(np.logical_xor(face_mask_dilate,cv2.erode(face_mask, kernel_edge))).astype(np.float32)

            kernel_edge = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            face_edge_dilate = cv2.dilate(face_edge_dilate, kernel_edge)

            face_edge_dilate = cv2.blur(face_edge_dilate,(3,3))
            # ------------
#             face_mask = cv2.erode(face_mask, kernel) # 执行腐蚀操作
#             for k in range(2):
#                 face_mask = cv2.blur(face_mask,(7,7))

            hair = cv2.resize(hair, (256,256))
#             hair = cv2.erode(hair, kernel) # 执行腐蚀操作
#             for k in range(2):
#                 hair = cv2.blur(hair,(7,7))


            face_hair_mask_list.append(face_hair_mask.transpose(2, 0, 1))
            face_mask_list.append(face_mask.transpose(2, 0, 1))
            hair_list.append(hair.transpose(2, 0, 1))
            face_edge_list.append(face_edge_dilate.transpose(2, 0, 1))
            background_list.append(background.transpose(2, 0, 1))
            only_face_list.append(only_face.transpose(2, 0, 1))
            sense5_list.append(sense5.transpose(2, 0, 1))
            eye_brow_list.append(eye_brow.transpose(2, 0, 1))
            eye_eye_list.append(eye_eye.transpose(2, 0, 1))
            if vis:
                cv2.namedWindow('face_hair_mask'+str_w,0)
                cv2.imshow('face_hair_mask'+str_w,face_hair_mask)
                cv2.namedWindow('face_mask'+str_w,0)
                cv2.imshow('face_mask'+str_w,face_mask)
                cv2.namedWindow('hair'+str_w,0)
                cv2.imshow('hair'+str_w,hair)
                cv2.namedWindow('only_face'+str_w,0)
                cv2.imshow('only_face'+str_w,only_face)
                cv2.namedWindow('sense5'+str_w,0)
                cv2.imshow('sense5'+str_w,sense5)
                cv2.namedWindow('eye_brow'+str_w,0)
                cv2.imshow('eye_brow'+str_w,eye_brow)
                cv2.namedWindow('eye_eye'+str_w,0)
                cv2.imshow('eye_eye'+str_w,eye_eye)

                cv2.waitKey(1)
            cv2.imwrite('{}_eye_eye.jpg'.format(str_w),(eye_eye*255).astype(np.uint8))
        #----------------------------
        face_hair_mask_list = np.array(face_hair_mask_list)
        face_hair_mask_list = face_hair_mask_list.astype(np.float32)
        face_mask_list = np.array(face_mask_list)
        face_mask_list = face_mask_list.astype(np.float32)
        hair_list = np.array(hair_list)
        hair_list = hair_list.astype(np.float32)
        face_edge_list = np.array(face_edge_list)
        face_edge_list = face_edge_list.astype(np.float32)
        background_list = np.array(background_list)
        background_list = background_list.astype(np.float32)
        only_face_list = np.array(only_face_list)
        only_face_list = only_face_list.astype(np.float32)
        sense5_list = np.array(sense5_list)
        sense5_list = sense5_list.astype(np.float32)
        eye_brow_list = np.array(eye_brow_list)
        eye_brow_list = eye_brow_list.astype(np.float32)
        eye_eye_list = np.array(eye_eye_list)
        eye_eye_list = eye_eye_list.astype(np.float32)



        if vis:
            print('face_hair_mask size : {} ,face_mask size : {}'.format(face_hair_mask_list.shape,face_mask_list.shape))
        face_hair_mask_list = torch.from_numpy(face_hair_mask_list)
        face_hair_mask_list = face_hair_mask_list.cuda()
        face_mask_list = torch.from_numpy(face_mask_list)
        face_mask_list = face_mask_list.cuda()
        hair_list = torch.from_numpy(hair_list)
        hair_list = hair_list.cuda()
        face_edge_list = torch.from_numpy(face_edge_list)
        face_edge_list = face_edge_list.cuda()
        background_list = torch.from_numpy(background_list)
        background_list = background_list.cuda()
        only_face_list = torch.from_numpy(only_face_list)
        only_face_list = only_face_list.cuda()
        sense5_list = torch.from_numpy(sense5_list)
        sense5_list = sense5_list.cuda()
        eye_brow_list = torch.from_numpy(eye_brow_list)
        eye_brow_list = eye_brow_list.cuda()
        eye_eye_list = torch.from_numpy(eye_eye_list)
        eye_eye_list = eye_eye_list.cuda()


        face_hair_mask_list.requires_grad = False
        face_mask_list.requires_grad = False
        hair_list.requires_grad = False
        face_edge_list.requires_grad = False
        background_list.requires_grad = False
        only_face_list.requires_grad = False
        sense5_list.requires_grad = False
        eye_brow_list.requires_grad = False
        eye_eye_list.requires_grad = False

    return face_hair_mask_list,face_mask_list,hair_list,face_edge_list,background_list,only_face_list,sense5_list,eye_brow_list,eye_eye_list

def create_landmarks_model(landmarks_model_path,landmarks_network= 'resnet_50',landmarks_num_classes = 196,landmarks_img_size=256):
    use_cuda = torch.cuda.is_available()
    #---------------------------------------------------------------- 构建 landmarks 模型
    if landmarks_network == 'resnet_18':
        landmarks_model=resnet18(num_classes=landmarks_num_classes, img_size=landmarks_img_size)
    elif landmarks_network == 'resnet_34':
        landmarks_model=resnet34(num_classes=landmarks_num_classes, img_size=landmarks_img_size)
    elif landmarks_network == 'resnet_50':
        landmarks_model=resnet50(num_classes=landmarks_num_classes, img_size=landmarks_img_size)
    elif landmarks_network == 'resnet_101':
        landmarks_model=resnet101(num_classes=landmarks_num_classes, img_size=landmarks_img_size)
    elif landmarks_network == 'resnet_152':
        landmarks_model=resnet152(num_classes=landmarks_num_classes, img_size=landmarks_img_size)
    elif landmarks_network == 'mobilenetv2':
        landmarks_model=MobileNetV2(n_class =ops.landmarks_num_classes, input_size=ops.landmarks_img_size[0])
    else:
        print('error no the struct model : {}'.format(ops.model))

    device = torch.device("cuda:0" if use_cuda else "cpu")


    # 加载测试模型
    if os.access(landmarks_model_path,os.F_OK):# checkpoint
        # chkpt = torch.load(ops.landmarks_model, map_location=device)
        # landmarks_model.load_state_dict(chkpt)

        chkpt = torch.load(landmarks_model_path, map_location=lambda storage, loc: storage)
        landmarks_model.load_state_dict(chkpt)
        landmarks_model.eval() # 设置为前向推断模式
        print('load landmarks model : {}'.format(landmarks_model_path))
        print('\n/******************* landmarks model acc  ******************/')
        acc_model('',landmarks_model)
    landmarks_model = landmarks_model.to(device)

    return landmarks_model
def get_landmarks_mask(batch_size):
    labels_list = []
    labels_list_n = []
    for i in range(batch_size):
        labelx_ = []
        labelx_n_ = []
        for j in range(196):
            if 65>= j >=0:
                labelx_.append(1.)
            else:
                labelx_.append(0.)

            if ((195)>= j >=(192)) or ((109)>= j >=(108)):
                labelx_n_.append(1.)
            else:
                labelx_n_.append(0.)

        labels_list.append(labelx_)
        labels_list_n.append(labelx_n_)

    labels_list = np.array(labels_list)
    labels_list = labels_list.astype(np.float32)
    labels_list_n = np.array(labels_list_n)
    labels_list_n = labels_list_n.astype(np.float32)

    use_cuda = torch.cuda.is_available()
    labels_list = torch.from_numpy(labels_list)
    labels_list_n = torch.from_numpy(labels_list_n)
    if use_cuda:
        labels_list = labels_list.cuda()  # (bs, 3, h, w)
        labels_list_n = labels_list_n.cuda()

    labels_list.requires_grad = False
    labels_list_n.requires_grad = False
    return labels_list,labels_list_n


if __name__ == '__main__':
    #------------------------------------
    print('\n/************************/\n')
    model_faceparse = create_faceparse_model('./faceparse_model/face_parse_latest2.pth')
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    print('\n/************************/\n')
    #------------------------------------
    landmarks_model = create_landmarks_model('./landmarks_model/resnet50_epoch-2350.pth')
    #------------------------------------
    vis_landmarks = False
    batch_size = 6

    lr_G = 4e-4
    lr_D = 4e-4

#     lr_G = 4e-5
#     lr_D = 4e-5

    max_epoch = 2000
    show_step = 2
    save_epoch = 1
    model_save_path = './saved_models/'
    optim_level = 'O0'

    # fine_tune_with_identity = False

    device = torch.device('cuda')
    # torch.set_num_threads(12)

    G = AEI_Net(c_id=512).to(device)
    D = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d).to(device)
    G.train()
    D.train()

    arcface = Backbone(50, 0.6, 'ir_se').to(device)
    arcface.eval()
    arcface.load_state_dict(torch.load('./id_model/model_ir_se50.pth', map_location=device), strict=False)
    # weight_decay (float, optional)：权重衰减(如L2惩罚)(默认: 0)
    opt_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0, 0.999),weight_decay = 0.)
    opt_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0, 0.999),weight_decay = 0.)

#     opt_G = optim.SGD(G.parameters(), momentum=0.999,lr=lr_G,weight_decay = 0., nesterov = True) # Nestrerov = true 加了二阶导数
#     opt_D = optim.SGD(D.parameters(), momentum=0.9999,lr=1e-4,weight_decay = 0., nesterov = True)

#     opt_D = optim.RMSprop(D.parameters(), lr=lr_D, alpha=0.999, eps=1e-08, weight_decay=0., momentum=0.999, centered=True)


    G, opt_G = amp.initialize(G, opt_G, opt_level=optim_level)
    D, opt_D = amp.initialize(D, opt_D, opt_level=optim_level)

    try:
        p_G = './saved_mask_landmarks_models/GEYE_ps_latest.pth'
        p_D = './saved_mask_landmarks_models/DEYE_ps_latest.pth'
        G.load_state_dict(torch.load(p_G, map_location=torch.device('cpu')), strict=False)
        D.load_state_dict(torch.load(p_D, map_location=torch.device('cpu')), strict=False)

        print('p_G : ',p_G)
        print('p_D : ',p_D)

    except Exception as e:
        print(e)

    dataset = FaceEmbed(['./train_datasets/Foreign-2020-09-06/'], same_prob=0.3,v_num = 3)
#     dataset = FaceEmbed(['./train_datasets/suu/'], same_prob=0.3,v_num = 3)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True)


    MSE = torch.nn.MSELoss()
    L1 = torch.nn.L1Loss()

    # print(torch.backends.cudnn.benchmark)
    torch.backends.cudnn.benchmark = True
    for epoch in range(0, max_epoch):
        # torch.cuda.empty_cache()
        for iteration, data in enumerate(dataloader):
            start_time = time.time()
            Xs, Xt, Xs_0,same_person = data
            Xs = Xs.to(device)
            Xt = Xt.to(device)
            Xs_0 = Xs_0.to(device) # 不带椒盐噪声

            #
            eye_area_mask= torch.zeros(batch_size,3,256,256)
            eye_area_mask[:,:,85:155,40:216] = 1.
            eye_area_mask = eye_area_mask.to(device)
            Xs0_eye = Xs_0*eye_area_mask
            Xt_eye = Xt*eye_area_mask

            #
            # embed = embed.to(device)
            with torch.no_grad():
                embed, Xs_feats = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
                embed_0, Xs_feats_0 = arcface(F.interpolate(Xs_0[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
            same_person = same_person.to(device)
            #diff_person = (1 - same_person)

            # train G
            opt_G.zero_grad()

            Y, Xt_attr = G(Xt, embed)
            #--------------------------------------------------
            Y_eye = Y*eye_area_mask
            f1_, _ = arcface(F.interpolate(Xs0_eye, [112, 112], mode='bilinear', align_corners=True))
            f2_, _ = arcface(F.interpolate(Y_eye, [112, 112], mode='bilinear', align_corners=True))
            L_id_eye =(1. - torch.cosine_similarity(f1_, f2_, dim=1)).mean()
            L_id_eye = L_id_eye*4.5


#             L_eye_rec_same = torch.sum(0.5 * torch.mean(torch.pow(Y_eye - Xt_eye, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
#             L_eye_rec_diff = torch.sum(3.25 * torch.mean(torch.pow(Y_eye - Xt_eye, 2).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)
#             L_eye_rec = L_eye_rec_same + L_eye_rec_diff

            print(' ------>>> L_id_eye : {} '.format(L_id_eye.item()))
            #--------------------------------------------------

            Di = D(Y)
            L_adv = 0

            for di in Di:
                L_adv += hinge_loss(di[0], True)


            Y_aligned = Y[:, :, 19:237, 19:237]
            ZY, Y_feats = arcface(F.interpolate(Y_aligned, [112, 112], mode='bilinear', align_corners=True))
            L_id =(1 - torch.cosine_similarity(embed_0, ZY, dim=1)).mean()

            Y_attr = G.get_attr(Y)
            L_attr = 0
            for i in range(len(Xt_attr)):
                L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(batch_size, -1), dim=1).mean()
            L_attr /= 2.0
            #--------------------------------------------------------------------------- face landmarks

            # print('Xt size :',Xt.size())
            Xt_landmarks = Xt*0.5+0.5
            Y_landmarks = Y*0.5+0.5
            # print('Xt_landmarks size :',Xt_landmarks.size())
            landmarks_Xt = landmarks_model((Xt-0.5).float())
            landmarks_Y = landmarks_model((Y-0.5).float())

            landmarks_mask,landmarks_mask_n = get_landmarks_mask(batch_size)


            # print('landmarks_Xt size : ',landmarks_Xt.size())
            # print('landmarks_mask size : ',landmarks_mask.size())
            if True:
                L_landmarks_pattern = 'ABS'
                L_landmarks_same = torch.sum(250. * torch.mean(torch.abs((landmarks_Xt - landmarks_Y)*landmarks_mask).reshape(batch_size, -1), dim=1)* same_person)/ (same_person.sum() + 1e-6)
                L_landmarks_diff = torch.sum(300. * torch.mean(torch.abs((landmarks_Xt - landmarks_Y)*landmarks_mask).reshape(batch_size, -1), dim=1)* same_person.lt(1.))/ (same_person.lt(1.).sum() + 1e-6)

                L_landmarks_same2 = torch.sum(250. * torch.mean(torch.abs((landmarks_Xt - landmarks_Y)*landmarks_mask_n).reshape(batch_size, -1), dim=1)* same_person)/ (same_person.sum() + 1e-6)
                L_landmarks_diff2 = torch.sum(50. * torch.mean((torch.gt((torch.abs((landmarks_Xt - landmarks_Y)*landmarks_mask_n).reshape(batch_size, -1)-0.08),0.)).float(), dim=1)* same_person.lt(1.))/ (same_person.lt(1.).sum() + 1e-6)

            else:
                L_landmarks_pattern = 'POW'
                L_landmarks_same = torch.sum(25000. * torch.mean(torch.pow((landmarks_Xt - landmarks_Y)*landmarks_mask, 2).reshape(batch_size, -1), dim=1)* same_person)/ (same_person.sum() + 1e-6)
                L_landmarks_diff = torch.sum(30000. * torch.mean(torch.pow((landmarks_Xt - landmarks_Y)*landmarks_mask, 2).reshape(batch_size, -1), dim=1)* same_person.lt(1.))/ (same_person.lt(1.).sum() + 1e-6)
                L_landmarks_same2 = torch.sum(25000. * torch.mean(torch.pow((landmarks_Xt - landmarks_Y)*landmarks_mask_n, 2).reshape(batch_size, -1), dim=1)* same_person)/ (same_person.sum() + 1e-6)
                L_landmarks_diff2 = torch.sum(30000. * torch.mean(torch.pow((landmarks_Xt - landmarks_Y)*landmarks_mask_n, 2).reshape(batch_size, -1), dim=1)* same_person.lt(1.))/ (same_person.lt(1.).sum() + 1e-6)
            L_landmarks = 1.6*(L_landmarks_diff + L_landmarks_same + L_landmarks_same2 + L_landmarks_diff2)
            print('                         L_landmarks_diff:{},L_landmarks_same:{}'.format(L_landmarks_diff.item(),L_landmarks_same.item()))
            print(' --->>> --->>> Pattern : {} loss_landmarks : {}'.format(L_landmarks_pattern,L_landmarks.item()))


            if vis_landmarks:
                output_Xt = landmarks_Xt.cpu().detach().numpy()
                output_Y = landmarks_Y.cpu().detach().numpy()

                Xt_landmarks_s = (((Xt_landmarks[0].detach().cpu().numpy())*255).transpose(1,2,0)).astype(np.uint8)
                Y_landmarks_s = (((Y_landmarks[0].detach().cpu().numpy())*255).transpose(1,2,0)).astype(np.uint8)
                # Xt_landmarks_s = np.array(Xt_landmarks_s)
                # Y_landmarks_s = np.array(Y_landmarks_s)
                # print('----->>> Xt_landmarks_s',Xt_landmarks_s.shape)


                hm_Xt = get_heatmap(Xt_landmarks_s, output_Xt[0],radius=7,img_size = 256,gaussian_op = True)
                hm_Y = get_heatmap(Y_landmarks_s, output_Y[0],radius=7,img_size = 256,gaussian_op = True)

                cv2.namedWindow('heatmap_Xt',0)
                cv2.imshow('heatmap_Xt',hm_Xt)
                cv2.namedWindow('heatmap_Y',0)
                cv2.imshow('heatmap_Y',hm_Y)
                cv2.imshow('landmarks_Xt_s',Xt_landmarks_s)
                cv2.imshow('Y_landmarks_s',Y_landmarks_s)
            #--------------------------------------------------------------------------- face mask
            face_hair_mask_Xt,face_mask_Xt,hair_Xt,edge_Xt,bg_Xt,only_face_Xt,sense5_Xt,eye_brow_Xt,eye_eye_Xt = create_mask(Xt,'Xt',vis = False)
            face_hair_mask_Y,face_mask_Y,hair_Y,edge_Y,bg_Y,only_face_Y,sense5_Y,eye_brow_Y,eye_eye_Y = create_mask(Y,'Y',vis = False)

            #--------------------- two eye id loss
            _,_,_,_,_,_,_,_,eye_eye_Xs = create_mask(Xs_0,'Xs',vis = False)
            Xs0_eye = Xs_0*eye_eye_Xs
#             Y_eye = Y*(eye_eye_Xt.bool()|eye_eye_Y.bool()).float()
            Y_eye = Y*eye_eye_Xt

            print('----->>>> --->>>> eye_eye_Xs sum :',(torch.sum(eye_eye_Xs)).item())

            if ((torch.sum(eye_eye_Xs)).item() > 0.) and ((torch.sum(eye_eye_Y)).item() > 0.):
                f1_eye, _ = arcface(F.interpolate(Xs0_eye, [112, 112], mode='bilinear', align_corners=True))
                f2_eye, _ = arcface(F.interpolate(Y_eye, [112, 112], mode='bilinear', align_corners=True))
                L_id_eye_eye =(1. - torch.cosine_similarity(f1_eye, f2_eye, dim=1)).mean()
                L_id_eye_eye = L_id_eye_eye*8.
            else:
                L_id_eye_eye = 0.
            print(' --->>> two eyes : L_id_eye_eye : ',L_id_eye_eye.item())
            #---------------------------------------------------------------------------
            Y_eye2 = Y*face_mask_Y
            Xt_eye2 = Xt*face_mask_Xt
            f11_, _ = arcface(F.interpolate(Xt_eye2, [112, 112], mode='bilinear', align_corners=True))
            f22_, _ = arcface(F.interpolate(Y_eye2, [112, 112], mode='bilinear', align_corners=True))
            L_id_eye1 =(1. - torch.cosine_similarity(f11_, f22_, dim=1)*same_person).mean() # 相同人差别小

            #不同人
            margin = 0.3
            diff_mask = ((torch.cosine_similarity(f11_, f22_, dim=1)-margin)<0.).float()# 不同人差距小于阈值需要计算损失
            L_id_eye2 =((margin - torch.cosine_similarity(f11_, f22_, dim=1))*diff_mask*same_person.lt(1.)).mean() # 不同人差别大
            L_id_eye_onlyp = L_id_eye1 + L_id_eye2

            print(' --->>> L_id_eye_onlyp : ',L_id_eye_onlyp.item())
            #---------------------------------------------------------------------------
            L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
            L_rec2 = torch.sum(200. * torch.mean(torch.pow(torch.mul(Y,face_hair_mask_Y) - torch.mul(Xt,face_hair_mask_Xt), 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
            L_rec3 = torch.sum(2. * torch.mean(torch.pow(torch.mul(Y,face_mask_Y) - torch.mul(Xt,face_mask_Xt), 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
            L_rec4 = torch.sum(60. * torch.mean(torch.pow(torch.mul(Y,hair_Y) - torch.mul(Xt,hair_Xt), 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
            L_rec5 = torch.sum(50. * torch.mean(torch.pow(torch.mul(Y,edge_Y) - torch.mul(Xt,edge_Xt), 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
            L_rec6 = torch.sum(8. * torch.mean(torch.pow(torch.mul(Y,bg_Y) - torch.mul(Xt,bg_Xt), 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
            L_rec7 = torch.sum(300. * torch.mean(torch.pow(torch.mul(torch.mul(Y,only_face_Xt),only_face_Y) - torch.mul(torch.mul(Xt,only_face_Xt),only_face_Y), 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

            L_rec_diff = torch.sum(3.25 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)

            L_rec1_diff = torch.sum(0. * torch.mean(torch.abs(Y - Xt).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)

            mask_reg = ((face_hair_mask_Y+face_hair_mask_Xt)>0.).float()
            L_rec2_diff = torch.sum(1. * torch.mean(torch.pow(torch.mul(Y,mask_reg) - torch.mul(Xt,mask_reg), 2).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)

            L_rec2_1_diff = torch.sum(0. * torch.mean(torch.abs(torch.mul(Y,face_hair_mask_Y) - torch.mul(Xt,face_hair_mask_Xt)).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)

            mask_reg = ((face_mask_Y+face_mask_Xt)>0.).float()
            L_rec3_diff = torch.sum(0.8* torch.mean(torch.pow(torch.mul(Y,mask_reg) - torch.mul(Xt,mask_reg), 2).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)

            mask_reg = ((hair_Y+hair_Xt)>0.).float()
            L_rec4_diff = torch.sum(8.5 * torch.mean(torch.pow(torch.mul(Y,mask_reg) - torch.mul(Xt,mask_reg), 2).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)

            mask_reg = ((edge_Y+edge_Xt)>0.).float()
            L_rec5_diff = torch.sum(0.25* torch.mean(torch.pow(torch.mul(Y,mask_reg) - torch.mul(Xt,mask_reg), 2).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)
            L_rec6_diff = torch.sum(6. * torch.mean(torch.pow(torch.mul(Y,bg_Y) - torch.mul(Xt,bg_Xt), 2).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)

#             L_rec7_diff = torch.sum(5. * torch.mean(torch.pow(torch.mul(torch.mul(Y,only_face_Xt),only_face_Y) - torch.mul(torch.mul(Xt,only_face_Xt),only_face_Y), 2).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)

            L_rec7_diff = torch.sum(18. * torch.mean(torch.pow(torch.mul(Y,only_face_Xt) - torch.mul(Xt,only_face_Xt), 2).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)


            print('\n\n')
            print('---------->>> L_rec L_rec2 L_rec3 L_rec4 L_rec5 L_rec6 L_rec7 loss : ',L_rec.item(),L_rec2.item(),L_rec3.item(),L_rec4.item(),L_rec5.item(),L_rec6.item(),L_rec7.item())
            print('---------->>> L_rec L_rec1_diff L_rec2_1_diff L_rec2 L_rec3 L_rec4 L_rec5 L_rec6 L_rec7_diff diff loss : ',L_rec_diff.item(),L_rec1_diff.item(),L_rec2_diff.item(),L_rec2_1_diff.item(),L_rec3_diff.item(),L_rec4_diff.item(),0.,L_rec6_diff.item(),L_rec7_diff.item())
            print('\n\n')
            L_rec = L_rec + L_rec2 + L_rec3 + L_rec4 + L_rec5 +L_rec6+ L_rec7 \
            + L_rec_diff +L_rec2_diff + L_rec3_diff + L_rec4_diff + L_rec7_diff

            L_rec = L_rec*1.4
            # print('Y,Xt : ',Y.size(),Xt.size(),Xt_p.size())

            lossG = 1.*L_adv + 10.*L_attr + 23.723*L_id + 8.*L_rec + L_landmarks + 1.*L_id_eye + L_id_eye_eye*2.

            lossG = lossG*2.
            # lossG = 1*L_adv + 10*L_attr + 5*L_id + 10*L_rec
            with amp.scale_loss(lossG, opt_G) as scaled_loss:
                scaled_loss.backward()

            # lossG.backward()
            opt_G.step()

            # train D
            opt_D.zero_grad()
            # with torch.no_grad():
            #     Y, _ = G(Xt, embed)
            fake_D = D(Y.detach())
            loss_fake = 0.
            for di in fake_D:
                loss_fake += hinge_loss(di[0], False)# 改此权重会让变伪能力增强

            true_D = D(Xs_0)
            loss_true = 0.
            for di in true_D:
                loss_true += hinge_loss(di[0], True)
            # true_score2 = D(Xt)[-1][0]
            #-----------------------------------------------------------------
            true_D = D(Xt)
            for di in true_D:
                loss_true += hinge_loss(di[0], True)

            lossD1 = 1.*(1.*loss_true.mean() + 1.*loss_fake.mean())
            #-----------------------------------------------------------------
            #Xs0_eye,Y_eye
            fake_eye_D = D(Y_eye.detach())
            loss_eye_fake = 0.
            for di in fake_eye_D:
                loss_eye_fake += hinge_loss(di[0], False)# 改此权重会让变伪能力增强

            true_eye_D = D(Xs0_eye)
            loss_eye_true = 0.
            for di in true_eye_D:
                loss_eye_true += hinge_loss(di[0], True)

            true_eye_D = D(Xt_eye)
            for di in true_eye_D:
                loss_eye_true += hinge_loss(di[0], True)

            lossD_eye = 1.*(0.6*loss_eye_true.mean() + 1.*loss_eye_fake.mean())
            #-----------------------------------------------------------------
            #--------------------- sense5
            mask_reg = (sense5_Y.bool()|sense5_Xt.bool()).float()
            lossD2 = 0.
            if True:
                fake_sense5_D = D(torch.mul(Y.detach(),mask_reg))
                loss_sense5_fake = 0.
                for di in fake_sense5_D:
                    loss_sense5_fake += hinge_loss(di[0], False)# 改此权重会让变伪能力增强

                true_sense5_D = D(torch.mul(Xt,mask_reg))
                loss_sense5_true = 0.
                for di in true_sense5_D:
                    loss_sense5_true += hinge_loss(di[0], True)

                lossD2 = 1.*(1.*loss_sense5_true.mean() + 1.*loss_sense5_fake.mean())

            #--------------------- eye_brow
            mask_reg = (eye_brow_Y.bool()|eye_brow_Xt.bool()).float()
            lossD3 = 0.
            if True:
                fake_eye_brow_D = D(torch.mul(Y.detach(),mask_reg))
                loss_eye_brow_fake = 0
                for di in fake_eye_brow_D:
                    loss_eye_brow_fake += hinge_loss(di[0], False)# 改此权重会让变伪能力增强

                true_eye_brow_D = D(torch.mul(Xt,mask_reg))
                loss_eye_brow_true = 0
                for di in true_eye_brow_D:
                    loss_eye_brow_true += hinge_loss(di[0], True)

                lossD3 = 0.1*(1.*loss_eye_brow_true.mean() + 1.*loss_eye_brow_fake.mean())

            #--------------------- face hair
            lossD4 = 0.
            if True:
                fake_hair_D = D(torch.mul(Y.detach(),face_hair_mask_Y))# #fake.detaach()进行反向截断，不让梯度流到G中，只反传到最初输入节点即可。
                loss_hair_fake = 0.
                for di in fake_hair_D:
                    loss_hair_fake += hinge_loss(di[0], False)# 改此权重会让变伪能力增强

                true_hair_D = D(torch.mul(Xt,face_hair_mask_Xt))
                loss_hair_true = 0.
                for di in true_hair_D:
                    loss_hair_true += hinge_loss(di[0], True)

                lossD4 = 1.6*(1.*loss_hair_true.mean() + 1.*loss_hair_fake.mean())
            #--------------------- only face
            mask_reg = (only_face_Y.bool()|only_face_Xt.bool()).float()
            lossD5 = 0.
            if True:
                fake_only_face_D = D(torch.mul(Y.detach(),mask_reg))
                loss_only_face_fake = 0.
                for di in fake_only_face_D:
                    loss_only_face_fake += hinge_loss(di[0], False)# 改此权重会让变伪能力增强

                true_only_face_D = D(torch.mul(Xt,mask_reg))
                loss_only_face_true = 0.
                for di in true_only_face_D:
                    loss_only_face_true += hinge_loss(di[0], True)

                lossD5 = 0.1*(1.*loss_only_face_true.mean() + 1.*loss_only_face_fake.mean())
            #--------------------- face
#             mask_reg = (face_mask_Y.bool() | face_mask_Xt.bool()).float()
            lossD6 = 0.
            if True:
                fake_face_mask_D = D(torch.mul(Y.detach(),face_mask_Y))
                loss_face_mask_fake = 0.
                for di in fake_face_mask_D:
                    loss_face_mask_fake += hinge_loss(di[0], False)# 改此权重会让变伪能力增强

                true_face_mask_D = D(torch.mul(Xt,face_mask_Xt))
                loss_face_mask_true = 0.
                for di in true_face_mask_D:
                    loss_face_mask_true += hinge_loss(di[0], True)

                lossD6 = 1.*(1.*loss_face_mask_true.mean() + 1.*loss_face_mask_fake.mean())
            #---------------------------------------------------------------------------
            lossD = lossD1*1. + (lossD2 + lossD3 + lossD4 + lossD5 + lossD6)*0.01 + lossD_eye*0.5
            print('---->> lossD1:{} , lossD2: {}, lossD3: {}, lossD4: {}, lossD5: {},lossD6: {} ,lossD_eye :{}'.format(lossD1.item(),lossD2.item(),lossD3.item(),lossD4.item(),lossD5.item(),lossD6.item(),lossD_eye.item()))

            with amp.scale_loss(lossD, opt_D) as scaled_loss:
                scaled_loss.backward()
            # lossD.backward()
            opt_D.step()
            batch_time = time.time() - start_time
            if iteration % show_step == 0:
                image = make_image(Xs, Xt, Y)
                # vis.image(image[::-1, :, :], opts={'title': 'result'}, win='result')
                cv2.imwrite('./gen_images/latest_ps_eye_AEI_landmarks_mask.jpg', (image*255).transpose([1,2,0]).astype(np.uint8))
            print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
            print(f'lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s')
            print(f'L_adv: {L_adv.item()} L_id: {L_id.item()} L_attr: {L_attr.item()} L_rec: {L_rec.item()}')

            if ((iteration % 100) == 0) and (iteration > 0):
                torch.save(G.state_dict(), './saved_mask_landmarks_models/GEYE_ps_latest.pth')
                torch.save(D.state_dict(), './saved_mask_landmarks_models/DEYE_ps_latest.pth')
        torch.save(G.state_dict(), './saved_mask_landmarks_models/GEYE_ps_epoch_{}.pth'.format(epoch))
        torch.save(D.state_dict(), './saved_mask_landmarks_models/DEYE_ps_epoch_{}.pth'.format(epoch))
