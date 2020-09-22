# -*- encoding: utf-8 -*-

from model import BiSeNet
import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

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

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path=''):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    hair_face = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    face_hair_mask = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    face_mask = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    hair = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)# 获得对应分类的的像素坐标
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]# 给对应的类别的掩码赋值

        if pi == 1:# 脸
            hair_face[index[0], index[1], :] = [255,0,0]
        elif pi == 2 or pi == 3:# 脸
            hair_face[index[0], index[1], :] = [0,255,0]
        elif pi == 4 or pi == 5:# 眉毛
            hair_face[index[0], index[1], :] = [0,255,255]
        elif pi == 6:# 眼镜
            hair_face[index[0], index[1], :] = [255,255,0]
        elif pi == 7 or pi == 8:# 耳朵
            hair_face[index[0], index[1], :] = [255,0,255]
        # elif pi == 9:# 耳环
        #     hair_face[index[0], index[1], :] = [0,120,255]
        elif pi == 10:# 鼻子
            hair_face[index[0], index[1], :] = [20,55,255]
        elif pi == 11:# 嘴内侧
            hair_face[index[0], index[1], :] = [25,250,55]
        elif pi == 12:# 嘴上外侧
            hair_face[index[0], index[1], :] = [25,250,155]
        elif pi == 13:# 嘴下外侧
            hair_face[index[0], index[1], :] = [95,250,155]
        elif pi == 14:# 脖子
            hair_face[index[0], index[1], :] = [12,250,185]
        elif pi == 17:# 头发
            hair_face[index[0], index[1], :] = [150,150,150]

        if pi in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
            face_hair_mask[index[0], index[1]] = 1.
            face_mask[index[0], index[1]] = 1.
        elif pi in [17,18]:
            hair[index[0], index[1]] = 1.
        #     face_hair_mask[index[0], index[1]] =0.3

    height, width = np.shape(face_hair_mask)
    cy, cx = height/2.0, width/2.0

    heatmap1 = CenterGaussianHeatMap(width, height, cx, cy, 160)
    heatmap2 = CenterGaussianHeatMap(width, height, cx, cy, 350)

    if 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        face_hair_mask = cv2.erode(face_hair_mask, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        img_dilate = cv2.dilate(face_hair_mask, kernel)

        for k in range(3):
            img_dilate = cv2.dilate(img_dilate, kernel)

        # print(img_dilate)
        hair = np.array(np.logical_xor(img_dilate,hair)).astype(np.float32)

        img_dilate = hair + img_dilate

        img_dilate = np.array(np.logical_xor(face_hair_mask,img_dilate)).astype(np.float32)

        img_dilate = img_dilate*heatmap1

        face_hair_mask = img_dilate + face_hair_mask
    else:
        face_hair_mask = hair + face_hair_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        face_hair_mask = cv2.dilate(face_hair_mask, kernel)
        face_hair_mask = face_hair_mask * heatmap1
        face_mask = face_mask * heatmap2
        face_hair_mask = np.minimum(1.,face_hair_mask[:,:])
        face_mask = np.minimum(1.,face_mask[:,:])


    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    hair_face = hair_face.astype(np.uint8)

    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    # Save result or not
    if save_im:
        im_stack = np.hstack((cv2.cvtColor(im, cv2.COLOR_RGB2BGR),vis_im))
        cv2.namedWindow('image',0)
        cv2.imshow('image',im_stack)
        cv2.namedWindow('mask_color',0)
        cv2.imshow('mask_color',vis_parsing_anno_color)
        cv2.namedWindow('hair_face',0)
        cv2.imshow('hair_face',hair_face)
        cv2.namedWindow('face_hair_mask',0)
        cv2.imshow('face_hair_mask',face_hair_mask)
        cv2.namedWindow('face_mask',0)
        cv2.imshow('face_mask',face_mask)
        # cv2.namedWindow('img_dilate',0)
        # cv2.imshow('img_dilate',img_dilate)


        cv2.namedWindow('heatmap1',0)
        cv2.imshow('heatmap1',heatmap1)

        cv2.imwrite(save_path, im_stack)



def evaluate( dspth='./', cp=''):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = cp
    print('model : {}'.format(save_pth))
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        idx = 0
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            print('--------------->>',image.size)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            idx += 1
            print('<{}> image : '.format(idx),np.unique(parsing))
            test_result = './test_result'
            if not osp.exists(test_result):
                os.makedirs(test_result)
            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(test_result, image_path))
            if cv2.waitKey(500) == 27:
                break
if __name__ == "__main__":
    evaluate(dspth='images', cp='./faceparse_model/face_parse_latest.pth')
