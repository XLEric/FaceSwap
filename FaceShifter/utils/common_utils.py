#-*-coding:utf-8-*-
# date:2020-04-11
# Author: xiang li

import os
import shutil
import cv2
import numpy as np
import json
import torch
from config import cfg
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode
def mkdir_(path, flag_rm=False):
    if os.path.exists(path):
        if flag_rm == True:
            shutil.rmtree(path)
            os.mkdir(path)
            print('remove {} done ~ '.format(path))
    else:
        os.mkdir(path)

def plot_box(bbox, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)# 目标的bbox
    if label:
        tf = max(tl - 2, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0] # label size
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 # 字体的bbox
        cv2.rectangle(img, c1, c2, color, -1)  # label 矩形填充
        # 文本绘制
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255],thickness=tf, lineType=cv2.LINE_AA)

class JSON_Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JSON_Encoder, self).default(obj)

def draw_landmarks(img,output,r_bboxes,draw_circle):
    img_width = img.shape[1]
    img_height = img.shape[0]
    dict_landmarks = {}

    # 眼镜、鼻子、嘴角5个关键点
    faceswap_list = []
    faceswap_list.append((output[96*2+0]*float(img_width)+r_bboxes[0],output[96*2+1]*float(img_height)+r_bboxes[1]))
    faceswap_list.append((output[97*2+0]*float(img_width)+r_bboxes[0],output[97*2+1]*float(img_height)+r_bboxes[1]))
    faceswap_list.append((output[54*2+0]*float(img_width)+r_bboxes[0],output[54*2+1]*float(img_height)+r_bboxes[1]))
    faceswap_list.append((output[76*2+0]*float(img_width)+r_bboxes[0],output[76*2+1]*float(img_height)+r_bboxes[1]))
    faceswap_list.append((output[82*2+0]*float(img_width)+r_bboxes[0],output[82*2+1]*float(img_height)+r_bboxes[1]))
    # 98个关键点遍历，方便绘制每个部分的轮廓
    for i in range(int(output.shape[0]/2)):
        x = output[i*2+0]*float(img_width)
        y = output[i*2+1]*float(img_height)
        #
        if 41>= i >=33:
            if 'left_eyebrow' not in dict_landmarks.keys():
                dict_landmarks['left_eyebrow'] = []
            dict_landmarks['left_eyebrow'].append([int(x),int(y),(0,255,0)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (0,255,0),-1)
        elif 50>= i >=42:
            if 'right_eyebrow' not in dict_landmarks.keys():
                dict_landmarks['right_eyebrow'] = []
            dict_landmarks['right_eyebrow'].append([int(x),int(y),(0,255,0)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (0,255,0),-1)
        elif 67>= i >=60:
            if 'left_eye' not in dict_landmarks.keys():
                dict_landmarks['left_eye'] = []
            dict_landmarks['left_eye'].append([int(x),int(y),(255,55,255)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (255,0,255),-1)
        elif 75>= i >=68:
            if 'right_eye' not in dict_landmarks.keys():
                dict_landmarks['right_eye'] = []
            dict_landmarks['right_eye'].append([int(x),int(y),(255,55,255)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (255,0,255),-1)
        elif 97>= i >=96:
            cv2.circle(img, (int(x),int(y)), 2, (0,0,255),-1)
        elif 54>= i >=51:
            if 'bridge_nose' not in dict_landmarks.keys():
                dict_landmarks['bridge_nose'] = []
            dict_landmarks['bridge_nose'].append([int(x),int(y),(0,170,255)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (0,170,255),-1)
        elif 32>= i >=0:
            if 'basin' not in dict_landmarks.keys():
                dict_landmarks['basin'] = []
            dict_landmarks['basin'].append([int(x),int(y),(255,30,30)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (255,30,30),-1)
        elif 59>= i >=55:
            if 'wing_nose' not in dict_landmarks.keys():
                dict_landmarks['wing_nose'] = []
            dict_landmarks['wing_nose'].append([int(x),int(y),(0,255,255)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (0,255,255),-1)
        elif 87>= i >=76:
            if 'out_lip' not in dict_landmarks.keys():
                dict_landmarks['out_lip'] = []
            dict_landmarks['out_lip'].append([int(x),int(y),(255,255,0)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (255,255,0),-1)
        elif 95>= i >=88:
            if 'in_lip' not in dict_landmarks.keys():
                dict_landmarks['in_lip'] = []
            dict_landmarks['in_lip'].append([int(x),int(y),(50,220,255)])
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (50,220,255),-1)
        else:
            if draw_circle:
                cv2.circle(img, (int(x),int(y)), 2, (255,0,255),-1)

    faceswap_list_e = []

    for i in range(5):
        faceswap_list_e.append(faceswap_list[i][0])
    for i in range(5):
        faceswap_list_e.append(faceswap_list[i][1])


    return dict_landmarks,faceswap_list_e

def draw_contour(image,dict,r_bbox):
    x0 = r_bbox[0]# 全图偏置
    y0 = r_bbox[1]

    for key in dict.keys():
        # print(key)
        _,_,color = dict[key][0]

        if 'left_eye' == key:
            eye_x = np.mean([dict[key][i][0]+x0 for i in range(len(dict[key]))])
            eye_y = np.mean([dict[key][i][1]+y0 for i in range(len(dict[key]))])
            cv2.circle(image, (int(eye_x),int(eye_y)), 3, (255,255,55),-1)
        if 'right_eye' == key:
            eye_x = np.mean([dict[key][i][0]+x0 for i in range(len(dict[key]))])
            eye_y = np.mean([dict[key][i][1]+y0 for i in range(len(dict[key]))])
            cv2.circle(image, (int(eye_x),int(eye_y)), 3, (255,215,25),-1)

        if 'basin' == key or 'wing_nose' == key:
            pts = np.array([[dict[key][i][0]+x0,dict[key][i][1]+y0] for i in range(len(dict[key]))],np.int32)
            # print(pts)
            cv2.polylines(image,[pts],False,color,thickness = 2)

        else:
            points_array = np.zeros((1,len(dict[key]),2),dtype = np.int32)
            for i in range(len(dict[key])):
                x,y,_ = dict[key][i]
                points_array[0,i,0] = x+x0
                points_array[0,i,1] = y+y0

            # cv2.fillPoly(image, points_array, color)
            cv2.drawContours(image,points_array,-1,color,thickness=2)
def refine_face_bbox(bbox,img_shape):
    height,width,_ = img_shape

    x1,y1,x2,y2 = bbox

    expand_w = (x2-x1)
    expand_h = (y2-y1)

    x1 -= expand_w*0.06
    y1 -= expand_h*0.15
    x2 += expand_w*0.06
    y2 += expand_h*0.06

    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

    x1 = int(max(0,x1))
    y1 = int(max(0,y1))
    x2 = int(min(x2,width-1))
    y2 = int(min(y2,height-1))

    return (x1,y1,x2,y2)
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def detect_faces(ops,detect_model,img_raw,device):
    resize = 1
    img = np.float32(img_raw)
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)


    loc, conf = detect_model(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > ops.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:ops.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    #keep = py_cpu_nms(dets, ops.nms_threshold)
    # keep = nms(dets, ops.nms_threshold,force_cpu=True)
    keep = py_cpu_nms(dets, ops.nms_threshold)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:ops.keep_top_k, :]

    return dets



def get_faces_batch_landmarks(ops,landmarks_model,dets,img_raw,scale_img,use_cuda,draw_bbox = True):
    # 绘制图像
    image_batch = None
    r_bboxes = []
    imgs_crop = []
    for b in dets:
        if b[4] < ops.vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))

        r_bbox = refine_face_bbox((b[0],b[1],b[2],b[3]),img_raw.shape)
        r_bboxes.append(r_bbox)
        img_crop = img_raw[r_bbox[1]:r_bbox[3],r_bbox[0]:r_bbox[2]]
        imgs_crop.append(img_crop)
        img_ = cv2.resize(img_crop, (ops.landmarks_img_size[1],ops.landmarks_img_size[0]), interpolation = cv2.INTER_LINEAR) # INTER_LINEAR INTER_CUBIC

        img_ = img_.astype(np.float32)
        img_ = (img_-128.)/256.

        img_ = img_.transpose(2, 0, 1)
        img_ = np.expand_dims(img_,0)

        if image_batch is None:
            image_batch = img_
        else:
            image_batch = np.concatenate((image_batch,img_),axis=0)
    for b in dets:
        if b[4] < ops.vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        if draw_bbox:
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] - 3
        cv2.putText(img_raw, text, (cx, cy),cv2.FONT_HERSHEY_DUPLEX, 0.6, (155, 155, 255),3)
        cv2.putText(img_raw, text, (cx, cy),cv2.FONT_HERSHEY_DUPLEX, 0.6, (155, 10, 10),1)

    # 填充最大 关键点 批次数据
    if len(dets) < ops.max_batch_size:
        im_mask = np.zeros([1,3,ops.landmarks_img_size[0],ops.landmarks_img_size[1]], dtype = np.float32)
        for i in range(ops.max_batch_size-len(dets)):
            if image_batch is None:
                image_batch = im_mask
            else:
                image_batch = np.concatenate((image_batch,im_mask),axis=0)

    image_batch = torch.from_numpy(image_batch).float()

    if use_cuda:
        image_batch = image_batch.cuda()  # (bs, 3, h, w)

    pre_ = landmarks_model(image_batch.float())
    # print(pre_.size())
    output = pre_.cpu().detach().numpy()
    # print('output shape : ',output.shape)
    # n_array = np.zeros([ops.landmarks_img_size[0],ops.landmarks_img_size[1],3], dtype = np.float)
    faceswap_landmarks = []
    for i in range(len(dets)):

        dict_landmarks,list_e = draw_landmarks(imgs_crop[i],output[i],r_bboxes[i],draw_circle = False)
        faceswap_landmarks.append(list_e)

        draw_contour(img_raw,dict_landmarks,r_bboxes[i])
    # print(dets)
    #-----------------------------------------------------------------------------------

    fs_dets = []
    fs_landmarks = []

    for  i in range(len(r_bboxes)):
        fs_dets.append((r_bboxes[i][0]/scale_img,r_bboxes[i][1]/scale_img,r_bboxes[i][2]/scale_img,r_bboxes[i][3]/scale_img))
        fs_land = []
        for k in range(10):
            fs_land.append(faceswap_landmarks[i][k]/scale_img)
        fs_landmarks.append(fs_land)

        bbox = r_bboxes[i]
        min_x = int(bbox[0])
        min_y = int(bbox[1])
        max_x = int(bbox[2])
        max_y = int(bbox[3])
        # cv2.rectangle(img_raw, (min_x, min_y), (max_x, max_y), (255, 0, 255), thickness=11)
        for k in range(5):
            x = int(faceswap_landmarks[i][k+0])
            y = int(faceswap_landmarks[i][k+5])
            if draw_bbox:
                cv2.circle(img_raw,(x,y),5+k*2,(0,0,255),-1)

    cv2.putText(img_raw, 'face:'+str(len(dets)), (3,35),cv2.FONT_HERSHEY_DUPLEX, 1.45, (55, 255, 255),5)
    cv2.putText(img_raw, 'face:'+str(len(dets)), (3,35),cv2.FONT_HERSHEY_DUPLEX, 1.45, (135, 135, 5),2)
    return fs_dets,fs_landmarks
