# Author: X.L.Eric

import sys
sys.path.append('./face_modules/')
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from network.AEI_Net import *
from face_modules.mtcnn import *
import cv2
import PIL.Image as Image
import numpy as np
import time
import copy
#------------------------------
from model import BiSeNet
import os
import torchvision.transforms as transforms
from tools import *
#------------------------------


import os
import sys
import time
sys.path.append('./')
import argparse
import torch
import torch.backends.cudnn as cudnn
from models.faceboxes import FaceBoxes

from utils.timer import Timer

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.mobilenetv2 import MobileNetV2
from utils.common_utils import *
from acc_model import acc_model

parser = argparse.ArgumentParser(description='FaceBoxes')


if __name__ == '__main__':
    #-----------------------------------------------
    parser = argparse.ArgumentParser(description=' FaceBoxes Inferece')
    # FaceBoxes_epoch_290
    parser.add_argument('-m', '--detect_model', default='weights_face/Final_FaceBoxes.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--GPUS', type=str, default = '0',help = 'GPUS') # GPU选择
    parser.add_argument('--confidence_threshold', default=0.65, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=200, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.25, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=200, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.65, type=float, help='visualization_threshold')
    #-----------------------------------------------------------------------------------------
    parser.add_argument('--landmarks_model', type=str, default = './landmarks_model/resnet50_epoch-2350.pth',
        help = 'landmarks_model') # 模型路径
    parser.add_argument('--landmarks_network', type=str, default = 'resnet_50',
        help = 'model : resnet_18,resnet_34,resnet_50,resnet_101,resnet_152,mobilenetv2') # 模型类型
    parser.add_argument('--landmarks_num_classes', type=int , default = 196,
        help = 'landmarks_num_classes') #  分类类别个数
    parser.add_argument('--landmarks_img_size', type=tuple , default = (256,256),
        help = 'landmarks_img_size') # 输入landmarks 模型图片尺寸
    #-----------------------------------------------------------------------------------------
    parser.add_argument('--force_cpu', type=bool, default = False,
        help = 'force_cpu') # 前向推断硬件选择
    parser.add_argument('--max_batch_size', type=int , default = 1,
        help = 'max_batch_size') #  最大 landmarks - max_batch_size

    parser.add_argument('--test_path', type=str, default = '../chapter_07/video/jk_1.mp4',
        help = 'test_path') # 测试文件路径

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(ops) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))
    use_cuda = torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    torch.set_num_threads(1)
    if use_cuda:
        cudnn.benchmark = True

    #---------------------------------------------------------------- 构建 landmarks 模型
    if ops.landmarks_network == 'resnet_18':
        landmarks_model=resnet18(num_classes=ops.landmarks_num_classes, img_size=ops.landmarks_img_size[0])
    elif ops.landmarks_network == 'resnet_34':
        landmarks_model=resnet34(num_classes=ops.landmarks_num_classes, img_size=ops.landmarks_img_size[0])
    elif ops.landmarks_network == 'resnet_50':
        landmarks_model=resnet50(num_classes=ops.landmarks_num_classes, img_size=ops.landmarks_img_size[0])
    elif ops.landmarks_network == 'resnet_101':
        landmarks_model=resnet101(num_classes=ops.landmarks_num_classes, img_size=ops.landmarks_img_size[0])
    elif ops.landmarks_network == 'resnet_152':
        landmarks_model=resnet152(num_classes=ops.landmarks_num_classes, img_size=ops.landmarks_img_size[0])
    elif ops.landmarks_network == 'mobilenetv2':
        landmarks_model=MobileNetV2(n_class =ops.landmarks_num_classes, input_size=ops.landmarks_img_size[0])
    else:
        print('error no the struct model : {}'.format(ops.model))

    device = torch.device("cuda:0" if use_cuda else "cpu")


    # 加载测试模型
    if os.access(ops.landmarks_model,os.F_OK):# checkpoint
        # chkpt = torch.load(ops.landmarks_model, map_location=device)
        # landmarks_model.load_state_dict(chkpt)

        chkpt = torch.load(ops.landmarks_model, map_location=lambda storage, loc: storage)
        landmarks_model.load_state_dict(chkpt)
        landmarks_model.eval() # 设置为前向推断模式
        print('load landmarks model : {}'.format(ops.landmarks_model))
        print('\n/******************* landmarks model acc  ******************/')
        acc_model(ops,landmarks_model)
    landmarks_model = landmarks_model.to(device)


    #--------------------------------------------------------------------------- 构建人脸检测模型
    # detect_model
    detect_model = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    detect_model = load_model(detect_model, ops.detect_model, True)
    detect_model.eval()
    print('\n/******************* detect model acc  ******************/')
    acc_model(ops,detect_model)
    detect_model = detect_model.to(device)

    print('Finished loading model!')
    # print(detect_model)

    detect_model = detect_model.to(device)
    #-----------------------------------------------
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = './faceparse_model/face_parse_latest.pth'
    print('model : {}'.format(save_pth))
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    #-----------------------------------------------
    detector = MTCNN()
    device = torch.device('cuda')
    G = AEI_Net(c_id=512)
    G.eval()
    G.load_state_dict(torch.load('./saved_mask_models/G_latest_s1.pth', map_location=torch.device('cpu')))
    G = G.cuda()

    arcface = Backbone(50, 0.6, 'ir_se').to(device)
    arcface.eval()
    arcface.load_state_dict(torch.load('./id_model/model_ir_se50.pth', map_location=device), strict=False)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    Xs_path = './samples/zb1.jpg'
    # Xt_path = './samples/s1.jpg'

    Xs_raw = cv2.imread(Xs_path)
    Xs_raw_r = cv2.resize(Xs_raw, (int(200*Xs_raw.shape[1]/Xs_raw.shape[0]),200), interpolation = cv2.INTER_LINEAR)
    print(Xs_raw.shape)

    cv2.namedWindow('source',0)
    cv2.imshow('source', Xs_raw)

    video_capture = cv2.VideoCapture('./test_video/zb2.mp4')
    # video_capture = cv2.VideoCapture('E:/m_cc/cv_cource/chapter_07/video/rw_4.mp4')


    loc_time = time.localtime()

    str_time = time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)
    flag_video_start = False

    idx = 0
    Xs0 = None
    while True:
        ret, Xt_raw = video_capture.read()

        if ret:
            idx += 1


            # if idx%2!=0 :
            #     continue
            # Xt_raw = np.rot90(Xt_raw)
            # Xt_raw = cv2.flip(Xt_raw,-1)
            if flag_video_start == False:
                video_writer = cv2.VideoWriter("./demo/demo_{}.mp4".format(str_time), cv2.VideoWriter_fourcc(*"mp4v"), fps=25, frameSize=(int(Xt_raw.shape[1]*2), int(Xt_raw.shape[0])))
                flag_video_start = True
            if 1:
                Xt_raw_w = copy.deepcopy(Xt_raw)

                # Xs = detector.align(Image.fromarray(Xs_raw), crop_size=(256, 256))
                if Xs0 is None:
                    img_xs_raw = copy.deepcopy(Xs_raw)
                    fs_landmarks = []
                    dets = detect_faces(ops,detect_model,img_xs_raw,device)
                    fs_dets,fs_landmarks = get_faces_batch_landmarks(ops,landmarks_model,dets,img_xs_raw,1.,use_cuda,draw_bbox = False)
                    Xs0 = detector.align_face_boxes(Image.fromarray(Xs_raw), fs_landmarks,crop_size=(256, 256), vis = False,return_trans_inv=False)
                Xs = copy.deepcopy(Xs0)
                # Xt = detector.align(Image.fromarray(Xt_raw), crop_size=(256, 256))

                try:
                    fs_landmarks = []
                    img_raw = copy.deepcopy(Xt_raw)
                    scale_img = 800./float(img_raw.shape[1])

                    img_raw = cv2.resize(img_raw, (int(img_raw.shape[1]*scale_img),int(img_raw.shape[0]*scale_img)), interpolation=cv2.INTER_LINEAR)

                    dets = detect_faces(ops,detect_model,img_raw,device)
                    fs_dets,fs_landmarks = get_faces_batch_landmarks(ops,landmarks_model,dets,img_raw,scale_img,use_cuda,draw_bbox = True)
                    Xt = detector.align_face_boxes(Image.fromarray(Xt_raw), fs_landmarks,crop_size=(256, 256), vis = False,return_trans_inv=False)

                    cv2.namedWindow('video',0)
                    cv2.imshow('video',img_raw)
                    # cv2.waitKey(1)
                    # continue
                except:
                    print('error ~ detect face')
                    continue

                if Xt is None:
                    Xt_raw_w2 = copy.deepcopy(Xt_raw_w)

                    Xt_raw_w2[0:Xs_raw_r.shape[0],0:Xs_raw_r.shape[1],:] = Xs_raw_r
                    # cv2.namedWindow('raw',0)
                    # cv2.imshow('raw', Xt_raw_w)
                    # cv2.namedWindow('fusion',0)
                    # cv2.imshow('fusion', merge)

                    img_stack = np.hstack([Xt_raw_w,Xt_raw_w2])
                    video_writer.write(img_stack)


                    cv2.namedWindow('fusion',0)
                    cv2.imshow('fusion', img_stack)
                    cv2.waitKey(1)

                    continue


                # print(Xs)
                # print(Xt)


                # Xs_raw = np.array(Xs)
                Xs = test_transform(Xs)
                Xs = Xs.unsqueeze(0).cuda()


                Xt_raw = np.array(Xt)
                Xt = test_transform(Xt)
                Xt = Xt.unsqueeze(0).cuda()
                with torch.no_grad():
                    embeds, _ = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))


                mask = np.zeros([256, 256], dtype=np.float)
                for i in range(256):
                    for j in range(256):
                        dist = np.sqrt((i-64)**2 + (j-64)**2)/64
                        dist = np.minimum(dist, 1)
                        mask[i, j] = 1.-dist/20.
                # mask = cv2.dilate(mask, None, iterations=20)

                #---------------------------------------------------------------

                Xt_img = Image.fromarray(Xt_raw_w)
#
                # Xt, trans_inv = detector.align(Xt_img, crop_size=(256, 256), vis = True,return_trans_inv=True)
                #------------------------------------------------
                # img_raw = copy.deepcopy(Xt_raw_w)
                # dets = detect_faces(ops,detect_model,img_raw,device)
                # fs_dets,fs_landmarks = get_faces_batch_landmarks(ops,landmarks_model,dets,img_raw,use_cuda,draw_bbox = False)
                Xt,trans_inv = detector.align_face_boxes(Xt_img, fs_landmarks,crop_size=(256, 256), vis = False,return_trans_inv=True)
                #-------------------------------------------------

                Xt = test_transform(Xt)

                Xt = Xt.unsqueeze(0)
                with torch.no_grad():
                    st = time.time()
                    Yt, _ = G(Xt.cuda(), embeds.cuda())
                    Yt = Yt.squeeze().detach().cpu().numpy()
                    st = time.time() - st
                    print(idx,') ',f'inference time: {st} sec')
                    Yt = Yt.transpose([1, 2, 0])*0.5 + 0.5
                    #------------------------------------- 通过人脸分割做掩码
                    # cv2.imshow('mask_origin',mask)
                    cv2.namedWindow('Yt',0)
                    cv2.imshow('Yt',Yt)
                    # 锐化
                    # Yt = cv2.blur(Yt,(3,3))
                    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
                    # Yt = cv2.filter2D(Yt, -1, kernel=kernel)
                    Yt = np.minimum(1,Yt[:,:,:])
                    Yt = np.maximum(0,Yt[:,:,:])
                    #
                    # Yt = cv2.GaussianBlur(Yt,(3,3),0)
                    #
                    Yt_ = np.minimum(255,Yt[:,:,:]*255)
                    Yt_ = Image.fromarray(Yt_.astype('uint8')).convert('RGB')

                    Yt_0 = Yt_.resize((512, 512), Image.BILINEAR)
                    print('--------------->>',Yt_.size)
                    Yt_ = to_tensor(Yt_0)
                    Yt_ = torch.unsqueeze(Yt_, 0)
                    Yt_ = Yt_.cuda()
                    out = net(Yt_)[0]
                    parsing = out.squeeze(0).cpu().numpy().argmax(0)
                    face_hair_mask = vis_parsing_maps(Yt_0, parsing, stride=1, save_im=False, save_path='')
                    if np.sum(face_hair_mask)!=0:
                        mask = mask*face_hair_mask
                    # cv2.imshow('mask_face_output',mask)
                    #------------------------------------ 通过人脸分割做掩码 well done


                    Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw_w, 1), np.size(Xt_raw_w, 0)), borderMode=cv2.BORDER_TRANSPARENT)

                    mask_ = cv2.warpAffine(mask, trans_inv, (np.size(Xt_raw_w, 1), np.size(Xt_raw_w, 0)), borderMode=cv2.BORDER_TRANSPARENT)

                    mask_ = np.expand_dims(mask_, 2)
                #----------------------------------------------------------------------------------
                if True:
                    Yt_trans_inv = mask_*Yt_trans_inv + (1.-mask_)*(Xt_raw_w.astype(np.float)/255.)
                    merge = (Yt_trans_inv*255).astype(np.uint8)
                else:
                    print('mask shape :',mask_.shape)
                    bpsong_mask = np.uint8((mask_[:,:,0] > 0.)*255)

                    hs, ws = np.where(bpsong_mask > 0)
                    hc = int(hs.min() + hs.max()) // 2
                    wc = int(ws.min() + ws.max()) // 2

                    Yt_trans_inv = cv2.seamlessClone((Yt_trans_inv*255).astype(np.uint8),Xt_raw_w, bpsong_mask, (wc, hc), cv2.NORMAL_CLONE)
                    for i in range(3):
                        Yt_trans_inv = cv2.seamlessClone((Yt_trans_inv),Xt_raw_w, bpsong_mask, (wc, hc), cv2.NORMAL_CLONE)
                    cv2.namedWindow('bpsong_mask',0)
                    cv2.imshow('bpsong_mask',bpsong_mask)

                    merge = Yt_trans_inv

                #----------------------------------------------------------------------------------


                # Xt_raw_w = cv2.resize(Xt_raw_w, (1280,720), interpolation = cv2.INTER_LINEAR)
                #
                # merge = cv2.resize(merge, (1280,720), interpolation = cv2.INTER_LINEAR)

                merge[0:Xs_raw_r.shape[0],0:Xs_raw_r.shape[1],:] = Xs_raw_r
                # cv2.namedWindow('raw',0)
                # cv2.imshow('raw', Xt_raw_w)
                # cv2.namedWindow('fusion',0)
                # cv2.imshow('fusion', merge)


                img_stack = np.hstack([Xt_raw_w,merge])
                img_stack = cv2.resize(img_stack, (int(img_stack.shape[1]),int(img_stack.shape[0])), interpolation = cv2.INTER_LINEAR)
                video_writer.write(img_stack)


                cv2.namedWindow('fusion',0)
                cv2.imshow('fusion', img_stack)
                print('idx-',idx,' ',img_stack.shape)

                if cv2.waitKey(1) == 27:
                    break
            # except:
            #     print('error~')
            #     continue
    cv2.destroyAllWindows()
    video_writer.release()
