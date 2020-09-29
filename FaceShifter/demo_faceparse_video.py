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

if __name__ == '__main__':
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

    video_capture = cv2.VideoCapture('./test_video/zb3.mp4')
    # video_capture = cv2.VideoCapture('E:/m_cc/cv_cource/chapter_07/video/rw_4.mp4')


    loc_time = time.localtime()

    str_time = time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)
    flag_video_start = False

    idx = 0
    while True:
        ret, Xt_raw = video_capture.read()

        if ret:
            idx += 1

            # if idx%2!=0 :
            #     continue
            # Xt_raw = np.rot90(Xt_raw)
            # Xt_raw = cv2.flip(Xt_raw,-1)
            if flag_video_start == False:
                video_writer = cv2.VideoWriter("./demo/demo_{}.mp4".format(str_time), cv2.VideoWriter_fourcc(*"mp4v"), fps=15, frameSize=(int(Xt_raw.shape[1]*2), int(Xt_raw.shape[0])))
                flag_video_start = True
            if 1:
                Xt_raw_w = copy.deepcopy(Xt_raw)

                Xs = detector.align(Image.fromarray(Xs_raw), crop_size=(256, 256))
                Xt = detector.align(Image.fromarray(Xt_raw), crop_size=(256, 256))

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
                        mask[i, j] = 1.-dist/16.99
                mask = cv2.dilate(mask, None, iterations=20)

                #---------------------------------------------------------------

                Xt_img = Image.fromarray(Xt_raw_w)

                Xt, trans_inv = detector.align(Xt_img, crop_size=(256, 256), return_trans_inv=True)

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
                    cv2.imshow('mask_face_output',mask)
                    #------------------------------------ 通过人脸分割做掩码 well done


                    Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw_w, 1), np.size(Xt_raw_w, 0)), borderMode=cv2.BORDER_TRANSPARENT)

                    mask_ = cv2.warpAffine(mask, trans_inv, (np.size(Xt_raw_w, 1), np.size(Xt_raw_w, 0)), borderMode=cv2.BORDER_TRANSPARENT)

                    mask_ = np.expand_dims(mask_, 2)
                #----------------------------------------------------------------------------------

                Yt_trans_inv = mask_*Yt_trans_inv + (1.-mask_)*(Xt_raw_w.astype(np.float)/255.)
                # Yt_trans_inv = cv2.seamlessClone(Xt_raw_w, Yt_trans_inv,np.where(mask_>0,255,0).astype(np.uint8), (int(Xt_raw_w.shape[1]/2),int(Xt_raw_w.shape[0]/2)), cv2.NORMAL_CLONE)
                cv2.namedWindow('mask',0)
                cv2.imshow('mask',mask_)
                merge = Yt_trans_inv

                #----------------------------------------------------------------------------------


                # Xt_raw_w = cv2.resize(Xt_raw_w, (1280,720), interpolation = cv2.INTER_LINEAR)
                #
                # merge = cv2.resize(merge, (1280,720), interpolation = cv2.INTER_LINEAR)
                merge = (merge*255).astype(np.uint8)
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

                if cv2.waitKey(1) == 27:
                    break
            # except:
            #     print('error~')
            #     continue
    cv2.destroyAllWindows()
    video_writer.release()
