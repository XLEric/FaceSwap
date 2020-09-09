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

def swap_faces(Xs_raw, Xt_raw):
    # print ("2----")
    # detector = MTCNN()
    # device = torch.device('cpu')
    # G = AEI_Net(c_id=512)
    # G.eval()
    # G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=device))
    # G = G.cpu()

    # arcface = Backbone(50, 0.6, 'ir_se').to(device)
    # arcface.eval()
    # arcface.load_state_dict(torch.load('./id_model/model_ir_se50.pth', map_location=device), strict=False)
    #
    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    #
    # Xs_img = Image.fromarray(Xs_raw)
    # Xs = detector.align(Xs_img, crop_size=(256, 256))
    #
    # Xs = test_transform(Xs)
    # Xs = Xs.unsqueeze(0).cpu()
    # with torch.no_grad():
    #     embeds, Xs_feats = arcface(F.interpolate(Xs, (112, 112), mode='bilinear', align_corners=True))
    #     embeds = embeds.mean(dim=0, keepdim=True)

    mask = np.zeros([256, 256], dtype=np.float)
    for i in range(256):
        for j in range(256):
            dist = np.sqrt((i-32)**2 + (j-32)**2)/32
            dist = np.minimum(dist, 1)
            mask[i, j] = 1.-dist/10
    mask = cv2.dilate(mask, None, iterations=20)

    Xt_img = Image.fromarray(Xt_raw)

    Xt, trans_inv = detector.align(Xt_img, crop_size=(256, 256), return_trans_inv=True)

    Xt = test_transform(Xt)

    Xt = Xt.unsqueeze(0).cpu()
    with torch.no_grad():
        st = time.time()
        Yt, _ = G(Xt, embeds)
        Yt = Yt.squeeze().detach().cpu().numpy()
        st = time.time() - st
        print(f'inference time: {st} sec')
        Yt = Yt.transpose([1, 2, 0])*0.5 + 0.5
        Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderMode=cv2.BORDER_TRANSPARENT)
        mask_ = cv2.warpAffine(mask, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderMode=cv2.BORDER_TRANSPARENT)
        mask_ = np.expand_dims(mask_, 2)
        Yt_trans_inv = mask_*Yt_trans_inv + (1-mask_)*(Xt_raw.astype(np.float)/255.)

        merge = Yt_trans_inv

        return merge

import dlib



if __name__ == '__main__':
    #人脸分类器
    detector_dlib = dlib.get_frontal_face_detector()
    # 获取人脸检测器
    predictor = dlib.shape_predictor(
        "./shape_predictor_68_face_landmarks.dat"
    )
    #-----------------------------------------------
    detector = MTCNN()
    device = torch.device('cuda')
    G = AEI_Net(c_id=512)
    G.eval()
    G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=torch.device('cpu')))
    G = G.cuda()

    arcface = Backbone(50, 0.6, 'ir_se').to(device)
    arcface.eval()
    arcface.load_state_dict(torch.load('./id_model/model_ir_se50.pth', map_location=device), strict=False)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    Xs_path = './samples/hd.jpg'
    # Xt_path = './samples/s1.jpg'

    Xs_raw = cv2.imread(Xs_path)
    Xs_raw_r = cv2.resize(Xs_raw, (256,256), interpolation = cv2.INTER_LINEAR)
    print(Xs_raw.shape)

    cv2.namedWindow('source',0)
    cv2.imshow('source', Xs_raw)

    video_capture = cv2.VideoCapture('./video/video1.mp4')
    # video_capture = cv2.VideoCapture('E:/m_cc/cv_cource/chapter_07/video/rw_4.mp4')


    loc_time = time.localtime()

    str_time = time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)

    video_writer = cv2.VideoWriter("./demo/demo_{}.mp4".format(str_time), cv2.VideoWriter_fourcc(*"mp4v"), fps=15, frameSize=(2560, 720))
    idx = 0
    while True:
        ret, Xt_raw = video_capture.read()
        if ret:
            idx += 1
            if 1:
                Xt_raw_w = copy.deepcopy(Xt_raw)

                Xs = detector.align(Image.fromarray(Xs_raw), crop_size=(256, 256))
                Xt = detector.align(Image.fromarray(Xt_raw), crop_size=(256, 256))

                if Xt is None:
                    Xt_raw_w = cv2.resize(Xt_raw_w, (1280,720), interpolation = cv2.INTER_LINEAR)

                    Xt_raw_w2 = copy.deepcopy(Xt_raw_w)
                    Xt_raw_w2[0:256,0:256,:] = Xs_raw_r
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
                        mask[i, j] = 1.-dist/8.99
                mask = cv2.dilate(mask, None, iterations=20)



                #---------------------------------------------------------------
                gray = cv2.cvtColor(Xt_raw_w, cv2.COLOR_BGR2GRAY)
                dets = detector_dlib(gray, 1)


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
                    Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw_w, 1), np.size(Xt_raw_w, 0)), borderMode=cv2.BORDER_TRANSPARENT)
                    mask_ = cv2.warpAffine(mask, trans_inv, (np.size(Xt_raw_w, 1), np.size(Xt_raw_w, 0)), borderMode=cv2.BORDER_TRANSPARENT)
                    mask_ = np.expand_dims(mask_, 2)



                #-----------------------------------------------------------
                if False:
                    for face in dets:
                        shape = predictor(Xt_raw_w, face)  # 寻找人脸的68个标定点
                        # 遍历所有点，打印出其坐标，并圈出来
                        f_idx  = 0
                        #
                        pts_cc = []
                        for pt in shape.parts():
                            pt_pos = (pt.x, pt.y)

                            if f_idx <=26:
                                pts_cc.append([int(pt.x),int(pt.y)])


                            # if f_idx<=16:
                            #     cv2.circle(Xt_raw_w, pt_pos, 2, (255, 0, 0), -1)
                            #
                            # elif 21>=f_idx>=17:
                            #     cv2.circle(Xt_raw_w, pt_pos, 2, (0, 0, 255), -1)
                            # elif 26>=f_idx>=22:
                            #     cv2.circle(Xt_raw_w, pt_pos, 2, (0, 255, 255), -1)
                            # else :
                            #     cv2.circle(Xt_raw_w, pt_pos, 1, (0, 255, 0), -1)

                            f_idx += 1
                    #-----------------------------------------------------------
                    points_array = np.zeros((1,27,2),dtype = np.int32)
                    for k in range(len(pts_cc)):
                        if k <=16:
                            points_array[0,k,0] = pts_cc[k][0]
                            points_array[0,k,1] = pts_cc[k][1]

                    points_array[0,17] = pts_cc[26]
                    points_array[0,18] = (pts_cc[25][0],pts_cc[25][1]-27)
                    points_array[0,19] = (pts_cc[24][0],pts_cc[24][1]-39)
                    points_array[0,20] = (pts_cc[23][0],pts_cc[23][1]-42)
                    points_array[0,21] = (pts_cc[22][0],pts_cc[22][1]-43)
                    points_array[0,22] = (pts_cc[21][0],pts_cc[21][1]-43)
                    points_array[0,23] = (pts_cc[20][0],pts_cc[20][1]-42)
                    points_array[0,24] = (pts_cc[19][0],pts_cc[19][1]-39)
                    points_array[0,25] = (pts_cc[18][0],pts_cc[18][1]-27)
                    points_array[0,26] = pts_cc[17]

                    # cv2.drawContours(Xt_raw_w,points_array,-1,(255,255,0),thickness=-1)
                    mask_roi = np.zeros([Xt_raw_w.shape[0], Xt_raw_w.shape[1]], dtype=np.float)
                    cv2.drawContours(mask_roi,points_array,-1,(1.),thickness=-1)
                    mask_roi = cv2.dilate(mask_roi, None, iterations=12)
                    cv2.imshow('mask_roi',mask_roi)

                    mask_ = np.expand_dims(mask_roi, 2)*mask_

                #----------------------------------------------------------------------------------

                Yt_trans_inv = mask_*Yt_trans_inv + (1-mask_)*(Xt_raw_w.astype(np.float)/255.)
                cv2.imshow('mask',mask_)
                merge = Yt_trans_inv

                #----------------------------------------------------------------------------------


                Xt_raw_w = cv2.resize(Xt_raw_w, (1280,720), interpolation = cv2.INTER_LINEAR)

                merge = cv2.resize(merge, (1280,720), interpolation = cv2.INTER_LINEAR)
                merge = (merge*255).astype(np.uint8)
                merge[0:256,0:256,:] = Xs_raw_r
                # cv2.namedWindow('raw',0)
                # cv2.imshow('raw', Xt_raw_w)
                # cv2.namedWindow('fusion',0)
                # cv2.imshow('fusion', merge)

                img_stack = np.hstack([Xt_raw_w,merge])
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
