import sys
sys.path.append('./')
# from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from torchvision import transforms as trans
import PIL.Image as Image
from mtcnn import MTCNN
import torch
import cv2
import os
import time
import random
import shutil
import cv2
import numpy as np
import json

def mkdir_(path, flag_rm=False):
    if os.path.exists(path):
        if flag_rm == True:
            shutil.rmtree(path)
            os.mkdir(path)
            print('remove {} done ~ '.format(path))
    else:
        os.mkdir(path)

if __name__ == '__main__':

    img_root_dir = 'E:/tools_e/FaceBBox_Tools/Make_Image_kk3'
    save_path = '../Asian_datasets/Asian_kk3'

    #--------------------------------------
    mkdir_(save_path, flag_rm=True)

    device = torch.device('cuda:0')
    mtcnn = MTCNN()

    # threshold = 1.54
    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    #--------------------------------------

    ind = 0
    embed_map = {}

    for root, dirs, files in os.walk(img_root_dir):
        for name in files:
            if name.endswith('jpg') or name.endswith('png'):
                try:
                    p = os.path.join(root, name)
                    img = cv2.imread(p)[:, :, ::-1]
                    # faces = mtcnn.align_multi(Image.fromarray(img), min_face_size=64, crop_size=(128, 128))
                    faces = mtcnn.align_multi(Image.fromarray(img), min_face_size=64, crop_size=(256, 256))
                    if len(faces) == 0:
                        continue
                    for face in faces:
                        # scaled_img = face.resize((64, 64), Image.ANTIALIAS)
                        scaled_img = face.resize((256, 256), Image.ANTIALIAS)
                        loc_time = time.localtime()


                        # new_path = '%08d.jpg'%ind
                        new_path = '{}_{:8}.jpg'.format(time.strftime("%Y-%m-%d_%H-%M-%S", loc_time),random.randint(0,999999))
                        # print('             ',new_path)
                        ind += 1
                        print(ind,') ',new_path)
                        scaled_img.save(os.path.join(save_path, new_path))
                except Exception as e:
                    continue
