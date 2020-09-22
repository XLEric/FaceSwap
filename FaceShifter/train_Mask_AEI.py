import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from network.AEI_Net import *
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

def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1-X).mean()
    else:
        return torch.relu(X+1).mean()


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

    heatmap1 = CenterGaussianHeatMap(width, height, cx, cy, 200)
    heatmap2 = CenterGaussianHeatMap(width, height, cx, cy, 350)
    face_hair_mask_list = []
    face_mask_list = []
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
            num_of_class = np.max(parsing)

            for pi in range(1, num_of_class + 1):
                index = np.where(parsing == pi)# 获得对应分类的的像素坐标
                # vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]# 给对应的类别的掩码赋值

                if pi in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
                    face_hair_mask[index[0], index[1]] = 1.
                    face_mask[index[0], index[1]] = 1.
                elif pi in [17,18]:
                    hair[index[0], index[1]] = 1.

            face_hair_mask = hair + face_hair_mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            face_hair_mask = cv2.dilate(face_hair_mask, kernel)
            face_hair_mask = face_hair_mask * np.expand_dims(heatmap1,axis =2)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            face_mask = cv2.dilate(face_mask, kernel)
            
            face_mask = face_mask * np.expand_dims(heatmap2,axis =2)
            face_hair_mask = np.minimum(1.,face_hair_mask[:,:,:])
            face_mask = np.minimum(1.,face_mask[:,:,:])

            face_hair_mask = cv2.resize(face_hair_mask, (256,256))
            face_mask = cv2.resize(face_mask, (256,256))
            face_hair_mask_list.append(face_hair_mask.transpose(2, 0, 1))
            face_mask_list.append(face_mask.transpose(2, 0, 1))
            if vis:
                cv2.namedWindow('face_hair_mask'+str_w,0)
                cv2.imshow('face_hair_mask'+str_w,face_hair_mask)
                cv2.namedWindow('face_mask'+str_w,0)
                cv2.imshow('face_mask'+str_w,face_mask)
                cv2.waitKey(1)
        #----------------------------
        face_hair_mask_list = np.array(face_hair_mask_list)
        face_hair_mask_list = face_hair_mask_list.astype(np.float32)
        face_mask_list = np.array(face_mask_list)
        face_mask_list = face_mask_list.astype(np.float32)
        if vis:
            print('face_hair_mask size : {} ,face_mask size : {}'.format(face_hair_mask_list.shape,face_mask_list.shape))
        face_hair_mask_list = torch.from_numpy(face_hair_mask_list)
        face_hair_mask_list = face_hair_mask_list.cuda()
        face_mask_list = torch.from_numpy(face_mask_list)
        face_mask_list = face_mask_list.cuda()
    return face_hair_mask_list,face_mask_list

if __name__ == '__main__':
    #------------------------------------
    print('/************************/\n')
    model_faceparse = create_faceparse_model('./faceparse_model/face_parse_latest.pth')
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    print('\n/************************/\n')
    #------------------------------------
    batch_size = 8
    lr_G = 4e-4
    lr_D = 4e-4
    max_epoch = 2000
    show_step = 10
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

    opt_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0, 0.999))

    G, opt_G = amp.initialize(G, opt_G, opt_level=optim_level)
    D, opt_D = amp.initialize(D, opt_D, opt_level=optim_level)

    try:
        G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=torch.device('cpu')), strict=False)
        D.load_state_dict(torch.load('./saved_models/D_latest.pth', map_location=torch.device('cpu')), strict=False)
    except Exception as e:
        print(e)

    dataset = FaceEmbed(['./train_datasets/Foreign-2020-09-06/'], same_prob=0.35)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)


    MSE = torch.nn.MSELoss()
    L1 = torch.nn.L1Loss()

    # print(torch.backends.cudnn.benchmark)
    torch.backends.cudnn.benchmark = True
    for epoch in range(0, max_epoch):
        # torch.cuda.empty_cache()
        for iteration, data in enumerate(dataloader):
            start_time = time.time()
            Xs, Xt, same_person = data
            Xs = Xs.to(device)
            Xt = Xt.to(device)
            # embed = embed.to(device)
            with torch.no_grad():
                embed, Xs_feats = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
            same_person = same_person.to(device)
            #diff_person = (1 - same_person)

            # train G
            opt_G.zero_grad()

            Y, Xt_attr = G(Xt, embed)

            Di = D(Y)
            L_adv = 0

            for di in Di:
                L_adv += hinge_loss(di[0], True)


            Y_aligned = Y[:, :, 19:237, 19:237]
            ZY, Y_feats = arcface(F.interpolate(Y_aligned, [112, 112], mode='bilinear', align_corners=True))
            L_id =(1 - torch.cosine_similarity(embed, ZY, dim=1)).mean()

            Y_attr = G.get_attr(Y)
            L_attr = 0
            for i in range(len(Xt_attr)):
                L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(batch_size, -1), dim=1).mean()
            L_attr /= 2.0
            #---------------------------------------------------------------------------
            face_hair_mask_Xt,face_mask_Xt = create_mask(Xt,'Xt',vis = False)
            face_hair_mask_Y,face_mask_Y = create_mask(Y,'Y',vis = False)
            #---------------------------------------------------------------------------
            L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
            L_rec2 = torch.sum(0.75 * torch.mean(torch.pow(torch.mul(Y,face_hair_mask_Y) - torch.mul(Xt,face_hair_mask_Xt), 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
            L_rec3 = torch.sum(0.75 * torch.mean(torch.pow(torch.mul(Y,face_mask_Y) - torch.mul(Xt,face_mask_Xt), 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

            L_rec_diff = torch.sum(0.25 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)
            L_rec2_diff = torch.sum(0.25 * torch.mean(torch.pow(torch.mul(Y,face_hair_mask_Y) - torch.mul(Xt,face_hair_mask_Xt), 2).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)
            L_rec3_diff = torch.sum(0.25 * torch.mean(torch.pow(torch.mul(Y,face_mask_Y) - torch.mul(Xt,face_mask_Xt), 2).reshape(batch_size, -1), dim=1) * same_person.lt(1.)) / (same_person.lt(1.).sum() + 1e-6)

            L_rec = L_rec + L_rec2 + L_rec3 + L_rec_diff + L_rec2_diff + L_rec3_diff
            print('\n\n')
            print('---------->>> L_rec L_rec2 L_rec3 loss : ',L_rec.item(),L_rec2.item(),L_rec3.item())
            print('---------->>> L_rec L_rec2 L_rec3 diff loss : ',L_rec_diff.item(),L_rec2_diff.item(),L_rec3_diff.item())
            print('\n\n')
            # print('Y,Xt : ',Y.size(),Xt.size(),Xt_p.size())

            lossG = 1.*L_adv + 10.*L_attr + 20.*L_id + 12.*L_rec
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
            loss_fake = 0
            for di in fake_D:
                loss_fake += hinge_loss(di[0], False)

            true_D = D(Xs)
            loss_true = 0
            for di in true_D:
                loss_true += hinge_loss(di[0], True)
            # true_score2 = D(Xt)[-1][0]

            lossD = 0.5*(loss_true.mean() + loss_fake.mean())

            with amp.scale_loss(lossD, opt_D) as scaled_loss:
                scaled_loss.backward()
            # lossD.backward()
            opt_D.step()
            batch_time = time.time() - start_time
            if iteration % show_step == 0:
                image = make_image(Xs, Xt, Y)
                # vis.image(image[::-1, :, :], opts={'title': 'result'}, win='result')
                cv2.imwrite('./gen_images/latest_AEI_mask.jpg', (image*255).transpose([1,2,0]).astype(np.uint8))
            print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
            print(f'lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s')
            print(f'L_adv: {L_adv.item()} L_id: {L_id.item()} L_attr: {L_attr.item()} L_rec: {L_rec.item()}')
            if iteration % 250 == 0 and iteration>0:
                torch.save(G.state_dict(), './saved_mask_models/G_latest.pth')
                torch.save(D.state_dict(), './saved_mask_models/D_latest.pth')
        torch.save(G.state_dict(), './saved_mask_models/G_epoch_{}.pth'.format(epoch))
        torch.save(D.state_dict(), './saved_mask_models/D_epoch_{}.pth'.format(epoch))
