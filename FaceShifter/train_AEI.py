import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Flag_256 = False
if Flag_256 :
    from network.AEI_Net import *
else:
    from network.AEI_Net_512 import *
    


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

if __name__ == '__main__':
    if Flag_256 :
        print('---------------->>> model 256*256')
    else:
        print('---------------->>> model 512*512')
    if Flag_256 :
        batch_size = 12
    else:
        batch_size = 3
    lr_G = 4e-4
    lr_D = 4e-4
    max_epoch = 2000
    show_step = 5
    save_epoch = 1
    model_save_path = './saved_models/'
    optim_level = 'O0'

    # fine_tune_with_identity = False

    device = torch.device('cuda')
    # torch.set_num_threads(12)

    G = AEI_Net(c_id=512).to(device)
    if Flag_256:
        D = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d).to(device)
    else:
        D = MultiscaleDiscriminator(input_nc=3, n_layers=7, norm_layer=torch.nn.InstanceNorm2d).to(device)
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
        if Flag_256 :
            Finetune_G_Model = './saved_models/G_latest.pth'
            Finetune_D_Model = './saved_models/D_latest.pth'
        else:
            Finetune_G_Model = './saved_models/G_latest_512.pth'
            Finetune_D_Model = './saved_models/D_latest_512.pth'
        
        G.load_state_dict(torch.load(Finetune_G_Model, map_location=torch.device('cpu')), strict=False)
        D.load_state_dict(torch.load(Finetune_D_Model, map_location=torch.device('cpu')), strict=False)
        
        print('Finetune_G_Model : ',Finetune_G_Model)
        print('Finetune_D_Model : ',Finetune_D_Model)
        
    except Exception as e:
        print(e)

    # if not fine_tune_with_identity:
        # dataset = FaceEmbed(['../celeb-aligned-256_0.85/', '../ffhq_256_0.85/', '../vgg_256_0.85/', '../stars_256_0.85/'], same_prob=0.5)
    # else:
        # dataset = With_Identity('../washed_img/', 0.8)
    # dataset = FaceEmbed(['../celeb-aligned-256_0.85/', '../ffhq_256_0.85/', '../vgg_256_0.85/', '../stars_256_0.85/'], same_prob=0.8)

    dataset = FaceEmbed(['./train_datasets/Foreign-2020-09-06/'], same_prob=0.5, Flag_256 = Flag_256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3, drop_last=True)


    MSE = torch.nn.MSELoss()
    L1 = torch.nn.L1Loss()

    # print(torch.backends.cudnn.benchmark)
    torch.backends.cudnn.benchmark = True
    for epoch in range(0, max_epoch):
        # torch.cuda.empty_cache()
        for iteration, data in enumerate(dataloader):
            start_time = time.time()
            Xs, Xt, same_person = data
            print('Xs.size ---->>>',Xs.size())
            Xs = Xs.to(device)
            Xt = Xt.to(device)
            # embed = embed.to(device)
            with torch.no_grad():
                if Flag_256 :
                    embed, Xs_feats = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
                else:
                    embed, Xs_feats = arcface(F.interpolate(Xs[:, :, 38:474, 38:474], [112, 112], mode='bilinear', align_corners=True))
            same_person = same_person.to(device)
            #diff_person = (1 - same_person)

            # train G
            opt_G.zero_grad()
            
            Y, Xt_attr = G(Xt, embed)

            Di = D(Y)
            L_adv = 0

            for di in Di:
                L_adv += hinge_loss(di[0], True)

            if Flag_256 :
                Y_aligned = Y[:, :, 19:237, 19:237]
            else:
                Y_aligned = Y[:, :, 38:474, 38:474]
            ZY, Y_feats = arcface(F.interpolate(Y_aligned, [112, 112], mode='bilinear', align_corners=True))
            L_id =(1 - torch.cosine_similarity(embed, ZY, dim=1)).mean()

            Y_attr = G.get_attr(Y)
            L_attr = 0
            for i in range(len(Xt_attr)):
                L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(batch_size, -1), dim=1).mean()
            L_attr /= 2.0

            L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
            
            if Flag_256 :
                lossG = 1.*L_adv + 10.*L_attr + 20.*L_id + 12.*L_rec
            else:
                lossG = 1.*L_adv + 10.*L_attr + 12.*L_id + 8.*L_rec
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
                if Flag_256 :
                    cv2.imwrite('./gen_images/latest_AEI_256.jpg', (image*255).transpose([1,2,0]).astype(np.uint8))
                else:
                    cv2.imwrite('./gen_images/latest_AEI_512.jpg', (image*255).transpose([1,2,0]).astype(np.uint8))
            print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
            print(f'lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s')
            print(f'L_adv: {L_adv.item()} L_id: {L_id.item()} L_attr: {L_attr.item()} L_rec: {L_rec.item()}')
            if iteration % 300 == 0 and iteration>0:
                if Flag_256 :
                    torch.save(G.state_dict(), './saved_models/G_latest.pth')
                    torch.save(D.state_dict(), './saved_models/D_latest.pth')
                else:
                    torch.save(G.state_dict(), './saved_models/G_latest_512.pth')
                    torch.save(D.state_dict(), './saved_models/D_latest_512.pth')
        if Flag_256 :
            torch.save(G.state_dict(), './saved_models/G_epoch_{}.pth'.format(epoch))
            torch.save(D.state_dict(), './saved_models/D_epoch_{}.pth'.format(epoch))
        else:
            torch.save(G.state_dict(), './saved_models/G_512_epoch_{}.pth'.format(epoch))
            torch.save(D.state_dict(), './saved_models/D_512_epoch_{}.pth'.format(epoch))
