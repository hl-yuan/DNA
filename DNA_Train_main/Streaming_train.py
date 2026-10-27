import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from torch import optim
import torch.nn.functional as F
import data_loader
import random as rn
from tqdm import tqdm
from Nmetrics import evaluate
import torch
from sklearn.mixture import GaussianMixture
from Loss import *
from Knowledge_Generation import *
from utils import *
from network import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_view_0(decoder_model, Dual_attention, optimizer_Streaming, Xv, Y, v, path, config, cluster, Buffer, loss_para):
    seed = 0
    np.random.seed(seed)
    rn.seed(seed)

    train_dataset = data_loader.TrainDataset_Com(Xv, Y)
    batch_sampler = data_loader.Data_Sampler(train_dataset, shuffle=True, batch_size=config.batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    t_progress = tqdm(range(config.epochs), desc=f'Streaming_View{v}_training')
    for epoch in t_progress:
        index_swap = None
        tot_loss = 0.0
        Proto_total = []
        sumloss = 0
        Rec_loss = Loss_Reconstruction().to(device)
        Guide_loss = Loss_Konw_Guide().to(device)

        for batch_idx, (xs, ys) in enumerate(train_loader):

            xs = np.squeeze(xs, axis=0)
            xs = torch.as_tensor(xs, device=device)
            decoder_model = decoder_model.to(device)
            Dual_attention = Dual_attention.to(device)
            xs = xs.squeeze(0)

            z, xr = decoder_model(xs)

            re_loss = Rec_loss(xs, xr)

            z = z.squeeze(0)
            z_a, c_a, attn_a = Dual_attention(z, index_swap)


            # print( "loss_guid=", loss_guid, "     ", "loss_constraint=", loss_constraint)
            loss_guid = Guide_loss(c_a,z)
            loss = loss_para[0]*(re_loss)+loss_para[1]*(loss_guid)
            sumloss = loss_guid + sumloss

            optimizer_Streaming.zero_grad()
            loss.backward()
            optimizer_Streaming.step()
            tot_loss = float(loss + tot_loss)
            Proto_total = Proto_total + c_a.tolist()
        # print("sumloss=", sumloss)
        Proto_total = torch.tensor(Proto_total)
        Buffer[0] = Proto_total

        if epoch+1 == 1 or (epoch + 1) % config.epochs == 0:
            with torch.no_grad():
                Dual_attention_G = dual_attention(dim=config.feature_dim, num=(config.Knowledge_size * (batch_idx + 1))).to(device)

                Xv = torch.as_tensor(Xv, device=device)
                fea_emb, Xr = decoder_model(Xv)
                Z_fea, P_fea, A_fea = Dual_attention_G(fea_emb, index_swap)
                Labels = Y[0]
                estimator = KMeans(n_clusters=cluster)
                fea_cluster = Z_fea.detach().cpu().numpy()
                estimator.fit(fea_cluster)
                pred_final = estimator.labels_

                acc, nmi, ari = evaluate(Labels, pred_final)


    return Z_fea, Buffer




def train_view(decoder_model, Dual_attention, optimizer_Streaming, Xv, Y, v, path, config, cluster, Buffer, loss_para):
    seed = 0
    np.random.seed(seed)
    rn.seed(seed)


    t_progress = tqdm(range(config.epochs), desc=f'Streaming_View{v}_training')
    fusion_model = Fast_DynamicFusion(
        feature_dim=config.feature_dim,
        similarity_alpha=config.similarity_alpha
    ).to(device)
    optimizer_Streaming.add_param_group({'params': fusion_model.parameters()})
    for epoch in t_progress:
        index_swap = None
        tot_loss = 0.0

        Proto_total = []
        Rec_loss = Loss_Reconstruction().to(device)
        Guide_loss = Loss_Konw_Guide().to(device)
        NMI_loss = Normalized_Mutual_Information().to(device)
        sumloss = 0
        for xs, ys, p_BF, batch_No in next_batch(Xv, Y, Buffer[0], config.batch_size, config.Knowledge_size):

            xs = torch.as_tensor(xs, device=device)
            p_BF = torch.as_tensor(p_BF, device=device)
            decoder_model = decoder_model.to(device)
            Dual_attention = Dual_attention.to(device)
            fusion_model = fusion_model.to(device)
            xs = xs.squeeze(0)
            z, xr = decoder_model(xs)
            re_loss1 = Rec_loss(xs, xr)
            z = z.squeeze(0)

            z_a, c_a, attn_a = Dual_attention(z, index_swap)
            # z_a, p_a, _, _, _= Information_aggregation(z, p_BF)
            loss_NMI = NMI_loss(c_a,p_BF)

            P_F,_ = fusion_model(p_BF, c_a)

            loss_guid = Guide_loss(P_F, z_a)

            loss = loss_para[0] * re_loss1 + loss_para[1] * loss_guid - loss_NMI

            # print( "loss_guid=", loss_guid, "     ", "re_loss2=", loss_NMI)
            sumloss  =loss_NMI + sumloss
            # print("sumloss=", sumloss)
            optimizer_Streaming.zero_grad()
            loss.backward()
            optimizer_Streaming.step()
            # tot_loss = float(loss + tot_loss)
            Proto_total = Proto_total + P_F.tolist()
        # print("sumloss=", sumloss)
        Proto_total = torch.tensor(Proto_total)
        Buffer[0] = Proto_total



        if epoch+1 == 1 or (epoch + 1) % config.epochs == 0:
            with torch.no_grad():
                Dual_attention_G = dual_attention(dim=config.feature_dim, num=(config.Knowledge_size * (batch_No))).to(device)
                Buffer[0] = torch.as_tensor(Buffer[0], device=device)
                Xv = torch.as_tensor(Xv, device=device)
                fea_emb, Xr = decoder_model(Xv)
                Z_fea, _, _ = Dual_attention_G(fea_emb, index_swap)
                Labels = Y[0]
                estimator = KMeans(n_clusters=cluster)
                fea_cluster = Z_fea.detach().cpu().numpy()
                estimator.fit(fea_cluster)
                pred_final = estimator.labels_

                acc, nmi, ari = evaluate(Labels, pred_final)


    return Z_fea, Buffer
