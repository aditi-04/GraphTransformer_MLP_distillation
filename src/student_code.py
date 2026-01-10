# from data import get_dataset

import time
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from early_stop import EarlyStopping, Stop_args
from model import TransformerModel
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
import argparse
import pickle
import torch.nn as nn
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import metrics
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt


class NORMAL_MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512 , dropout_ratio=0.3):
        super(NORMAL_MLP, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)        
        self.dropout = nn.Dropout(dropout_ratio)
        self.lin2 = nn.Linear(hidden_size,output_size)
        

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        mid_op = x.clone()
        x = self.lin2(x)
        
        return x, mid_op
        
class MLP_ATT_U(torch.nn.Module):
    def __init__(self, input_size, hidden_size=512, dropout_ratio=0.5, n_layers = 1, param_hops = 8, param_hidden_dim=64,param_n_heads= 8):
        super(MLP_ATT_U, self).__init__()
        self.param_hops = param_hops
        self.param_hidden_dim = param_hidden_dim
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size,param_n_heads*n_layers*(self.param_hops+1)*(self.param_hops+1))
        self.dropout = nn.Dropout(dropout_ratio)
        self.param_n_heads = param_n_heads

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        us = self.lin2(x).view(-1,self.param_n_heads, self.param_hops+1,self.param_hops+1)

       
        return us, 0
    
class MLP_ATT_V(torch.nn.Module):
    def __init__(self, input_size, hidden_size=512, dropout_ratio=0.5, n_layers = 1, param_hops = 8, param_hidden_dim=64,param_n_heads= 8):
        super(MLP_ATT_V, self).__init__()
        self.param_hops = param_hops
        self.param_hidden_dim = param_hidden_dim
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size,n_layers*(self.param_hops+1)*self.param_hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.param_n_heads  = param_n_heads

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        vt = self.lin2(x).view(-1,(self.param_hops+1),self.param_hidden_dim)
        
        return 0, vt
    
class MLP_ATT_UV(torch.nn.Module):
    def __init__(self, input_size, hidden_size=512, dropout_ratio=0.5, n_layers = 1, param_hops = 8, param_hidden_dim=64,param_n_heads= 8):
        super(MLP_ATT_UV, self).__init__()
        self.param_hops = param_hops
        self.param_hidden_dim = param_hidden_dim
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2_1 = nn.Linear(hidden_size,param_n_heads*n_layers*(self.param_hops+1)*(self.param_hops+1))
        self.lin2_2 = nn.Linear(hidden_size,n_layers*(self.param_hops+1)*self.param_hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.param_n_heads = param_n_heads

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
     
        us = self.lin2_1(x).view(-1,self.param_n_heads,self.param_hops+1,self.param_hops+1)
        vt = self.lin2_2(x).view(-1,(self.param_hops+1),self.param_hidden_dim)
       
        return us, vt

class MLP_ATT_GATED(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, dropout_ratio=0.3, n_layers=1, param_hops=4 , param_hidden_dim=126,param_n_heads=3):
        super(MLP_ATT_GATED, self).__init__()
        self.param_hops = param_hops
        self.param_hidden_dim = param_hidden_dim
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2_u = nn.Linear(hidden_size,param_n_heads*n_layers*(self.param_hops+1)*(self.param_hops+1))
        self.lin2_v = nn.Linear(hidden_size,n_layers*(self.param_hops+1)*self.param_hidden_dim)
        
        self.lin2_op_1 = nn.Linear(hidden_size, hidden_size)
        
        self.op_sz = hidden_size + param_n_heads*n_layers*(self.param_hops+1)*(self.param_hops+1) + n_layers*(self.param_hops+1)*self.param_hidden_dim
        self.lin2_op_2 = nn.Linear(self.op_sz , output_size)
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.param_n_heads = param_n_heads

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        op = self.lin2_op_1(x)
        op_inter = op.clone()
        op = F.relu(op)
        op = self.dropout(op)
        
        us = self.lin2_u(x)
        vt = self.lin2_v(x)
        concat_op = torch.cat((op, us, vt), dim=1)
        
        
        op = self.lin2_op_2(concat_op)
        us = us.view(-1,self.param_n_heads, self.param_hops+1,self.param_hops+1)
        vt = vt.view(-1, (self.param_hops+1),self.param_hidden_dim)
       
        return us, vt, op, op_inter

'''''
Normal MLP 

'''''
def train_process_normal(model,path, 
                         all_data_loader,
                         train_data_loader,
                         val_data_loader ,
                         dataset = 'cora',
                         exp_type = 'normal_mlp',
                         epoches = 1000,
                         lr = 1e-3,
                         weight_decay = 1e-5,
                         dropout_ratio = 0.5,
                         inp_shape = None,
                         op_shape = None,
                         if_sl=True,
                         if_rsd=False,
                         device='cpu',
                         lamb_kl = 0.3, 
                         lamb_rsd = 0.3): 
    
    
    base_path = path


    param_model_folder = base_path+f"Results/{dataset}/model_weights/"
    param_plots_folder = base_path+f"Results/{dataset}/plots/"
    param_output_folder = base_path+f"Results/{dataset}/output/"
    
    
    mlp_weights = param_model_folder+f"{exp_type}_weights.pth"
    mlp_output_txt_file = param_output_folder+f"{exp_type}_output.txt"



    mlp_train_loss = param_output_folder + f"{exp_type}_train_loss.pkl"
    mlp_val_loss = param_output_folder + f"{exp_type}_test_loss.pkl"

    if os.path.exists(mlp_output_txt_file):
        os.remove(mlp_output_txt_file)


    mlp = NORMAL_MLP(inp_shape, op_shape,dropout_ratio = dropout_ratio).to(device)
    mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr = lr, weight_decay = weight_decay)

    stopping_args = Stop_args(patience=50,max_epochs=epoches)
    early_stopping = EarlyStopping(mlp, **stopping_args)


    lr_scheduler = PolynomialDecayLR(
                    mlp_optimizer,
                    warmup_updates=200,
                    tot_updates=500,
                    lr=lr,
                    end_lr=lr*0.1,
                    power=1.0,
                )
    
    loss_train_lst = []
    loss_val_lst = []

    t_total = time.time()
   

    for epoch in range(0, epoches):
        start = time.time()
        loss=0
        loss_train_b, loss_val, acc_val = train_student_with_attention_normal(epoch, 
                                                                           model,
                                                                           mlp, 
                                                                           mlp_optimizer, 
                                                                           mlp_output_txt_file, 
                                                                           all_data_loader,
                                                                           train_data_loader, 
                                                                           val_data_loader,
                                                                           if_sl = if_sl, 
                                                                           if_rsd = if_rsd,
                                                                           device= device,
                                                                           lamb_kl =lamb_kl, 
                                                                           lamb_rsd = lamb_kl)


        loss_train_lst.append(loss_train_b)
        loss_val_lst.append(loss_val)
        lr_scheduler.step()


        if early_stopping.check([acc_val, loss_val], epoch):
            break
         
    print("Optimization Finished!")
    print("Train cost: {:.4f}s".format(time.time() - t_total))
    print('Loading {}th epoch'.format(early_stopping.best_epoch+1))

    with open(mlp_output_txt_file, 'a') as f:       
        print("Optimization Finished!", file = f)
        print("Train cost: {:.4f}s".format(time.time() - t_total), file = f)
        print('Loading {}th epoch'.format(early_stopping.best_epoch+1), file = f)

    mlp.load_state_dict(early_stopping.best_state)
    
    
    print("Saving MLP and losses")
    
    torch.save(mlp.state_dict(),mlp_weights)
    with open(mlp_train_loss, 'wb') as f:
        pickle.dump(loss_train_lst, f)

    with open(mlp_val_loss, 'wb') as f:
        pickle.dump(loss_val_lst, f)
        
    print("Saved MLP Successfully")
    
    return mlp


def train_student_with_attention_normal(epoch, 
                                        model,
                                        normal_mlp, 
                                        normal_mlp_optimizer, 
                                        output_txt_file, 
                                        all_data_loader,
                                        train_data_loader,
                                        val_data_loader, 
                                        if_sl = False, 
                                        if_rsd = False,
                                        device='gpu', 
                                        lamb_kl = 0.3, 
                                        lamb_rsd = 0.3):

    normal_mlp.train()
    loss_train_b = 0
    
    
    
    for _,item in enumerate(train_data_loader):
        loss_train_non_distill = 0
        nodes_features_mlp = item[2].to(device)
        nodes_features = item[0].to(device)
        labels = item[1].to(device)
        
        distill_output, distill_op_inter = normal_mlp(nodes_features_mlp)
        s_distill_op_inter = torch.mm(distill_op_inter, distill_op_inter.T)
        output = F.log_softmax(distill_output, dim=1)
        
        loss_ce = (1-lamb_kl-lamb_rsd)*F.nll_loss(output,labels)
        loss_train_non_distill += loss_ce
        

            
        normal_mlp_optimizer.zero_grad()
        loss_train_non_distill.backward()
        normal_mlp_optimizer.step()
            
        loss_train_b += loss_train_non_distill.item()
        
     
    if if_sl or if_rsd:
        
        for _,item in enumerate(all_data_loader):
            loss_train_distill = 0
            nodes_features_mlp = item[2].to(device)
            nodes_features = item[0].to(device)
            labels = item[1].to(device)

            gnn_output,gnn_feats, train_attentions = model(nodes_features) 
            s_gnn_feats = torch.mm(gnn_feats, gnn_feats.T)
            y_soft = gnn_output.log_softmax(dim=-1)


            distill_output, distill_op_inter = normal_mlp(nodes_features_mlp)
            s_distill_op_inter = torch.mm(distill_op_inter, distill_op_inter.T)
            output = F.log_softmax(distill_output, dim=1)
            
            if if_rsd:
                loss_rsd = 1 - F.cosine_similarity(s_gnn_feats.reshape(1, -1), s_distill_op_inter.reshape(1, -1))[0]
                lamb_rsd =  lamb_rsd*loss_rsd
                loss_train_non_distill += lamb_rsd
                
            if if_sl:
                loss_sl = F.kl_div(output, y_soft, reduction='batchmean',log_target=True) 
                loss_sl = lamb_kl*loss_sl
                loss_train_distill += loss_sl
            
            normal_mlp_optimizer.zero_grad()
            loss_train_distill.backward()
            normal_mlp_optimizer.step()

            loss_train_b += loss_train_distill.item()

        
        
    normal_mlp.eval()
    loss_val_b = 0
    acc_val_b = 0
    
    for _, item in enumerate(val_data_loader):
        nodes_features_mlp = item[2].to(device)
        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        gnn_output,gnn_feats, train_attentions = model(nodes_features) 
        s_gnn_feats = torch.mm(gnn_feats, gnn_feats.T)
        y_soft = gnn_output.log_softmax(dim=-1)
        
        
        distill_output, distill_op_inter = normal_mlp(nodes_features_mlp)
        s_distill_op_inter = torch.mm(distill_op_inter, distill_op_inter.T)
        output = F.log_softmax(distill_output, dim=1)
        
        
        loss_ce = F.nll_loss(output,labels)
        loss_val = (1-lamb_kl-lamb_rsd)*loss_ce
        
        if if_sl:
            loss_sl = F.kl_div(output, y_soft, reduction='batchmean',log_target=True) 

        
        if if_rsd:
            loss_rsd = 1 - F.cosine_similarity(s_gnn_feats.reshape(1, -1), s_distill_op_inter.reshape(1, -1))[0]
            loss_val += lamb_rsd*loss_rsd
        
        
        loss_val_b += loss_ce.item()
        acc_val_b +=  utils.accuracy_batch(distill_output, labels).item()
        
    print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train_b),
            'loss_val: {:.4f}'.format(loss_val_b),
            'acc_val: {:.4f}'.format(acc_val_b/len(val_data_loader.dataset)),)
        
    with open(output_txt_file, 'a') as f:
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train_b),
            'loss_val: {:.4f}'.format(loss_val_b),
            'acc_val: {:.4f}'.format(acc_val_b/len(val_data_loader.dataset)),file=f)
    
    return loss_train_b, loss_val_b, acc_val_b


@torch.no_grad()
def test_student_normal(model, normal_mlp, data_loader, output_txt_file,device='gpu'):
    start_time = time.time()
    
    normal_mlp.eval()
    accs = []
    tf_embeds=[]
    mlp_embeds=[]
    correct=0
    total=0
    loss_test = 0
    acc_test = 0
    all_labels =[]
    h20_scores = pd.DataFrame()
    for item in data_loader:
        nodes_features = item[2].to(device)
        nodes_features_gnn = item[0].to(device)
        labels = item[1].to(device)

        out, distill_op_inter = normal_mlp(nodes_features)
        gnn_output,gnn_feats_inter, train_attentions = model(nodes_features_gnn) 
        pred = out.argmax(dim=-1)
        
        tf_embeds.append(gnn_feats_inter)
        mlp_embeds.append(distill_op_inter)
        all_labels.append(labels)
        acc = torch.sum(pred == labels).item() / len(labels)
        correct += torch.sum(pred == labels).item()
        total +=  len(labels)
        accs.append(acc)
        acc_test += torch.sum(pred == labels).item() 
        
        output = F.log_softmax(out, dim=1)
        
        loss_test += F.nll_loss(output, labels).item()
        temp_h20_scores = pd.DataFrame()
        temp_h20_scores['label_3'] = np.array(labels.cpu().detach())
        temp_h20_scores['p1'] = np.array(output[:,1].cpu().detach())
        if( h20_scores.shape[0] == 0):
            h20_scores = temp_h20_scores
        else:
            h20_scores = pd.concat([h20_scores,temp_h20_scores])
        
    end_time = time.time()

    total_time = end_time - start_time

    h20_scores['p1'] = h20_scores['p1'].fillna(0)
    thre = h20_scores['p1'].sort_values(ascending=False).iloc[(h20_scores['label_3']==1).sum()]
    cr_h20_scores = pd.crosstab(h20_scores['label_3'], (h20_scores['p1']>thre).astype('int'))#[1,1]/
    print(cr_h20_scores)
    roc_auc =  average_precision_score(h20_scores['label_3'], h20_scores['p1']) #metrics.roc_auc_score(h20_scores['label_3'], h20_scores['p1'])
    print(f"Test AUC_PR: {roc_auc}")
    
    if(cr_h20_scores.shape[1] == 2 and cr_h20_scores.shape[0] == 2):
        cr = cr_h20_scores.iloc[1,1]/cr_h20_scores.iloc[1,0]
        print(f"Test CR: {cr}")
        thre = h20_scores['p1'].sort_values(ascending=False).iloc[ int(h20_scores['label_3'].shape[0]/100) ]
        cr_h20_scores = pd.crosstab(h20_scores['label_3'], (h20_scores['p1']>thre).astype('int'))
        print(f"Test Top 1%: {cr_h20_scores.iloc[1,1]/(cr_h20_scores.iloc[1,1] + cr_h20_scores.iloc[1,0])}" )

        thre = h20_scores['p1'].sort_values(ascending=False).iloc[ int(h20_scores['label_3'].shape[0]/20) ]
        cr_h20_scores = pd.crosstab(h20_scores['label_3'], (h20_scores['p1']>thre).astype('int'))
        print(f"Test Top 5%: {cr_h20_scores.iloc[1,1]/(cr_h20_scores.iloc[1,1] + cr_h20_scores.iloc[1,0])}")

        thre = h20_scores['p1'].sort_values(ascending=False).iloc[ int(h20_scores['label_3'].shape[0]/10) ]
        cr_h20_scores = pd.crosstab(h20_scores['label_3'], (h20_scores['p1']>thre).astype('int'))
        print(f"Test Top 10%: {cr_h20_scores.iloc[1,1]/(cr_h20_scores.iloc[1,1] + cr_h20_scores.iloc[1,0])}" )
        print(cr_h20_scores)
    
    print("Test set results:",
        "loss= {:.4f}".format(loss_test),
        "accuracy= {:.4f}".format(acc_test/len(data_loader.dataset)))
    
    with open(output_txt_file, 'a') as f:
        print("/n/n/n/n/n/", file=f)
        print("Test set results:",
        "loss= {:.4f}".format(loss_test),
        "accuracy= {:.4f}".format(acc_test/len(data_loader.dataset)), file=f)
        
        print(f"Total time taken in Inference is: {total_time} seconds for {len(data_loader.dataset)} records" , file = f)
        
    return  mlp_embeds, tf_embeds, all_labels

'''''
Dual MLP with us abd vt estimation
'''''

def train_process_dual(model,path, 
                       all_data_loader,
                       train_data_loader,
                       val_data_loader ,
                       dataset = 'cora',
                       exp_type = 'normal_mlp',
                       epoches = 1000,
                       lr = 1e-3,
                       weight_decay = 1e-5,
                       att_lr= 1e-5,
                       dropout_ratio = 0.5,
                       inp_shape = None,
                       op_shape = None,
                       param_hops= None,
                       param_hidden_dim = None,
                       param_n_layers=None,
                        param_n_heads = None,
                       if_u = True,
                       if_v = False, 
                       if_rsd = False,
                       if_full_att = False,
                       if_gated = False,
                       device='cpu',
                       lamb_kl = 0.2, 
                       lamb_rsd = 0.2,
                       lamb_u=0.2,
                       lamb_v = 0.2): 
    
    
    base_path = path


    param_model_folder = base_path+f"Results/{dataset}/model_weights/"
    param_plots_folder = base_path+f"Results/{dataset}/plots/"
    param_output_folder = base_path+f"Results/{dataset}/output/"
    
    
    mlp_weights = param_model_folder+f"{exp_type}_weights.pth"
    att_mlp_weights = param_model_folder+f"att_{exp_type}_weights.pth"
    
    
    mlp_output_txt_file = param_output_folder+f"{exp_type}_output.txt"



    mlp_train_loss = param_output_folder + f"{exp_type}_train_loss.pkl"
    mlp_val_loss = param_output_folder + f"{exp_type}_test_loss.pkl"

        
    inp_shape_us = inp_shape + param_n_heads*(param_hops+1)*(param_hops+1)*param_n_layers
    inp_shape_vt = inp_shape + (param_hops+1)*(param_hidden_dim)*param_n_layers
    inp_shape_us_vt = inp_shape + (param_hops+1)*(param_hidden_dim)*param_n_layers + param_n_heads*(param_hops+1)*(param_hops+1)*param_n_layers
    # print(inp_shape_vt)
    
    
    if if_gated:
        mlp = MLP_ATT_GATED(inp_shape, op_shape, n_layers = param_n_layers,param_hops = param_hops, param_hidden_dim=param_hidden_dim, dropout_ratio=dropout_ratio).to(device)
        att_mlp = None
    elif if_u and if_v:
        mlp = NORMAL_MLP(inp_shape_us_vt, op_shape, dropout_ratio=dropout_ratio).to(device)
        att_mlp = MLP_ATT_UV(inp_shape, n_layers = param_n_layers,param_hops = param_hops, param_hidden_dim=param_hidden_dim, param_n_heads= param_n_heads,dropout_ratio=dropout_ratio).to(device)
    else:
        if if_u:
            mlp = NORMAL_MLP(inp_shape_us, op_shape, dropout_ratio=dropout_ratio).to(device)
            att_mlp = MLP_ATT_U(inp_shape, n_layers = param_n_layers,param_hops = param_hops, param_hidden_dim=param_hidden_dim, param_n_heads= param_n_heads,dropout_ratio=dropout_ratio).to(device)      
        if if_v:
            mlp = NORMAL_MLP(inp_shape_vt, op_shape, dropout_ratio=dropout_ratio).to(device)
            att_mlp = MLP_ATT_V(inp_shape, n_layers = param_n_layers,param_hops = param_hops, param_hidden_dim=param_hidden_dim,param_n_heads= param_n_heads,dropout_ratio=dropout_ratio).to(device)
        if if_full_att:
            mlp = NORMAL_MLP(inp_shape_vt, op_shape, dropout_ratio=dropout_ratio).to(device)
            att_mlp = MLP_ATT_V(inp_shape, n_layers = param_n_layers,param_hops = param_hops, param_hidden_dim=param_hidden_dim,param_n_heads= param_n_heads,dropout_ratio=dropout_ratio).to(device)

    
    
    mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr = lr, weight_decay = weight_decay)

    stopping_args = Stop_args(patience=50,max_epochs=epoches)
    early_stopping = EarlyStopping(mlp, **stopping_args)
    
    if if_gated == False:    
        att_mlp_optimizer = torch.optim.Adam(att_mlp.parameters(), lr = att_lr, weight_decay = weight_decay)

        early_stopping_att = EarlyStopping(att_mlp, **stopping_args)
        
        att_lr_scheduler = PolynomialDecayLR(
                    att_mlp_optimizer,
                    warmup_updates=200,
                    tot_updates=500,
                    lr=att_lr,
                    end_lr=att_lr*0.01,
                    power=1.0,
                )
    else:
        att_mlp_optimizer = None
        early_stopping_att = None
        att_lr_scheduler = None
        

    
    lr_scheduler = PolynomialDecayLR(
                    mlp_optimizer,
                    warmup_updates=200,
                    tot_updates=500,
                    lr=lr,
                    end_lr=lr*0.1,
                    power=1.0,
                )
    
    
    
    loss_train_lst = []
    loss_val_lst = []

    t_total = time.time()
    

    for epoch in range(0, epoches):
        start = time.time()
        loss=0
        loss_train_b, loss_val, acc_val, loss_att = train_student_with_attention_dual(epoch,
                                                                                                   model,
                                                                                                   mlp, 
                                                                                                   att_mlp , 
                                                                                                   mlp_optimizer, 
                                                                                                   att_mlp_optimizer, 
                                                                                                   mlp_output_txt_file, 
                                                                                                   all_data_loader,
                                                                                                   train_data_loader,
                                                                                                   val_data_loader,
                                                                                                   if_u = if_u,
                                                                                                   if_v = if_v, 
                                                                                                   if_rsd = if_rsd,
                                                                                                   if_full_att = if_full_att,
                                                                                                   if_gated = if_gated,
                                                                                                   device= device,
                                                                                                   lamb_kl = lamb_kl, 
                                                                                                   lamb_rsd =lamb_rsd,
                                                                                                   lamb_u=lamb_u,
                                                                                                   lamb_v = lamb_v)
    
        lr_scheduler.step()
        if att_lr_scheduler:
            att_lr_scheduler.step()
        
        

        loss_train_lst.append(loss_train_b)
        loss_val_lst.append(loss_val)
        
        if early_stopping_att:
            early_stopping_att.check([acc_val, loss_val], epoch)
        if early_stopping.check([acc_val, loss_val], epoch) :
             break
                
    print("Optimization Finished!")
    print("Train cost: {:.4f}s".format(time.time() - t_total))
    print('Loading {}th epoch'.format(early_stopping.best_epoch+1))  

    with open(mlp_output_txt_file, 'a') as f:       
        print("Optimization Finished!", file = f)
        print("Train cost: {:.4f}s".format(time.time() - t_total), file = f)
        print('Loading {}th epoch'.format(early_stopping.best_epoch+1), file = f)

    mlp.load_state_dict(early_stopping.best_state)
    if att_mlp:
        att_mlp.load_state_dict(early_stopping_att.best_state)
    
    print("Saving MLP and losses")
    
    torch.save(mlp.state_dict(),mlp_weights)
    if att_mlp:
        print(" Not Saving Att MLP")
        torch.save(att_mlp.state_dict(),att_mlp_weights)
    
    with open(mlp_train_loss, 'wb') as f:
        pickle.dump(loss_train_lst, f)

    with open(mlp_val_loss, 'wb') as f:
        pickle.dump(loss_val_lst, f)
        
    print("Saved MLP Successfully")
    
    return mlp, att_mlp


def train_student_with_attention_dual(epoch, 
                                      model,
                                      normal_mlp, 
                                      att_mlp , 
                                      normal_mlp_optimizer,
                                      att_mlp_optimizer,
                                      output_txt_file, 
                                      all_data_loader,
                                      train_data_loader, 
                                      val_data_loader,
                                      if_u = False, 
                                      if_v = False, 
                                      if_rsd = False,
                                      if_full_att = False,
                                      if_gated = False,
                                      device='gpu',
                                      lamb_kl = 0.2, 
                                      lamb_rsd = 0.2,
                                      lamb_u=0.2,
                                      lamb_v = 0.2):

    normal_mlp.train()
    normal_mlp_optimizer.zero_grad()
    
    if att_mlp:
        att_mlp.train()
    if att_mlp_optimizer:
        att_mlp_optimizer.zero_grad()
    model.eval()
    
    loss_train_b = 0
    
    
    

        
     
    if if_rsd or if_u or if_v or if_full_att or if_gated:
        
        for _,item in enumerate(all_data_loader):
            loss_train_distill = 0
            nodes_features_mlp = item[2].to(device)
            nodes_features = item[0].to(device)
            labels = item[1].to(device)

            gnn_output,gnn_feats, train_attentions = model(nodes_features) 
            s_gnn_feats = torch.mm(gnn_feats, gnn_feats.T)
            y_soft = gnn_output.log_softmax(dim=-1)
            
            
            US_list, Vt_list, ATT_lst = [], [], []
            for tensor in train_attentions:
                numpy_array = tensor.detach().cpu().numpy()  
                U, S, Vt = np.linalg.svd(numpy_array,full_matrices=False)
                US = (U * S[..., None, :])
                US_list.append(US)
                Vt_list.append(Vt)
                ATT_lst.append(US@Vt)

            US_tensor_list = [torch.tensor(array) for array in US_list]
            US_tensor = torch.cat(US_tensor_list, dim=0).to(device)
            Vt_tensor_list = [torch.tensor(array) for array in Vt_list]
            Vt_tensor= torch.cat(Vt_tensor_list,dim=0).to(device)
            # print('hiii')
            # print(Vt_tensor.shape)
            # print(US_tensor.shape)

            Att_tensor_list = [torch.tensor(array) for array in ATT_lst]
            Att_tensor= torch.cat(Att_tensor_list,dim=0).to(device)
            
            if att_mlp:
                us, vt = att_mlp(nodes_features_mlp)



            US_tensor = US_tensor.view(US_tensor.shape[0], -1)
            Vt_tensor = Vt_tensor.view(US_tensor.shape[0], -1)
            Att_tensor = Att_tensor.view(US_tensor.shape[0], -1)
            # print(US_tensor.shape, Vt_tensor.shape)

            
            att_mat = None
            
                
            if if_full_att:
                att_mat = vt
                att_mat = att_mat.view(att_mat.shape[0], -1)
                nodes_features_mlp  = torch.cat((nodes_features_mlp, att_mat),dim=1) 
            
            if if_u:
                us = us.view(us.shape[0], -1)
                nodes_features_mlp  = torch.cat((nodes_features_mlp, us),dim=1)
            if if_v:
                vt = vt.view(vt.shape[0], -1)
                nodes_features_mlp  = torch.cat((nodes_features_mlp, vt),dim=1)

            # print(us.shape, vt.shape)
            if if_gated:
                # print("here")
                # print(normal_mlp)
                us, vt, distill_output, distill_op_inter = normal_mlp(nodes_features_mlp)
                # print(us.shape, US_tensor.shape)
                # print(vt.shape, Vt_tensor.shape)
                
            else:    
                distill_output, distill_op_inter = normal_mlp(nodes_features_mlp)
                
                
            if if_full_att:
                loss_att = 1 - F.cosine_similarity(Att_tensor, att_mat,dim=1).mean()  
                loss_att = (lamb_u+lamb_v)*loss_att
                loss_train_distill += loss_att
            
            if if_u:
                loss_us = 1 - F.cosine_similarity(US_tensor, us,dim=1).mean()
                loss_us = lamb_u*loss_us
                loss_train_distill += loss_us
            if if_v:
                loss_vt = 1 - F.cosine_similarity(Vt_tensor, vt,dim=1).mean()
                loss_vt = lamb_v*loss_vt
                loss_train_distill += loss_vt
                
            if if_gated:
                # print("here2")
                # print(us.shape, US_tensor.shape)
                # print(US_tensor.shape, Vt_tensor.shape)
                us = us.view(us.shape[0], -1)
                loss_us = 1 - F.cosine_similarity(US_tensor, us,dim=1).mean()
                loss_us = lamb_u*loss_us
                loss_train_distill += loss_us
                vt = vt.view(vt.shape[0], -1)
                
                loss_vt = 1 - F.cosine_similarity(Vt_tensor, vt,dim=1).mean()
                loss_vt = lamb_v*loss_vt
                loss_train_distill += loss_vt
            
            
            s_distill_op_inter = torch.mm(distill_op_inter, distill_op_inter.T)
            output = F.log_softmax(distill_output, dim=1)

            if if_rsd:
                loss_rsd = 1 - F.cosine_similarity(s_gnn_feats.reshape(1, -1), s_distill_op_inter.reshape(1, -1))[0]
                loss_rsd =  lamb_rsd*loss_rsd
    
                loss_train_distill += loss_rsd
                
            
            loss_sl = F.kl_div(output, y_soft, reduction='batchmean',log_target=True) 
            loss_sl = lamb_kl*loss_sl
            loss_train_distill += loss_sl
        
                
                    
           
            normal_mlp_optimizer.zero_grad()
            if att_mlp_optimizer:
                att_mlp_optimizer.zero_grad()
            loss_train_distill.backward()
            normal_mlp_optimizer.step()
            if att_mlp_optimizer:
                att_mlp_optimizer.step()
            
            loss_train_b += loss_train_distill.item()
            
            
    for _,item in enumerate(train_data_loader):
        loss_train_non_distill = 0
        nodes_features_mlp = item[2].to(device)
        nodes_features = item[0].to(device)
        labels = item[1].to(device)
        
        if att_mlp:
            us, vt = att_mlp(nodes_features_mlp)
        
        if if_full_att:
            att_mat = vt
            att_mat = att_mat.view(att_mat.shape[0], -1)
            nodes_features_mlp  = torch.cat((nodes_features_mlp, att_mat),dim=1) 

        if if_u:
            us = us.view(us.shape[0], -1)
            nodes_features_mlp  = torch.cat((nodes_features_mlp, us),dim=1)
        if if_v:
            vt = vt.view(vt.shape[0], -1)
            nodes_features_mlp  = torch.cat((nodes_features_mlp, vt),dim=1)

       
        
        if if_gated:
                us, vt, distill_output, distill_op_inter = normal_mlp(nodes_features_mlp)
        else:    
            distill_output, distill_op_inter = normal_mlp(nodes_features_mlp)

        s_distill_op_inter = torch.mm(distill_op_inter, distill_op_inter.T)
        output = F.log_softmax(distill_output, dim=1)
        
        loss_ce = (1-lamb_kl-lamb_rsd)*F.nll_loss(output,labels)
        loss_train_non_distill += loss_ce
        

            
        normal_mlp_optimizer.zero_grad()
        loss_train_non_distill.backward()
        normal_mlp_optimizer.step()
            
        loss_train_b += loss_train_non_distill.item()
    
       
  
    normal_mlp.eval()
    if att_mlp:
        att_mlp.eval()
    
    loss_val_b = 0
    acc_val_b = 0
    loss_att_b = 0
    
    for _, item in enumerate(val_data_loader):
        nodes_features_mlp = item[2].to(device)
        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        gnn_output,gnn_feats, val_attentions = model(nodes_features) 
        s_gnn_feats = torch.mm(gnn_feats, gnn_feats.T)
        y_soft = gnn_output.log_softmax(dim=-1)
        
        
        US_list, Vt_list, ATT_lst = [], [], []
        for tensor in val_attentions:
            numpy_array = tensor.detach().cpu().numpy()  
            U, S, Vt = np.linalg.svd(numpy_array,full_matrices=False)
            US = (U * S[..., None, :])
            US_list.append(US)
            Vt_list.append(Vt)
            ATT_lst.append(US@Vt)

        US_tensor_list = [torch.tensor(array) for array in US_list]
        US_tensor = torch.cat(US_tensor_list, dim=0).to(device)
        Vt_tensor_list = [torch.tensor(array) for array in Vt_list]
        Vt_tensor= torch.cat(Vt_tensor_list,dim=0).to(device)
        
        Att_tensor_list = [torch.tensor(array) for array in ATT_lst]
        Att_tensor= torch.cat(Att_tensor_list,dim=0).to(device)        
        Att_tensor = Att_tensor.view(Att_tensor.shape[0], -1)
        
        US_tensor = US_tensor.view(US_tensor.shape[0], -1)
        Vt_tensor = Vt_tensor.view(US_tensor.shape[0], -1)
        
        if att_mlp:
            us, vt = att_mlp(nodes_features_mlp)
        
        att_mat = None
        if if_full_att:
            att_mat = vt
            att_mat = att_mat.view(att_mat.shape[0], -1)
            nodes_features_mlp  = torch.cat((nodes_features_mlp,att_mat),dim=1)
        if if_u:
            us = us.view(us.shape[0], -1)
            nodes_features_mlp  = torch.cat((nodes_features_mlp, us),dim=1)
        if if_v:
            vt = vt.view(vt.shape[0], -1)
            nodes_features_mlp  = torch.cat((nodes_features_mlp, vt),dim=1)

        

        if if_gated:
                us, vt, distill_output, distill_op_inter = normal_mlp(nodes_features_mlp)
        else:    
            distill_output, distill_op_inter = normal_mlp(nodes_features_mlp)
                
        s_distill_op_inter = torch.mm(distill_op_inter, distill_op_inter.T)
        output = F.log_softmax(distill_output, dim=1)
        
        
        loss_ce = F.nll_loss(output,labels)
        loss_val = loss_ce
        
        loss_sl = F.kl_div(output, y_soft, reduction='batchmean',log_target=True) 
        loss_val += lamb_kl*loss_sl

            
        s_gnn_feats = s_gnn_feats.view(s_gnn_feats.shape[0], -1)
        s_distill_op_inter = s_distill_op_inter.view(s_distill_op_inter.shape[0], -1)
        
        
        
        if if_rsd:
            loss_rsd = (1 - F.cosine_similarity(s_gnn_feats, s_distill_op_inter,dim=1).mean())
            loss_val += lamb_rsd*loss_rsd
            
            
        if if_u:
            loss_us = 1 - F.cosine_similarity(US_tensor.reshape(1, -1), us.reshape(1, -1))[0]
            loss_val += lamb_u*loss_us
            loss_att_b+=loss_us
            
        if if_v:     
            loss_vt = 1 - F.cosine_similarity(Vt_tensor.reshape(1, -1), vt.reshape(1, -1))[0]
            loss_val += lamb_v*loss_vt
            loss_att_b+=loss_vt
            
        if if_full_att:
            loss_att = 1 - F.cosine_similarity(Att_tensor, att_mat,dim=1).mean()             
            loss_val += (lamb_u+lamb_v)*loss_att            
            loss_att_b += loss_att
            
        if if_gated:
            us = us.view(us.shape[0], -1)
            loss_us = 1 - F.cosine_similarity(US_tensor, us,dim=1).mean()
            loss_us = lamb_u*loss_us
            loss_att_b += loss_us
            
            vt = vt.view(vt.shape[0], -1)
            loss_vt = 1 - F.cosine_similarity(Vt_tensor, vt,dim=1).mean()
            loss_vt = lamb_v*loss_vt
            loss_att_b += loss_vt

        loss_val_b += loss_val.item()
        acc_val_b +=  utils.accuracy_batch(distill_output, labels).item()
        loss_att_b = loss_att_b.item()
        
        
    print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train_b),
            'loss_val: {:.4f}'.format(loss_val_b),
            'acc_val: {:.4f}'.format(acc_val_b/len(val_data_loader.dataset)),'loss_val_att :{:4f}'.format(loss_att_b))
        
    with open(output_txt_file, 'a') as f:
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train_b),
            'loss_val: {:.4f}'.format(loss_val_b),
            'acc_val: {:.4f}'.format(acc_val_b/len(val_data_loader.dataset)),'loss_val_att :{:4f}'.format(loss_att_b),file=f)
    
    return loss_train_b, loss_val_b, acc_val_b, loss_att_b

@torch.no_grad()
def test_student_dual(model, normal_mlp, att_mlp, data_loader, output_txt_file, if_u = False, if_v = False, if_rsd = False,if_full_att=False, if_gated = False,device='gpu'):
    start_time = time.time()
    
    normal_mlp.eval()
    if att_mlp:
        att_mlp.eval()
    
    accs = []
    tf_embeds=[]
    mlp_embeds=[]
    all_labels=[]
    correct=0
    total=0
    loss_test = 0
    acc_test = 0
    h20_scores = pd.DataFrame()
    
    for item in data_loader:
        nodes_features = item[2].to(device)
        nodes_features_gnn = item[0].to(device)
        labels = item[1].to(device)
        gnn_output,gnn_feats_inter, val_attentions = model(nodes_features_gnn) 
        tf_embeds.append(gnn_feats_inter)
        if att_mlp:
            us, vt = att_mlp(nodes_features)
            
        if if_full_att:
            att_mat = vt
            nodes_features  = torch.cat((nodes_features, att_mat.view(att_mat.shape[0],-1)),dim=1)
        else:        
            if if_u:
                nodes_features  = torch.cat((nodes_features, us.view(nodes_features.shape[0],-1)),dim=1)
            if if_v:
                nodes_features  = torch.cat((nodes_features, vt.view(nodes_features.shape[0],-1)),dim=1)
                
        if if_gated:
            us, vt, out, distill_op_inter = normal_mlp(nodes_features)
        else:    
            out, distill_op_inter = normal_mlp(nodes_features)
       
        mlp_embeds.append(distill_op_inter)
        pred = out.argmax(dim=-1)
        all_labels.append(labels)
        acc = torch.sum(pred == labels).item() / len(labels)
        correct += torch.sum(pred == labels).item()
        total +=  len(labels)
        accs.append(acc)
        
        acc_test += torch.sum(pred == labels).item()
        output = F.log_softmax(out, dim=1)
        loss_test += F.nll_loss(output, labels).item()
        temp_h20_scores = pd.DataFrame()
        temp_h20_scores['label_3'] = np.array(labels.cpu().detach())
        temp_h20_scores['p1'] = np.array(output[:,1].cpu().detach())
        if( h20_scores.shape[0] == 0):
            h20_scores = temp_h20_scores
        else:
            h20_scores = pd.concat([h20_scores,temp_h20_scores])
        
    end_time = time.time()
    total_time = end_time - start_time

    # acc = correct / total
    h20_scores['p1'] = h20_scores['p1'].fillna(0)
    thre = h20_scores['p1'].sort_values(ascending=False).iloc[(h20_scores['label_3']==1).sum()]
    cr_h20_scores = pd.crosstab(h20_scores['label_3'], (h20_scores['p1']>thre).astype('int'))#[1,1]/
    print(cr_h20_scores)
    roc_auc =  average_precision_score(h20_scores['label_3'], h20_scores['p1']) #metrics.roc_auc_score(h20_scores['label_3'], h20_scores['p1'])
    print(f"Test AUC_PR: {roc_auc}")
    
    if(cr_h20_scores.shape[1] == 2 and cr_h20_scores.shape[0] == 2):
        cr = cr_h20_scores.iloc[1,1]/cr_h20_scores.iloc[1,0]
        print(f"Test CR: {cr}")
        thre = h20_scores['p1'].sort_values(ascending=False).iloc[ int(h20_scores['label_3'].shape[0]/100) ]
        cr_h20_scores = pd.crosstab(h20_scores['label_3'], (h20_scores['p1']>thre).astype('int'))
        print(f"Test Top 1%: {cr_h20_scores.iloc[1,1]/(cr_h20_scores.iloc[1,1] + cr_h20_scores.iloc[1,0])}" )

        thre = h20_scores['p1'].sort_values(ascending=False).iloc[ int(h20_scores['label_3'].shape[0]/20) ]
        cr_h20_scores = pd.crosstab(h20_scores['label_3'], (h20_scores['p1']>thre).astype('int'))
        print(f"Test Top 5%: {cr_h20_scores.iloc[1,1]/(cr_h20_scores.iloc[1,1] + cr_h20_scores.iloc[1,0])}")

        thre = h20_scores['p1'].sort_values(ascending=False).iloc[ int(h20_scores['label_3'].shape[0]/10) ]
        cr_h20_scores = pd.crosstab(h20_scores['label_3'], (h20_scores['p1']>thre).astype('int'))
        print(f"Test Top 10%: {cr_h20_scores.iloc[1,1]/(cr_h20_scores.iloc[1,1] + cr_h20_scores.iloc[1,0])}" )
        print(cr_h20_scores)
    
    print("Test set results:",
        "loss= {:.4f}".format(loss_test),
        "accuracy= {:.4f}".format(acc_test/len(data_loader.dataset)))
    
    with open(output_txt_file, 'a') as f:
        print("/n/n/n/n/n/", file=f)
        print("Test set results:",
        "loss= {:.4f}".format(loss_test),
        "accuracy= {:.4f}".format(acc_test/len(data_loader.dataset)), file=f)
        
        print(f"Total time taken in Inference is: {total_time} seconds for {len(data_loader.dataset)} records", file = f)
        
    return mlp_embeds, tf_embeds, all_labels

