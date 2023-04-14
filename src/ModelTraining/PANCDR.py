import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import pandas as pd
import itertools
israndom=False
from itertools import cycle
from sklearn.model_selection import StratifiedKFold,KFold

from ModelTraining.model import Encoder, GCN, ADV
device = torch.device('cuda')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class train_PANCDR():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def train(self,params,weight_path='../checkpoint/model.pt'):
        nz,d_dim,lr,lr_adv,lam,batch_size = params.values()

        X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,t_gexpr_feature = self.train_data
        TX_drug_feat_data_test,TX_drug_adj_data_test,TX_gexpr_data_test,TY_test = self.test_data
        X_t_train, X_t_val = train_test_split(t_gexpr_feature.T.values, test_size=0.05, random_state=0)
        X_drug_feat_data_train,X_drug_feat_data_val,X_drug_adj_data_train,X_drug_adj_data_val,X_gexpr_data_train,X_gexpr_data_val,Y_train,Y_val= train_test_split(X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,test_size=0.05, random_state=0)
        
        X_drug_feat_train = torch.FloatTensor(X_drug_feat_data_train)
        X_drug_adj_train = torch.FloatTensor(X_drug_adj_data_train)
        X_gexpr_train = torch.FloatTensor(X_gexpr_data_train)
        X_t_gexpr_train = torch.FloatTensor(X_t_train)
        Y_train = torch.FloatTensor(Y_train)

        X_drug_feat_val = torch.FloatTensor(X_drug_feat_data_val).to(device)
        X_drug_adj_val = torch.FloatTensor(X_drug_adj_data_val).to(device)
        X_gexpr_val = torch.FloatTensor(X_gexpr_data_val).to(device)
        X_t_gexpr_val = torch.FloatTensor(X_t_val).to(device)
        Y_val = torch.FloatTensor(Y_val).to(device)
        
        GDSC_Dataset = torch.utils.data.TensorDataset(X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train)
        GDSC_Loader = torch.utils.data.DataLoader(dataset=GDSC_Dataset, batch_size = batch_size[0], shuffle=True, drop_last=True)
        E_TEST_Dataset = torch.utils.data.TensorDataset(X_t_gexpr_train)
        E_TEST_Loader = torch.utils.data.DataLoader(dataset=E_TEST_Dataset, batch_size = batch_size[1], shuffle=True, drop_last=True)

        wait, best_auc = 0, 0
        EN_model = Encoder(X_gexpr_train.shape[1], nz, device)
        GCN_model = GCN(X_drug_feat_train.shape[2],[256,256,256],h_dims=[d_dim, nz+d_dim],use_dropout=
        False)
        ADV_model = ADV(nz)
        EN_model.to(device)
        GCN_model.to(device)
        ADV_model.to(device)

        optimizer = torch.optim.Adam(itertools.chain(EN_model.parameters(),GCN_model.parameters()), lr=lr)
        optimizer_adv = torch.optim.Adam(ADV_model.parameters(), lr=lr_adv)
        loss = torch.nn.BCELoss()

        for epoch in range(1000):
            for i,data in enumerate(zip(GDSC_Loader, cycle(E_TEST_Loader))):

                DataG = data[0]
                t_gexpr = data[1][0]
                drug_feat, drug_adj, gexpr, y_true = DataG
                drug_feat = drug_feat.to(device)
                drug_adj = drug_adj.to(device)
                gexpr = gexpr.to(device)
                y_true = y_true.view(-1,1).to(device)
                t_gexpr = t_gexpr.to(device)
                EN_model.train()
                GCN_model.train()
                ADV_model.train()

                optimizer_adv.zero_grad()
                F_gexpr,_,_ = EN_model(gexpr)
                F_t_gexpr,_,_ = EN_model(t_gexpr)

                F_g_t_gexpr = torch.cat((F_gexpr,F_t_gexpr))
                z_true = torch.cat((torch.zeros(F_gexpr.shape[0], device=device), torch.ones(F_t_gexpr.shape[0], device=device)))
                z_true = z_true.view(-1,1)
                z_pred = ADV_model(F_g_t_gexpr)
                if IsNaN(z_pred): return -1
                adv_loss = loss(z_pred, z_true)
                adv_loss.backward()
                optimizer_adv.step()

                optimizer.zero_grad()

                g_latents, _, _ = EN_model(gexpr)
                t_latents, _, _ = EN_model(t_gexpr)

                F_g_t_latents = torch.cat((g_latents,t_latents))
                z_true_ = torch.cat((torch.ones(g_latents.shape[0], device=device), torch.zeros(t_latents.shape[0], device=device)))
                z_true_ = z_true_.view(-1,1)
                z_pred_ = ADV_model(F_g_t_latents)
                y_pred = GCN_model(drug_feat,drug_adj,g_latents)
                if IsNaN(z_pred_) or IsNaN(y_pred): return -1
                adv_loss_ = loss(z_pred_, z_true_)
                cdr_loss = loss(y_pred, y_true)

                Loss = cdr_loss + lam*adv_loss_
                Loss.backward()
                optimizer.step()

            with torch.no_grad():
                EN_model.eval()
                GCN_model.eval()
                ADV_model.eval()

                F_gexpr_val,_,_ = EN_model(X_gexpr_val)
                F_t_gexpr_val,_,_ = EN_model(X_t_gexpr_val)

                F_g_t_gexpr_val = torch.cat((F_gexpr_val, F_t_gexpr_val))
                z_pred_val = ADV_model(F_g_t_gexpr_val)
                y_pred_val = GCN_model(X_drug_feat_val, X_drug_adj_val, F_gexpr_val)
                loss_val = loss(y_pred_val, Y_val.view(-1,1)) + lam*loss(z_pred_val, torch.ones(z_pred_val.shape, device=device))
                auc_val = roc_auc_score(Y_val.cpu().detach().numpy(), y_pred_val.cpu().detach().numpy())
                
                F_TEST_gexpr,_,_ = EN_model(TX_gexpr_data_test)
                y_pred_TEST = GCN_model(TX_drug_feat_data_test, TX_drug_adj_data_test, F_TEST_gexpr)
                auc_TEST = roc_auc_score(TY_test.cpu().detach().numpy(), y_pred_TEST.cpu().detach().numpy())
                
            if auc_val >= best_auc:
                wait = 0
                best_auc = auc_val
                best_auc_TEST = auc_TEST
                torch.save({'EN_model': EN_model.state_dict(), 'GCN_model':GCN_model.state_dict(), 
                            'ADV_model':ADV_model.state_dict()}, weight_path)
                
            else:
                wait += 1
                if wait >= 10: break
        
        return best_auc_TEST

    def predict(self, data, params, weight_path):
        nz,d_dim,_,_,_,_ = params.values()
        drug_feat, drug_adj, gexpr = data

        EN_model = Encoder(gexpr.shape[1], nz, device)
        GCN_model = GCN(drug_feat.shape[2],[256,256,256],h_dims=[d_dim, nz+d_dim],use_dropout=
        False)

        checkpoint = torch.load(weight_path)
        EN_model.load_state_dict(checkpoint['EN_model'])
        GCN_model.load_state_dict(checkpoint['GCN_model'])
        EN_model.to(device)
        GCN_model.to(device)

        with torch.no_grad():
            EN_model.eval()
            GCN_model.eval()

            F_gexpr = EN_model(gexpr)[0]
            y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)

        return y_pred.cpu()

def train_PANCDR_nested(n_outer_splits,data,best_params_file):
    X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,t_gexpr_feature = data
    outer_splits = StratifiedKFold(n_splits=n_outer_splits,shuffle=True,random_state=0)
    outer_folds = outer_splits.split(X_drug_feat_data, Y)
    auc_test_df = pd.DataFrame(columns=['Test_AUC','Best_params'])
    best_params_df = pd.read_csv(best_params_file, index_col=0)
    for outer_fold,(idx,test_idx) in enumerate(outer_folds):
        X_drug_feat_data_ = X_drug_feat_data[idx]
        X_drug_adj_data_ = X_drug_adj_data[idx]
        X_gexpr_data_ = X_gexpr_data[idx]
        Y_ = Y[idx]
        best_params = eval(best_params_df.loc["Fold_%d"%outer_fold,"Best_params"])

        X_drug_feat_data_test = X_drug_feat_data[test_idx]
        X_drug_adj_data_test = X_drug_adj_data[test_idx]
        X_gexpr_data_test = X_gexpr_data[test_idx]
        Y_test = Y[test_idx]

        X_drug_feat_test = torch.FloatTensor(X_drug_feat_data_test).to(device)
        X_drug_adj_test = torch.FloatTensor(X_drug_adj_data_test).to(device)
        X_gexpr_test = torch.FloatTensor(X_gexpr_data_test).to(device)
        Y_test = torch.FloatTensor(Y_test).to(device)

        train_data = [X_drug_feat_data_,X_drug_adj_data_,X_gexpr_data_,Y_,t_gexpr_feature]
        test_data = [X_drug_feat_test,X_drug_adj_test,X_gexpr_test,Y_test]
        model = train_PANCDR(train_data, test_data)
        while True:
            auc_TEST = model.train(best_params, weight_path='../checkpoint/kfold/model_best_outerfold_%d.pt'%outer_fold)
            if auc_TEST != -1: break
        temp_test_df = pd.DataFrame([[auc_TEST,best_params]], index=['Fold_%d'%outer_fold], columns=['Test_AUC','Best_params'])
        auc_test_df = pd.concat([auc_test_df,temp_test_df])

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    print('Mean test AUC - %.4f\n'%auc_test_df['Test_AUC'].mean())
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    auc_test_df = pd.concat([auc_test_df, pd.DataFrame(auc_test_df['Test_AUC'].mean(), index=['mean'], columns=['Test_AUC'])])
    return auc_test_df

class train_PANCDR_regr():
    def __init__(self,train_data,test_data):
        self.train_data = train_data
        self.test_data = test_data
    def train(self,params,weight_path='../checkpoint/kfold/model.pt'):
        nz,d_dim,lr,lr_adv,lam,batch_size = params.values()

        X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,t_gexpr_feature = self.train_data
        TX_drug_feat_data_test,TX_drug_adj_data_test,TX_gexpr_data_test,TY_test = self.test_data
        X_t_train, X_t_val = train_test_split(t_gexpr_feature.T.values, test_size=0.05, random_state=0)
        X_drug_feat_data_train,X_drug_feat_data_val,X_drug_adj_data_train,X_drug_adj_data_val,X_gexpr_data_train,X_gexpr_data_val,Y_train,Y_val= train_test_split(X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,test_size=0.05, random_state=0)
        
        X_drug_feat_train = torch.FloatTensor(X_drug_feat_data_train)
        X_drug_adj_train = torch.FloatTensor(X_drug_adj_data_train)
        X_gexpr_train = torch.FloatTensor(X_gexpr_data_train)
        X_t_gexpr_train = torch.FloatTensor(X_t_train)
        Y_train = torch.FloatTensor(Y_train)

        X_drug_feat_val = torch.FloatTensor(X_drug_feat_data_val).to(device)
        X_drug_adj_val = torch.FloatTensor(X_drug_adj_data_val).to(device)
        X_gexpr_val = torch.FloatTensor(X_gexpr_data_val).to(device)
        X_t_gexpr_val = torch.FloatTensor(X_t_val).to(device)
        Y_val = torch.FloatTensor(Y_val).to(device)
        
        GDSC_Dataset = torch.utils.data.TensorDataset(X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train)
        GDSC_Loader = torch.utils.data.DataLoader(dataset=GDSC_Dataset, batch_size = batch_size[0], shuffle=True, drop_last=True)
        E_TEST_Dataset = torch.utils.data.TensorDataset(X_t_gexpr_train)
        E_TEST_Loader = torch.utils.data.DataLoader(dataset=E_TEST_Dataset, batch_size = batch_size[1], shuffle=True, drop_last=True)

        wait, best_p = 0, 0
        EN_model = Encoder(X_gexpr_train.shape[1], nz, device)
        GCN_model = GCN(X_drug_feat_train.shape[2],[256,256,256],h_dims=[d_dim, nz+d_dim],use_dropout=False,is_regr=True)
        ADV_model = ADV(nz)
        EN_model.to(device)
        GCN_model.to(device)
        ADV_model.to(device)

        optimizer = torch.optim.Adam(itertools.chain(EN_model.parameters(),GCN_model.parameters()), lr=lr)
        optimizer_adv = torch.optim.Adam(ADV_model.parameters(), lr=lr_adv)
        criterion = torch.nn.MSELoss()
        loss = torch.nn.BCELoss()

        for epoch in range(1000):
            for i,data in enumerate(zip(GDSC_Loader, cycle(E_TEST_Loader))):

                DataG = data[0]
                t_gexpr = data[1][0]
                drug_feat, drug_adj, gexpr, y_true = DataG
                drug_feat = drug_feat.to(device)
                drug_adj = drug_adj.to(device)
                gexpr = gexpr.to(device)
                y_true = y_true.view(-1,1).to(device)
                t_gexpr = t_gexpr.to(device)
                EN_model.train()
                GCN_model.train()
                ADV_model.train()

                optimizer_adv.zero_grad()
                F_gexpr,_,_ = EN_model(gexpr)
                F_t_gexpr,_,_ = EN_model(t_gexpr)

                F_g_t_gexpr = torch.cat((F_gexpr,F_t_gexpr))
                z_true = torch.cat((torch.zeros(F_gexpr.shape[0], device=device), torch.ones(F_t_gexpr.shape[0], device=device)))
                z_true = z_true.view(-1,1)
                z_pred = ADV_model(F_g_t_gexpr)
                if IsNaN(z_pred): return -1
                adv_loss = loss(z_pred, z_true)
                adv_loss.backward()
                optimizer_adv.step()

                optimizer.zero_grad()

                g_latents, _, _ = EN_model(gexpr)
                t_latents, _, _ = EN_model(t_gexpr)

                F_g_t_latents = torch.cat((g_latents,t_latents))
                z_true_ = torch.cat((torch.ones(g_latents.shape[0], device=device), torch.zeros(t_latents.shape[0], device=device)))
                z_true_ = z_true_.view(-1,1)
                z_pred_ = ADV_model(F_g_t_latents)
                y_pred = GCN_model(drug_feat,drug_adj,g_latents)
                if IsNaN(z_pred_) or IsNaN(y_pred): return -1
                
                adv_loss_ = loss(z_pred_, z_true_)
                cdr_loss = criterion(y_pred, y_true)

                Loss = cdr_loss + lam*adv_loss_
                Loss.backward()
                optimizer.step()

            with torch.no_grad():
                EN_model.eval()
                GCN_model.eval()
                ADV_model.eval()

                F_gexpr_val,_,_ = EN_model(X_gexpr_val)
                F_t_gexpr_val,_,_ = EN_model(X_t_gexpr_val)

                F_g_t_gexpr_val = torch.cat((F_gexpr_val, F_t_gexpr_val))
                z_pred_val = ADV_model(F_g_t_gexpr_val)
                y_pred_val = GCN_model(X_drug_feat_val, X_drug_adj_val, F_gexpr_val)
                loss_val = criterion(y_pred_val, Y_val.view(-1,1)) + lam*loss(z_pred_val, torch.ones(z_pred_val.shape, device=device))
                p_val = pearsonr(Y_val.cpu().view(-1).detach().numpy(), y_pred_val.cpu().view(-1).detach().numpy())[0]
                
            if p_val >= best_p:#loss_val <= best_loss:
                wait = 0
                best_p = p_val
                best_loss = loss_val
                torch.save({'EN_model': EN_model.state_dict(), 'GCN_model':GCN_model.state_dict(), 
                            'ADV_model':ADV_model.state_dict()}, weight_path)

                
            else:
                wait += 1
                if wait >= 10: break
        
        return best_p

    def predict(self, data, params, weight_path):
        nz,d_dim,_,_,_,_ = params.values()
        drug_feat, drug_adj, gexpr = data

        EN_model = Encoder(gexpr.shape[1], nz, device)
        GCN_model = GCN(drug_feat.shape[2],[256,256,256],h_dims=[d_dim, nz+d_dim],use_dropout=
        False,is_regr=True)

        checkpoint = torch.load(weight_path)
        EN_model.load_state_dict(checkpoint['EN_model'])
        GCN_model.load_state_dict(checkpoint['GCN_model'])
        EN_model.to(device)
        GCN_model.to(device)

        with torch.no_grad():
            EN_model.eval()
            GCN_model.eval()

            F_gexpr = EN_model(gexpr)[0]
            y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)

        return y_pred.cpu()

def train_PANCDR_nested_regr(n_outer_splits,data,best_params_file):
    outer_splits = KFold(n_splits=n_outer_splits,shuffle=True,random_state=0)
    p_test_df = pd.DataFrame(columns=['Test_Pearson','Best_params'])
    X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,t_gexpr_feature = data
    best_params_df = pd.read_csv(best_params_file,index_col=0)
    for outer_fold,(idx,test_idx) in enumerate(outer_splits.split(X_drug_feat_data)):
        X_drug_feat_data_ = X_drug_feat_data[idx]
        X_drug_adj_data_ = X_drug_adj_data[idx]
        X_gexpr_data_ = X_gexpr_data[idx]
        Y_ = Y[idx]
        best_params = eval(best_params_df.loc["Fold_%d"%outer_fold,"Best_params"])
        weight_path='../checkpoint/kfold/regr_model_best_outerfold_%d.pt'%outer_fold

        X_drug_feat_data_test = X_drug_feat_data[test_idx]
        X_drug_adj_data_test = X_drug_adj_data[test_idx]
        X_gexpr_data_test = X_gexpr_data[test_idx]
        Y_test = Y[test_idx]

        X_drug_feat_test = torch.FloatTensor(X_drug_feat_data_test).to(device)
        X_drug_adj_test = torch.FloatTensor(X_drug_adj_data_test).to(device)
        X_gexpr_test = torch.FloatTensor(X_gexpr_data_test).to(device)
        Y_test = torch.FloatTensor(Y_test).to(device)

        train_data = [X_drug_feat_data_,X_drug_adj_data_,X_gexpr_data_,Y_,t_gexpr_feature]
        test_data = [X_drug_feat_test,X_drug_adj_test,X_gexpr_test,Y_test]

        model = train_PANCDR_regr(train_data,test_data)
        while True:
            p_val = model.train(best_params, weight_path)
            if p_val != -1: break
        y_pred_TEST = model.predict(test_data[:-1],best_params, weight_path)
        p_TEST = pearsonr(Y_test.cpu().view(-1).detach().numpy(), y_pred_TEST.view(-1).detach().numpy())
        temp_test_df = pd.DataFrame([[p_TEST,best_params]], index=['Fold_%d'%outer_fold], columns=['Test_Pearson','Best_params'])
        p_test_df = pd.concat([p_test_df,temp_test_df])

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    print('Mean test Pearson - %.4f\n'%p_test_df['Test_Pearson'].mean())
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    p_test_df = pd.concat([p_test_df, pd.DataFrame(p_test_df['Test_Pearson'].mean(), index=['mean'], columns=['Test_Pearson'])])
    return p_test_df

def IsNaN(pred):
    return torch.isnan(pred).sum()>0