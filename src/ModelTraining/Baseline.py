import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import pandas as pd
israndom=False
from sklearn.model_selection import StratifiedKFold,KFold

from ModelTraining.model import DeepCDR
device = torch.device('cuda')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class train_baseline():
    def __init__(self,train_data,test_data):
        self.train_data = train_data
        self.test_data = test_data

    def train(self,params,weight_path='../checkpoint/kfold/base_model.pt'):
        nz,d_dim,lr,batch_size = params.values()

        X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y = self.train_data
        TX_drug_feat_data_test,TX_drug_adj_data_test,TX_gexpr_data_test,TY_test = self.test_data
        X_drug_feat_data_train,X_drug_feat_data_val,X_drug_adj_data_train,X_drug_adj_data_val,X_gexpr_data_train,X_gexpr_data_val,Y_train,Y_val= train_test_split(X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,test_size=0.05, random_state=0)
        
        X_drug_feat_train = torch.FloatTensor(X_drug_feat_data_train)
        X_drug_adj_train = torch.FloatTensor(X_drug_adj_data_train)
        X_gexpr_train = torch.FloatTensor(X_gexpr_data_train)
        Y_train = torch.FloatTensor(Y_train)

        X_drug_feat_val = torch.FloatTensor(X_drug_feat_data_val).to(device)
        X_drug_adj_val = torch.FloatTensor(X_drug_adj_data_val).to(device)
        X_gexpr_val = torch.FloatTensor(X_gexpr_data_val).to(device)
        Y_val = torch.FloatTensor(Y_val).to(device)
        
        GDSC_Dataset = torch.utils.data.TensorDataset(X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train)
        GDSC_Loader = torch.utils.data.DataLoader(dataset=GDSC_Dataset, batch_size = batch_size, shuffle=True, drop_last=True)

        wait, best_auc = 0, 0
        model = DeepCDR(X_gexpr_train.shape[1], nz,X_drug_feat_train.shape[2],[256,256,256],h_dim=d_dim)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss = torch.nn.BCELoss()

        for epoch in range(1000):
            for i,data in enumerate(GDSC_Loader):

                drug_feat, drug_adj, gexpr, y_true = data
                drug_feat = drug_feat.to(device)
                drug_adj = drug_adj.to(device)
                gexpr = gexpr.to(device)
                y_true = y_true.view(-1,1).to(device)
                model.train()

                optimizer.zero_grad()

                y_pred = model(drug_feat,drug_adj,gexpr)
                Loss = loss(y_pred, y_true)

                Loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()

                y_pred_val = model(X_drug_feat_val, X_drug_adj_val, X_gexpr_val)
                loss_val = loss(y_pred_val, Y_val.view(-1,1))
                auc_val = roc_auc_score(Y_val.cpu().detach().numpy(), y_pred_val.cpu().detach().numpy())
                
                y_pred_TEST = model(TX_drug_feat_data_test, TX_drug_adj_data_test, TX_gexpr_data_test)
                auc_TEST = roc_auc_score(TY_test.cpu().detach().numpy(), y_pred_TEST.cpu().detach().numpy())
                
                
            if auc_val >= best_auc:
                wait = 0
                best_auc = auc_val
                best_auc_TEST = auc_TEST
                torch.save({'model': model.state_dict()}, weight_path)

                
            else:
                wait += 1
                if wait >= 10: break
        
        return best_auc_TEST

    def predict(self, data, params, weight_path):
        nz,d_dim,_,_ = params.values()
        drug_feat, drug_adj, gexpr = data

        model = DeepCDR(gexpr.shape[1], nz,drug_feat.shape[2],[256,256,256],h_dim=d_dim)
        

        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['model'])
        model.to(device)

        with torch.no_grad():
            model.eval()
            y_pred = model(drug_feat, drug_adj, gexpr)

        return y_pred.cpu()

def train_baseline_nested(n_outer_splits,data,best_params_file):
    X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y= data
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

        train_data = [X_drug_feat_data_,X_drug_adj_data_,X_gexpr_data_,Y_]
        test_data = [X_drug_feat_test,X_drug_adj_test,X_gexpr_test,Y_test]

        model = train_baseline(train_data,test_data)
        auc_TEST = model.train(best_params, weight_path='../checkpoint/kfold/base_model_best_outerfold_%d.pt'%outer_fold)
        temp_test_df = pd.DataFrame([[auc_TEST,best_params]], index=['Fold_%d'%outer_fold], columns=['Test_AUC','Best_params'])
        auc_test_df = pd.concat([auc_test_df,temp_test_df])

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    print('Mean test AUC - %.4f\n'%auc_test_df['Test_AUC'].mean())
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    auc_test_df = pd.concat([auc_test_df, pd.DataFrame(auc_test_df['Test_AUC'].mean(), index=['mean'], columns=['Test_AUC'])])
    return auc_test_df