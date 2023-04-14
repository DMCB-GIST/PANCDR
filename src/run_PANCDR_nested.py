import torch
import random,os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
israndom=False
from utils import DataGenerate, DataFeature
from ModelTraining.PANCDR import train_PANCDR_nested

os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TCGA_label_set = ["ACC","ALL","BLCA","BRCA","CESC","CLL","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG","LUSC","MB",
                  "MESO","MM","NB","OV","PAAD","PRAD","SCLC","SKCM","STAD",
                  "THCA","UCEC",'COAD/READ','']
DPATH = '../data'
Drug_info_file = '%s/GDSC/GDSC_drug_binary.csv'%DPATH
Cell_line_info_file = '%s/GDSC/Cell_Lines_Details.txt'%DPATH
Drug_feature_file = '%s/GDSC/drug_graph_feat'%DPATH
Cancer_response_exp_file = '%s/GDSC/GDSC_binary_response_151.csv'%DPATH
Gene_expression_file = '%s/GDSC/GDSC_expr_z_702.csv'%DPATH
Max_atoms = 100
P_Gene_expression_file = '%s/TCGA/Pretrain_TCGA_expr_702_01A.csv'%DPATH
T_Drug_info_file = '%s/TCGA/TCGA_drug_new.csv'%DPATH
T_Patient_info_file = '%s/TCGA/TCGA_type_new.txt'%DPATH
T_Drug_feature_file = '%s/TCGA/drug_graph_feat'%DPATH
T_Cancer_response_exp_file = '%s/TCGA/TCGA_response_new.csv'%DPATH
T_Gene_expression_file = '%s/TCGA/TCGA_expr_z_702.csv'%DPATH

nz_ls = [100, 128, 256]
h_dims_ls = [100, 128, 256]
lr_ls = [0.001, 0.0001]
lam_ls = [1, 0.1, 0.01]
batch_size_ls = [[128,14],[256,28]]

def f1(y_true, y_pred):
    fpr, tpr, thr = metrics.roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr-fpr)
    optimal_thr = thr[optimal_idx]
    y_pred_ = (y_pred > optimal_thr).astype(int)
    output = metrics.f1_score(y_true, y_pred_)
    return output


if __name__ == '__main__':
    drug_feature,gexpr_feature, t_gexpr_feature, data_idx = DataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,P_Gene_expression_file,Cancer_response_exp_file)
    T_drug_feature, T_gexpr_feature, T_data_idx = DataGenerate(T_Drug_info_file,T_Patient_info_file,T_Drug_feature_file,T_Gene_expression_file,None,T_Cancer_response_exp_file,dataset="TCGA")
    TX_drug_data_test,TX_gexpr_data_test,TY_test,Tcancer_type_test_list = DataFeature(T_data_idx,T_drug_feature,T_gexpr_feature,dataset="TCGA")

    TX_drug_feat_data_test = [item[0] for item in TX_drug_data_test]
    TX_drug_adj_data_test = [item[1] for item in TX_drug_data_test]
    TX_drug_feat_data_test = np.array(TX_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    TX_drug_adj_data_test = np.array(TX_drug_adj_data_test)#nb_instance * Max_stom * Max_stom  

    TX_drug_feat_data_test = torch.FloatTensor(TX_drug_feat_data_test).to(device)
    TX_drug_adj_data_test = torch.FloatTensor(TX_drug_adj_data_test).to(device)
    TX_gexpr_data_test = torch.FloatTensor(TX_gexpr_data_test).to(device)
    TY_test = torch.FloatTensor(TY_test).to(device)

    X_drug_data,X_gexpr_data,Y,cancer_type_train_list = DataFeature(data_idx,drug_feature,gexpr_feature)
    X_drug_feat_data = [item[0] for item in X_drug_data]
    X_drug_adj_data = [item[1] for item in X_drug_data]
    X_drug_feat_data = np.array(X_drug_feat_data)
    X_drug_adj_data = np.array(X_drug_adj_data)

    data = [X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,t_gexpr_feature]
    
    n_outer_splits= 10
    best_params_file = "tuned_hyperparameters/GDSC_nested_outer_CV_params.csv"
    auc_test_df = train_PANCDR_nested(n_outer_splits,data,best_params_file)

    auc_test_df.to_csv('GDSC_nested.csv', sep=',')

