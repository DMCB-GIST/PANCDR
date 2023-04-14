import torch
import random,os
import numpy as np
from sklearn import metrics
import pandas as pd
from utils import DataGenerate, DataFeature
from ModelTraining.PANCDR import train_PANCDR_regr
os.environ["CUDA_VISIBLE_DEVICES"]="8"
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
Drug_info_file = '%s/GDSC/GDSC_drug_IC50.csv'%DPATH
Cell_line_info_file = '%s/GDSC/Cell_Lines_Details.txt'%DPATH
Drug_feature_file = '%s/GDSC/drug_graph_feat'%DPATH
Cancer_response_exp_file = '%s/GDSC/GDSC_IC50_response_357.csv'%DPATH
Gene_expression_file = '%s/GDSC/GDSC_expr_z_702.csv'%DPATH
Max_atoms = 100
P_Gene_expression_file = '%s/TCGA/Pretrain_TCGA_expr_702_01A.csv'%DPATH
T_Drug_info_file = '%s/TCGA/TCGA_drug_new.csv'%DPATH
T_Patient_info_file = '%s/TCGA/TCGA_type_new.txt'%DPATH
T_Drug_feature_file = '%s/TCGA/drug_graph_feat'%DPATH
T_Cancer_response_exp_file = '%s/TCGA/TCGA_response_new.csv'%DPATH
T_Gene_expression_file = '%s/TCGA/TCGA_expr_z_702.csv'%DPATH

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

    train_data = [X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,t_gexpr_feature]
    test_data = [TX_drug_feat_data_test,TX_drug_adj_data_test,TX_gexpr_data_test,TY_test]
    
    df = pd.read_csv("tuned_hyperparameters/TCGA_CV_params.csv")
    best_params = eval(df.loc[(df["Model"]=="PANCDR") & (df["Classification"]=="F"),"Best_params"].values[0])
    model = train_PANCDR_regr(train_data,test_data)
    results = []
    print("Training......")
    for iter in range(2):
        weight_path = '../checkpoint/TCGA/%d_regr_model.pt'%iter
        p_val = model.train(best_params,weight_path=weight_path)
        print('iter %d - p-val: %.4f'%(iter,p_val))
        results.append(p_val)
    
    result_df = pd.DataFrame(results, columns=['TCGA AUC'])
    result_df.loc['mean',] = result_df.mean().values
    result_df.to_csv('TCGA_10train_regr.csv')

    best_iter = result_df.idxmax()[0]
    best_weight_path = '../checkpoint/TCGA/%d_regr_model.pt'%best_iter

    y_test_pred = model.predict(test_data[:-1],best_params,best_weight_path)
    result = pd.DataFrame([y_test_pred.view(-1).tolist(),TY_test.view(-1).tolist()],index=['predicted IC50','response']).T
    result.to_csv("predicted_TCGA_IC50.csv")