import csv
from fcntl import DN_DELETE
import pandas as pd
import os
import hickle as hkl
import numpy as np
import scipy.sparse as sp
import random
from matplotlib import pyplot as plt
import umap.umap_ as umap
import torch
from sklearn import metrics
device = torch.device('cuda')

israndom=False
Max_atoms = 100
TCGA_label_set = ["ACC","ALL","BLCA","BRCA","CESC","CLL","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG","LUSC","MB",
                  "MESO","MM","NB","OV","PAAD","PRAD","SCLC","SKCM","STAD",
                  "THCA","UCEC",'COAD/READ','']

def DataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,P_Gene_expression_file,Cancer_response_exp_file,is_regr=False,dataset='GDSC'):
    if dataset.upper() == "GDSC":
        data = MetadataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,P_Gene_expression_file,Cancer_response_exp_file,is_regr)
    elif dataset.upper() == "TCGA":
        data = T_MetadataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,Cancer_response_exp_file)
    else:
        print("Please check the dataset. This function works only for \"GDSC\" or \"TCGA\"")
    return data

def DataFeature(data_idx,drug_feature,gexpr_feature,dataset="GDSC"):
    if dataset.upper() == "GDSC":
        data = FeatureExtract(data_idx,drug_feature,gexpr_feature)
    elif dataset.upper() == "TCGA":
        data = T_FeatureExtract(data_idx,drug_feature,gexpr_feature)
    else:
        print("Please check the dataset. This function works only for \"GDSC\" or \"TCGA\"")
    return data

#split into training and test set
def DataSplit(data_idx,ratio = 0.95):
    data_train_idx,data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1]==each_type]
        train_list = random.sample(data_subtype_idx,int(ratio*len(data_subtype_idx)))
        test_list = [item for item in data_subtype_idx if item not in train_list]
        data_train_idx += train_list
        data_test_idx += test_list
    return data_train_idx,data_test_idx

def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm
def random_adjacency_matrix(n):   
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix
def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])        
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2    
    return [feat,adj_mat]

def MetadataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,P_Gene_expression_file,Cancer_response_exp_file,is_regr=False):
    #drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file,'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[-1]:item[-1] for item in rows if item[-1].isdigit()}

    #map cellline --> cancer type
    cellline2cancertype ={}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[0]
        TCGA_label = line.strip().split('\t')[9]
        #if TCGA_label in TCGA_label_set:
        cellline2cancertype[cellline_id] = TCGA_label

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(Drug_feature_file,each))
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(drug_feature.values())
    
    #load gene expression faetures
    gexpr_feature = pd.read_csv(Gene_expression_file,sep=',',header=0,index_col=[0])
    
    experiment_data = pd.read_csv(Cancer_response_exp_file,sep=',',header=0,index_col=[0])
    experiment_data.columns = experiment_data.columns.astype(str)
    #filter experiment data
    drug_match_list=[item for item in experiment_data.columns if item in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data[drug_match_list]
    
    #load TCGA pretrain gene expression
    t_gexpr_feature = pd.read_csv(P_Gene_expression_file, index_col=0)
    #t_gexpr_feature = t_gexpr_feature.T

    data_idx = []
    if is_regr:
        for each_drug in experiment_data_filtered.columns:
            for each_cellline in experiment_data_filtered.index:
                pubchem_id = drugid2pubchemid[each_drug]
                if str(pubchem_id) in drug_pubchem_id_set and each_cellline in gexpr_feature.index:
                    if not np.isnan(experiment_data_filtered.loc[each_cellline,each_drug]) and each_cellline in cellline2cancertype.keys():
                        IC50 = float(experiment_data_filtered.loc[each_cellline,each_drug])
                        data_idx.append((each_cellline,pubchem_id,IC50,cellline2cancertype[each_cellline]))  
    else:
        for each_drug in experiment_data_filtered.columns:
            for each_cellline in experiment_data_filtered.index:
                pubchem_id = drugid2pubchemid[each_drug]
                if str(pubchem_id) in drug_pubchem_id_set and each_cellline in gexpr_feature.index:
                    if not np.isnan(experiment_data_filtered.loc[each_cellline,each_drug]) and each_cellline in cellline2cancertype.keys():
                        binary_IC50 = int(experiment_data_filtered.loc[each_cellline,each_drug])
                        data_idx.append((each_cellline,pubchem_id,binary_IC50,cellline2cancertype[each_cellline])) 
    
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.'%(len(data_idx),nb_celllines,nb_drugs))
    return drug_feature, gexpr_feature, t_gexpr_feature, data_idx

def FeatureExtract(data_idx,drug_feature,gexpr_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    nb_gexpr_features = gexpr_feature.shape[1]
    drug_data = [[] for item in range(nb_instance)]
    gexpr_data = np.zeros((nb_instance,nb_gexpr_features),dtype='float32') 
    target = np.zeros(nb_instance,dtype='float32')
    for idx in range(nb_instance):
        cell_line_id,pubchem_id,binary_IC50,cancer_type = data_idx[idx]
        #modify
        feat_mat,adj_list,_ = drug_feature[str(pubchem_id)]
        #fill drug data,padding to the same size with zeros
        drug_data[idx] = CalculateGraphFeat(feat_mat,adj_list)
        #randomlize X A
        gexpr_data[idx,:] = gexpr_feature.loc[cell_line_id].values
        target[idx] = binary_IC50
        cancer_type_list.append([cancer_type,cell_line_id,pubchem_id])
    return drug_data,gexpr_data,target,cancer_type_list

def T_MetadataGenerate(T_Drug_info_file,T_Patient_info_file,T_Drug_feature_file,T_Gene_expression_file,T_Cancer_response_exp_file):
        #drug_id --> pubchem_id
    reader = csv.reader(open(T_Drug_info_file,'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[-1]:item[-1] for item in rows if item[-1].isdigit()}

    #map patient --> cancer type
    patient2cancertype ={}
    for line in open(T_Patient_info_file).readlines()[1:]:
       patient_id = line.split('\t')[0]
       TCGA_label = line.strip().split('\t')[1]
        # if TCGA_label in TCGA_label_set:
       patient2cancertype[patient_id] = TCGA_label

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(T_Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(T_Drug_feature_file,each))
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(drug_feature.values())
    
    #load gene expression faetures
    gexpr_feature = pd.read_csv(T_Gene_expression_file,sep=',',header=0,index_col=[0])
    
    experiment_data = pd.read_csv(T_Cancer_response_exp_file,sep=',',header=0,index_col=[0])
    #filter experiment data
    drug_match_list=[item for item in experiment_data.columns if item in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data[drug_match_list]

    data_idx = []
    for each_drug in experiment_data_filtered.columns:
        for each_patient in experiment_data_filtered.index:
            pubchem_id = drugid2pubchemid[each_drug]
            if str(pubchem_id) in drug_pubchem_id_set and each_patient in gexpr_feature.index:
                if not np.isnan(experiment_data_filtered.loc[each_patient,each_drug]) and each_patient in patient2cancertype.keys():
                    binary_IC50 = int(experiment_data_filtered.loc[each_patient,each_drug])
                    data_idx.append((each_patient,pubchem_id,binary_IC50,patient2cancertype[each_patient])) 
    nb_patient = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d patients and %d drugs were generated.'%(len(data_idx),nb_patient,nb_drugs))
    return drug_feature, gexpr_feature, data_idx

def T_FeatureExtract(data_idx,drug_feature,gexpr_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    nb_gexpr_features = gexpr_feature.shape[1]
    drug_data = [[] for _ in range(nb_instance)]
    gexpr_data = np.zeros((nb_instance,nb_gexpr_features),dtype='float32') 
    target = np.zeros(nb_instance,dtype='float32')
    for idx in range(nb_instance):
        patient_id,pubchem_id,binary_IC50,cancer_type = data_idx[idx] ###
        #modify
        feat_mat,adj_list,_ = drug_feature[str(pubchem_id)]
        #fill drug data,padding to the same size with zeros
        drug_data[idx] = CalculateGraphFeat(feat_mat,adj_list)
        #randomlize X A
        gexpr_data[idx,:] = gexpr_feature.loc[patient_id].values
        target[idx] = binary_IC50
        cancer_type_list.append([cancer_type,patient_id,pubchem_id])
    return drug_data,gexpr_data,target,cancer_type_list

def scores(y_true, y_pred):
    fpr, tpr, thr = metrics.roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr-fpr)
    optimal_thr = thr[optimal_idx]
    y_pred_ = (y_pred > optimal_thr).astype(int)
    auc = metrics.roc_auc_score(y_true,y_pred)
    acc = metrics.accuracy_score(y_true,y_pred_)
    precision = metrics.precision_score(y_true,y_pred_)
    recall = metrics.recall_score(y_true,y_pred_)
    f1 = metrics.f1_score(y_true, y_pred_)
    return auc,acc,precision,recall,f1

def umap_img(model, gexpr, t_gexpr, path):
    ## model: feature extract model
    ## gexpr, t_gexpr: numpy array, GDSC and TCGA gene expression
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['FE_model'])
    model.to(device)
    
    before = pd.concat([gexpr,t_gexpr])
    
    reducer = umap.UMAP(random_state = 12345)
    encoded  = reducer.fit_transform(before.values)  
    embedding_df = pd.DataFrame(encoded, index = before.index)
    
    colors = ['#ED4C67', '#1289A7']
    label_names = ['GDSC', 'TCGA']
    
    fig, ax = plt.subplots(nrows=1,ncols=2)
    fig.set_size_inches(10,5)

    SMALL_SIZE = 10
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 30

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    #plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    encoded_data = embedding_df.values
    labels = np.concatenate((np.zeros(shape=(gexpr.shape[0],),dtype=int), np.ones(shape=(t_gexpr.shape[0],), dtype=int)))
    label_types = np.unique(labels)
    
    label_flags = [0, 0]
    for i in range(encoded_data.shape[0]):
        if label_flags[np.where(label_types == labels[i])[0][0]] == 0:
            ax[0].scatter(encoded_data[i, 0], encoded_data[i, 1], 
                       s = 100,  alpha = 0.4,  linewidth='3',
                       color = '#ffffff', edgecolor = colors[np.where(label_types == labels[i])[0][0]],
                       label = label_names[np.where(label_types == labels[i])[0][0]])
            label_flags[np.where(label_types == labels[i])[0][0]] = 1
        else:
            ax[0].scatter(encoded_data[i, 0], encoded_data[i, 1], 
                       s = 100, alpha = 0.4,  linewidth='3',
                       color = '#ffffff', edgecolor = colors[np.where(label_types == labels[i])[0][0]])
            
    ax[0].legend()
    ax[0].axis("off")
    ax[0].title.set_text("before")
    
    with torch.no_grad():
        model.eval()
        after,_,_ = model(torch.FloatTensor(before.values).to(device))
    after = pd.DataFrame(after.detach().cpu().numpy())
    reducer = umap.UMAP(random_state = 12345)
    encoded  = reducer.fit_transform(after.values)  
    embedding_df = pd.DataFrame(encoded, index = embedding_df.index)
    encoded_data = embedding_df.values
    label_flags = [0, 0]
    for i in range(encoded_data.shape[0]):
        if label_flags[np.where(label_types == labels[i])[0][0]] == 0:
            ax[1].scatter(encoded_data[i, 0], encoded_data[i, 1], 
                   s = 100,  alpha = 0.4,  linewidth='3',
                   color = '#ffffff', edgecolor = colors[np.where(label_types == labels[i])[0][0]],
                   label = label_names[np.where(label_types == labels[i])[0][0]])
            label_flags[np.where(label_types == labels[i])[0][0]] = 1
        else:
            ax[1].scatter(encoded_data[i, 0], encoded_data[i, 1], 
                   s = 100, alpha = 0.4,  linewidth='3',
                   color = '#ffffff', edgecolor = colors[np.where(label_types == labels[i])[0][0]])
            
    ax[1].legend()
    ax[1].axis("off")
    ax[1].title.set_text("after")
    
    plt.show()
    plt.close()
