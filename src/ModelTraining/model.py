import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from typing import Optional

class GraphConv(nn.Module):
    def __init__(self,
                 in_channels,units,step_num=1):
        super(GraphConv,self).__init__()
        self.weight = Parameter(torch.empty((in_channels, units)))
        self.bias = Parameter(torch.empty(units))
        self.step_num = step_num
        self.reset_parameters()
        
    def reset_parameters(self)->None:
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
        
    def _get_walked_edges(self, edges, step_num):
        if step_num <= 1:
            return edges
        deeper = self._get_walked_edges(torch.matmul(edges,edges), step_num//2) ##?
        if step_num %2 == 1:
            deeper += edges
        return torch.gt(deeper, 0.0) 
        
    def _forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        features, edges = input
        outputs = torch.matmul(features, weight) + bias
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        outputs = torch.matmul(edges.permute(0,2,1),outputs)
        return outputs.permute(0,2,1)
        
    def forward(self, input: Tensor) -> Tensor:
        return self._forward(input, self.weight, self.bias)
        
    def get_config(self):
        config = {'in_channels': self.inchannels, ' units':self.units}
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DeepCDR(nn.Module):
    def __init__(self, input_dim, nz, in_channels, units_list, h_dim=100, n_hidden=256, is_regr=False):
        super(DeepCDR, self).__init__()
        self.nz = nz
        self.h_dim=h_dim
        self.fe = nn.Sequential(nn.Linear(input_dim, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.Tanh(),
                                nn.Dropout(0.1),
                                nn.Linear(n_hidden,nz),
                                nn.ReLU())
        self.fe.apply(self.init_weights)

        self.GC1 = GraphConv(in_channels,units=units_list[0],step_num=1)
        self.GC2 = GraphConv(units_list[0],units=units_list[1],step_num=1)
        self.GC3 = GraphConv(units_list[1],units=units_list[2],step_num=1)
        self.GC4 = GraphConv(units_list[2],units=h_dim,step_num=1)
        RBD1 = [nn.ReLU(),nn.BatchNorm1d(units_list[0]),nn.Dropout(0.1)]
        RBD2 = [nn.ReLU(),nn.BatchNorm1d(units_list[1]),nn.Dropout(0.1)]
        RBD3 = [nn.ReLU(),nn.BatchNorm1d(units_list[2]),nn.Dropout(0.1)]
        RBD4 = [nn.ReLU(),nn.BatchNorm1d(h_dim),nn.Dropout(0.1)]
        self.RBD1 = nn.Sequential(*RBD1)
        self.RBD2 = nn.Sequential(*RBD2)
        self.RBD3 = nn.Sequential(*RBD3)
        self.RBD4 = nn.Sequential(*RBD4)

        self.Pool = nn.AdaptiveAvgPool1d(1)
        self.Linear = nn.Sequential(nn.Linear(nz+h_dim,300),
                                    nn.Dropout(0.1))
        self.CONV = nn.Sequential(nn.Conv2d(1, 30, kernel_size=(150,1), stride=(1,1)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((2,1)),
                                  nn.Conv2d(30, 10, kernel_size=(5,1), stride=(1,1)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((3,1)),
                                  nn.Conv2d(10, 5, kernel_size=(5,1), stride=(1,1)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((3,1)),
                                  nn.Dropout(0.1),
                                  nn.Flatten(),
                                  nn.Dropout(0.2))
                                  
        if is_regr:
            self.fc = nn.Linear(30,1)
        else:
            self.fc = nn.Sequential(nn.Linear(30,1),
                                    nn.Sigmoid())
        self.Linear.apply(self.init_weights)
        self.CONV.apply(self.init_weights)
        self.fc.apply(self.init_weights)

        
    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
   
    def forward(self, drug_feat, drug_adj, gexpr):
        GCN_layer = self.RBD1(self.GC1([drug_feat, drug_adj])).permute(0,2,1)
        GCN_layer = self.RBD2(self.GC2([GCN_layer, drug_adj])).permute(0,2,1)
        GCN_layer = self.RBD3(self.GC3([GCN_layer, drug_adj])).permute(0,2,1)
        GCN_layer = self.RBD4(self.GC4([GCN_layer, drug_adj]))
        GCN_layer = self.Pool(GCN_layer)
        x_drug = GCN_layer.view(GCN_layer.shape[0],-1)
        x_gexpr = self.fe(gexpr)
        x = torch.cat((x_drug,x_gexpr),1)
        x = self.Linear(x)
        x = x.view(-1,1,300,1)
        x = self.CONV(x)
        output = self.fc(x)
        return output
    

class FE(nn.Module):
    def __init__(self, n_input, nz, device, n_hidden=256):
        super(FE, self).__init__()
        self.nz = nz
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.device = device
        
        encoder = [nn.Linear(self.n_input,self.n_hidden),
                   nn.BatchNorm1d(self.n_hidden),
                   nn.ReLU()]
        n_layers = 3
        for i in range(n_layers):
            encoder += [nn.Linear(self.n_hidden,self.n_hidden),
                        nn.BatchNorm1d(self.n_hidden),
                        nn.ReLU()]
        encoder += [nn.Linear(self.n_hidden,self.n_hidden)]
        self.encoder = nn.Sequential(*encoder)
        
        self.fc = nn.Linear(self.n_hidden, self.nz)
        
    def forward(self, x):
        return self.fc(self.encoder(x))


class Encoder(nn.Module):
    def __init__(self, n_input, nz, device, n_hidden=256):
        super(Encoder, self).__init__()
        self.nz = nz
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.device = device
        
        encoder = [nn.Linear(self.n_input,self.n_hidden),
                   nn.BatchNorm1d(self.n_hidden),
                   nn.ReLU()]
        n_layers = 3
        for i in range(n_layers):
            encoder += [nn.Linear(self.n_hidden,self.n_hidden),
                        nn.BatchNorm1d(self.n_hidden),
                        nn.ReLU()]
        encoder += [nn.Linear(self.n_hidden,self.n_hidden)]
        self.encoder = nn.Sequential(*encoder)
        
        self.fc1 = nn.Linear(self.n_hidden, self.nz)
        self.fc2 = nn.Linear(self.n_hidden, self.nz)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc1(h), self.fc2(h)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)

        return eps * std + mu

    
class Decoder(nn.Module):
    def __init__(self, n_output, nz, n_hidden=256):
        super(Decoder,self).__init__()
        decoder = [nn.Linear(nz,n_hidden),
                   nn.BatchNorm1d(n_hidden),
                   nn.ReLU()]
        
        n_layers = 3
        for i in range(n_layers):
            decoder += [nn.Linear(n_hidden,n_hidden),
                        nn.BatchNorm1d(n_hidden),
                        nn.ReLU()]
        decoder += [nn.Linear(n_hidden,n_output)]
        self.decoder = nn.Sequential(*decoder)
        
    def forward(self,z):
        res = self.decoder(z)
        return res
    
    
        
class GCN(nn.Module):
    def __init__(self,in_channels,units_list, h_dims=[100,200], use_dropout=False, is_regr=False): #units_list = [256,256,256]
        super(GCN, self).__init__()
        self.GC1 = GraphConv(in_channels,units=units_list[0],step_num=1)
        self.GC2 = GraphConv(units_list[0],units=units_list[1],step_num=1)
        self.GC3 = GraphConv(units_list[1],units=units_list[2],step_num=1)
        self.GC4 = GraphConv(units_list[2],units=h_dims[0],step_num=1)
        BRD1 = [nn.BatchNorm1d(units_list[0]),nn.ReLU()]
        BRD2 = [nn.BatchNorm1d(units_list[1]),nn.ReLU()]
        BRD3 = [nn.BatchNorm1d(units_list[2]),nn.ReLU()]
        BRD4 = [nn.BatchNorm1d(h_dims[0]),nn.ReLU()]

        if use_dropout:
            BRD1 += [nn.Dropout(0.1)]
            BRD2 += [nn.Dropout(0.1)]
            BRD3 += [nn.Dropout(0.1)]
            BRD4 += [nn.Dropout(0.1)]
            
        self.BRD1 = nn.Sequential(*BRD1)
        self.BRD2 = nn.Sequential(*BRD2)
        self.BRD3 = nn.Sequential(*BRD3)
        self.BRD4 = nn.Sequential(*BRD4)
        
        self.Pool = nn.AdaptiveAvgPool1d(1)
        self.Linear = nn.Sequential(nn.Linear(h_dims[1],300),
                                    nn.Dropout(0.1))
        self.CONV = nn.Sequential(nn.Conv2d(1, 30, kernel_size=(150,1), stride=(1,1)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((2,1)),
                                  nn.Conv2d(30, 10, kernel_size=(5,1), stride=(1,1)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((3,1)),
                                  nn.Conv2d(10, 5, kernel_size=(5,1), stride=(1,1)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((3,1)),
                                  nn.Dropout(0.2))
                                  
        if is_regr:
            self.fc = nn.Linear(30,1)
        else:
            self.fc = nn.Sequential(nn.Linear(30,1),
                                    nn.Sigmoid())
        
        #self.Linear.apply(self.init_weights)
        #self.CONV.apply(self.init_weights)
        #self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        
         
    def forward(self, drug_feat, drug_adj, x_gexpr):
        GCN_layer = self.BRD1(self.GC1([drug_feat, drug_adj])).permute(0,2,1)
        GCN_layer = self.BRD2(self.GC2([GCN_layer, drug_adj])).permute(0,2,1)
        GCN_layer = self.BRD3(self.GC3([GCN_layer, drug_adj])).permute(0,2,1)
        GCN_layer = self.BRD4(self.GC4([GCN_layer, drug_adj]))
        GCN_layer = self.Pool(GCN_layer)
        x_drug = GCN_layer.view(GCN_layer.shape[0],-1)
        x = torch.cat((x_drug,x_gexpr),1)
        x = self.Linear(x)
        x = x.view(-1,1,300,1)
        output = self.CONV(x)
        output = self.fc(output.view(output.shape[0],-1))
        return output
        
class ADV(nn.Module):
    def __init__(self, nz):
        super(ADV, self).__init__()
        self.adv = nn.Sequential(nn.Linear(nz,nz//2),
                                 nn.ReLU(),
                                 nn.Linear(nz//2,nz//4),
                                 nn.ReLU(),
                                 nn.Linear(nz//4,1),
                                 nn.Sigmoid())
    
    def forward(self, x):
        output = self.adv(x)
        return output
        
        