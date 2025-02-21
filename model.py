import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import cuda
from numbers import Number
import sys 
import torch as np
import torchvision.models as models

class IBNet(nn.Module): 

    def __init__(self, K=256,  model_name='CNN4', model_mode='stochastic',mean_normalization_flag=False,std_normalization_flag=False):
        if model_mode not in ['stochastic','deterministic']:
            print("The model mode can be either stochastic or deterministic!")
            sys.exit()
               
        super(IBNet, self).__init__()
        self.K = K
        self.model_mode= model_mode
        self.model_name = model_name
        self.mean_normalization_flag = mean_normalization_flag
        self.std_normalization_flag = std_normalization_flag

        if model_name == 'CNN4':
            print("CNN 4 is chosen!")
            self.encode = nn.Sequential(
            nn.Conv2d(3, 8, 5, 1, 2),
            nn.Conv2d(8, 8, 5, 1, 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(1024, 256),  # if CIFAR10 dataset is used
            #nn.Linear(21904,256), # if INTEL dataset is used
            nn.LeakyReLU(),
            nn.Linear(256, 2*self.K),
            )
        elif model_name == "Resnet18":
            print("Resnet 18 is chosen!")
            self.encode = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1") #getattr(models, "resnet18")(weights=weights)
            input_last = self.encode.fc.in_features
            self.encode.fc = nn.Linear(input_last, 2*self.K)
        elif model_name == "Resnet50":
            print("Resnet 50 is chosen!")
            self.encode = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1") #getattr(models, "resnet18")(weights=weights)
            input_last = self.encode.fc.in_features
            self.encode.fc = nn.Linear(input_last, 2*self.K)
 
        self.decode = nn.Sequential( nn.Linear(self.K, 10) )

        self.layer_norm1 = nn.Sequential(torch.nn.LayerNorm([self.K],elementwise_affine=True))
        self.layer_norm2 = nn.Sequential(torch.nn.LayerNorm([self.K],elementwise_affine=True))
        
        self.layer_norm_variance = nn.Sequential(nn.Linear(1,1,bias=False))

        
        
    def forward(self, x, num_sample=1):

        statistics = self.encode(x)
        mu  = statistics[:,:self.K]
        std = F.softplus(statistics[:,self.K:]-5,beta=1)

        if self.mean_normalization_flag:
            mu = self.layer_norm1(mu)       # Layer normalization
        if self.std_normalization_flag:
            std = 0.5*self.layer_norm2(std) # Layer normalization

        if self.model_mode == 'stochastic':
            encoding = self.reparametrize_n(mu,std,num_sample)
        else:
            encoding = mu

        logit = self.decode(encoding)

        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        if n != 1 :
                mu - self.expand(mu,n)
                std = self.expand(std,n)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))
        return mu + eps * std

    def expand(self,v,n):
            if isinstance(v, Number): return torch.Tensor([v]).expand(n, 1)
            else: return v.expand(n, *v.size())
   
    def weight_init(self):
        if self.model_name == "CNN4":
            for child in self.children():
                for ii in range(len(child)):
                    if type(child[ii])==nn.LayerNorm:
                        child[ii].track_running_stats = False

            xavier_init(self.encode)
            self.layer_norm1[0].weight.data.fill_(1)
            self.layer_norm1[0].bias.data.fill_(0)
            self.layer_norm_variance[0].weight.data.fill_(self.K**0.5/4)
            
        xavier_init(self.decode)
    
def xavier_init(ms):
    torch.manual_seed(0)
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
