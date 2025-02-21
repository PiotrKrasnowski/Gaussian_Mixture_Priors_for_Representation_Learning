import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from datasets import return_data
from model import IBNet
import math
import scipy.stats as st
import random
import sys 
import os

from utils import cuda,set_constants_utils
from utils import KL_DG_DG,KL_lossy_DG_DG
from utils import KL_DG_DGM_prod, KL_lossy_DG_DGM_var
from utils import contribution_Gaussian_to_GM,contribution_lossy_Gaussian_to_GM
 
class Solver(object):

    def __init__(self, args):
        self.args = args

        # Training parameters
        self.cuda        = (args["cuda"] and torch.cuda.is_available())
        self.seed        = args["seed"]
        self.epoch_1     = args["epoch_num"]
        self.batch_size  = args["batch_size"]
        self.lr          = args["lr"]
        self.eps         = 1e-15
        self.global_iter = 0
        self.global_epoch = 0

        # Model parameters
        self.K      = args["K"]
        self.beta   = args["beta"]
        self.num_avg = args["num_avg"]
        self.loss_id = args["loss_id"] 
        self.model_mode = args["model_mode"]
        self.model_name = args["model_name"]

        if self.model_mode == 'deterministic' and self.loss_id not in [0,1,2,3]:
            print("wrong loss!")
            sys.exit()
    
        if self.num_avg > 1 and self.model_mode == 'deterministic':
            self.num_avg = 1 

        # CDVIB parameters
        self.M      = args["centers_num"] 
        self.moving_coefficient_mean    = args["mov_coeff_mul"]
        self.moving_coefficient_alpha = args["mov_coeff_alpha"]
        self.moving_coefficient_var = args["mov_coeff_var"]
        self.temp_coeff = 2
        self.coeff_normalization = 10

        # Network
        self.IBnet = cuda(IBNet(K= self.K,model_name = self.model_name,model_mode = self.model_mode,mean_normalization_flag = args["mean_normalization_flag"],std_normalization_flag=args["std_normalization_flag"]), self.cuda)
        self.IBnet.weight_init()
         
        # Dataset size and name
        if args["dataset"] == 'MNIST':   
            self.train_size = 50000
            self.class_size = 10
            self.HY = math.log(self.class_size,2)
        elif args["dataset"] == 'CIFAR10': 
            self.train_size = 50000
            self.class_size = 10
            self.HY = math.log(self.class_size,2)
        elif args["dataset"] == 'INTEL':
            self.train_size = 13986
            self.class_size = 6
            self.HY = math.log(self.class_size,2)

        # CDVIB and GMVIB initializations
        self.lossy_variance = 4     
        self.lossy_variance_var = 0.5
        self.decaying_lossy_constants = 1
        self.center_initialization_args = 'scattered'
        self.dataset_args = args["dataset"]
        self.dset_dir_args = args["dset_dir"]

        # Optimizer
        self.optim    = optim.Adam(self.IBnet.parameters(),lr=self.lr,betas=(0.5,0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optim,gamma=0.97)

        # Dataset Loading
        self.data_loader = return_data(args["dataset"], args["dset_dir"], args["batch_size"])
              
        # Other
        self.sqrt2pi = cuda(torch.sqrt(torch.tensor(2*torch.pi)), self.cuda) 
        self.matrix_A = cuda(torch.normal(0, 1, size=(2,self.K)),self.cuda) #Projection Matrix
        self.timestamp = args["timestamp"]

        self.loss_choice = 'prod_var'                 # This choice can be either 'prod_var', or 'prod', or 'var'
        self.update_centers_rule = "D_KL"             # This choice can be either 'expectation', or 'D_KL'
        self.update_centers_coeff_mode = "not_prop"   # This will make the updates proportional to either only self.M ('prop_M'), 
                                                      # or only number of importance ('prop_b') or both ('prop_M_b') 
                                                      # or none of them ('not_prop') or ('prop_b_logistic')     
        self.epoch_tested_list=[]
        self.train_accuracy_list=[]
        self.test_accuracy_list=[]
        self.beta_KL_list = []
        self.beta_exp_list = []
        self.inner_prod_std_list = []
        self.sigma_term_std_list = []
        self.alpha_std_list = []
        
        self.running_mean = cuda(torch.ones(self.class_size,self.K),self.cuda) # [C,K]
        self.coeff_running_mean_new = 0.1

        self.noise_level = args["noise_level"]
        self.average_distance_centers_means = 0
        self.initialize_centers(self.center_initialization_args,self.dataset_args,self.dset_dir_args)
        self.perturbation_flag = args["perturbation_flag"]

        set_constants_utils(self.M,self.eps,self.lossy_variance,self.cuda)

   
    def train_full(self):
        print('Loss ID:{}, Beta:{:.0e}, NumCent:{}, MovCoeff:{}, Mode:{}, Loss Choice:{}, update_center_rule:{}'.format(self.loss_id,self.beta,self.M,self.moving_coefficient_mean,self.model_mode,self.loss_choice,self.update_centers_rule))

        if self.model_mode == 'deterministic':
            print("Deterministic mode activated!")

        ##################
        # First training #
        ##################
        # reinitialize seeds
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # train
        self.train_step1()

        print("Training 1 is finished")
      
        self.epoch_tested_list.append(self.epoch_1+1)
        self.test('train')

        self.train_accuracy_list.append(self.accuracy.detach())
        
        print('Loss ID:{}, Beta:{:.0e}, NumCent:{}, MovCoeff:{}, Mode:{}, Loss Choice:{}, update_center_rule:{}, lossy_var{}, lossy_var_variance{}'.format(self.loss_id,self.beta,self.M,self.moving_coefficient_mean,self.model_mode,self.loss_choice,self.update_centers_rule,self.lossy_variance,self.lossy_variance_var))
        print('Final training accuracy: {:.3f}'.format(self.accuracy))      
        
        self.train1_train_dataset = {}
        self.train1_train_dataset["accuracy"] = self.accuracy.cpu().numpy()
        self.train1_train_dataset["accuracy_confidence_low"] = self.accuracy_confidence_intervals[0]
        self.train1_train_dataset["accuracy_confidence_high"] = self.accuracy_confidence_intervals[1]

        self.train1_train_dataset["log_likelihood"] = self.log_likelihood.cpu().numpy()
        self.train1_train_dataset["log_likelihood_confidence_low"] = self.log_likelihood_confidence_intervals[0]
        self.train1_train_dataset["log_likelihood_confidence_high"] = self.log_likelihood_confidence_intervals[1]

        self.train1_train_dataset["izy_bound"] = self.izy_bound.cpu().numpy()
        self.train1_train_dataset["izx_bound"] = self.izx_bound.cpu().numpy()
        self.train1_train_dataset["reg_complexity"] = self.reg_complexity.cpu().numpy()

        self.test('test')
        print('Final test accuracy: {:.3f}'.format(self.accuracy))

        self.test_accuracy_list.append(self.accuracy.detach())
                    
        self.train1_test_dataset = {}
        self.train1_test_dataset["accuracy"] = self.accuracy.cpu().numpy()
        self.train1_test_dataset["accuracy_confidence_low"] = self.accuracy_confidence_intervals[0]
        self.train1_test_dataset["accuracy_confidence_high"] = self.accuracy_confidence_intervals[1]

        self.train1_test_dataset["log_likelihood"] = self.log_likelihood.cpu().numpy()
        self.train1_test_dataset["log_likelihood_confidence_low"] = self.log_likelihood_confidence_intervals[0]
        self.train1_test_dataset["log_likelihood_confidence_high"] = self.log_likelihood_confidence_intervals[1]

        self.train1_test_dataset["izy_bound"] = self.izy_bound.cpu().numpy()
        self.train1_test_dataset["izx_bound"] = self.izx_bound.cpu().numpy()
        self.train1_test_dataset["reg_complexity"] = self.reg_complexity.cpu().numpy()
        print("Testing 1 is finished")


    def train_step1(self):
        
        self.set_mode('train')

        self.izy_relevance_epochs, self.izx_complexity_epochs, self.reg_complexity_epochs = [], [], []

        self.first_iter_epoch_flag = False       

        optimization_mode = 'joint'
        for e in range(self.epoch_1):

            self.temp_beta_KL = 0
            self.temp_beta_exp = 0
            self.temp_inner_prod_std = 0
            self.temp_sigma_term_std = 0

            self.global_epoch += 1

            #### Perturbing the centers
            if self.perturbation_flag:
                ### reseting all centers whose alpha>0.95 for one of the centers ####
                best_indices = (self.moving_alpha_multiple_tensor.view(self.class_size,self.M).max(dim=1)[1]+ self.M * cuda(torch.arange(self.class_size),self.cuda)).view(-1,1).repeat(1,self.M).view(-1)
                Uni_center_labels = (self.moving_alpha_multiple_tensor.view(self.class_size,self.M).max(dim=1)[0]> 0.95)

                eps = cuda(torch.randn(self.moving_mean_multiple_tensor.view(self.class_size,self.M,self.K)[Uni_center_labels,:,:].size()), self.cuda)
                self.moving_mean_multiple_tensor.view(self.class_size,self.M,self.K)[Uni_center_labels,:,:]= self.moving_mean_multiple_tensor[best_indices,:].view(self.class_size,self.M,self.K)[Uni_center_labels,:,:] + eps* self.average_distance_centers_means/np.sqrt(self.K)/2
                self.moving_variance_multiple_tensor.view(self.class_size,self.M,self.K)[Uni_center_labels,:,:]= self.moving_variance_multiple_tensor[best_indices,:].view(self.class_size,self.M,self.K)[Uni_center_labels,:,:] 
                
                self.moving_alpha_multiple_tensor.view(self.class_size,self.M)[Uni_center_labels,:] =1/self.M

                bad_indices = (self.moving_alpha_multiple_tensor<1/self.M/100).nonzero().squeeze(-1)
                eps = cuda(torch.randn(self.moving_mean_multiple_tensor[bad_indices,:].size()), self.cuda)
                
                self.moving_mean_multiple_tensor[bad_indices,:]= (self.moving_mean_multiple_tensor[best_indices,:])[bad_indices,:]+ eps* self.average_distance_centers_means/np.sqrt(self.K)/2
                self.moving_variance_multiple_tensor[bad_indices,:]= (self.moving_variance_multiple_tensor[best_indices,:])[bad_indices,:]

                ### reseting all centers whose alpha<0.95 for one of the centers ####
                avg_diff_centers = (self.moving_mean_multiple_tensor.view(self.class_size,self.M,1,self.K)-self.moving_mean_multiple_tensor.view(self.class_size,1,self.M,self.K)).pow(2).sum(-1).sqrt().div((self.M-1)/self.M).mean()
                if avg_diff_centers < self.average_distance_centers_means/5:
                    eps = cuda(torch.randn(self.moving_mean_multiple_tensor.size()), self.cuda)
                    self.moving_mean_multiple_tensor += eps* self.average_distance_centers_means/np.sqrt(self.K)/2
                
            self.lossy_variance *= self.decaying_lossy_constants                 # The variance used for lossy computations
            self.lossy_variance_var *= self.decaying_lossy_constants             # The offset variance used for lossy computations of the term related to variances
            set_constants_utils(self.M,self.eps,self.lossy_variance,self.cuda)

            for idx, (images,labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1

                x = Variable(cuda(images, self.cuda))
                y = Variable(cuda(labels, self.cuda))
                (mu, std), logit = self.IBnet(x)

                class_loss = F.cross_entropy(logit,y).div(math.log(2))
                info_loss = self.regularization(mu, std, y, self.loss_id) 
                total_loss = class_loss + self.beta*info_loss
                
                self.first_iter_epoch_flag = False

                ###### Computing mean of the current batch and it's difference with average mean
                if optimization_mode != 'no_batches':
                    self.optim.zero_grad()
                    total_loss.backward()
                    self.optim.step()

                if optimization_mode != 'no_centers' and self.loss_id not in [0,1,2,3,8]:
                    center_means = self.moving_mean_multiple_tensor
                    center_variacnces = self.moving_variance_multiple_tensor

                    eps = Variable(cuda(center_variacnces.data.new(center_variacnces.size()).normal_(), cuda))
                    if self.loss_id == 23:
                        logit_centers = self.IBnet.decode(center_means+eps* torch.clamp(center_variacnces- self.lossy_variance_var ,min = 0).sqrt())
                    else:
                        logit_centers = self.IBnet.decode(center_means+eps*center_variacnces.sqrt())
                    
                    y_centers = cuda(torch.arange(self.class_size).unsqueeze(-1).repeat(1,self.M).view(self.class_size *self.M), self.cuda)

                    centers_loss = F.cross_entropy(logit_centers,y_centers).div(math.log(2))
                    if optimization_mode == 'no_batches':
                        total_loss = centers_loss + self.beta*info_loss
                    elif optimization_mode == 'joint':
                        total_loss = centers_loss 

                    self.optim.zero_grad()
                    total_loss.backward()
                    self.optim.step()

            self.scheduler.step()
            print(e)


    def set_mode(self,mode='train'):
        if mode == 'train' :
            self.IBnet.train()
            self.mode = 'train'
        elif mode == 'eval' :
            self.IBnet.eval()
            self.mode='eval'
        else : raise('mode error. It should be either train or eval')


    def update_centers_loss_3(self,mu,var,y,gamma_values_KL,gamma_values_expectation = 0):
        
        gamma_indices = cuda(torch.arange(self.M).reshape((1,self.M)), self.cuda).repeat(y.shape[0],1) + self.M * y[:,None]#.detach()
        
        moving_centers_multiple_gamma_KL = cuda(torch.zeros(y.shape[0], self.class_size * self.M), self.cuda).scatter_(1, gamma_indices, gamma_values_KL)#.detach()
        moving_centers_multiple_gamma_expectation = cuda(torch.zeros(y.shape[0], self.class_size * self.M), self.cuda).scatter_(1, gamma_indices, gamma_values_expectation)#.detach()
        
        beta_values_KL = moving_centers_multiple_gamma_KL.sum(dim=0)
        beta_values_expectation = moving_centers_multiple_gamma_expectation.sum(dim=0)
        beta_values = (beta_values_KL+beta_values_expectation)/2
        
        moving_centers_multiple_gamma_alt =  (2*moving_centers_multiple_gamma_KL+moving_centers_multiple_gamma_expectation)/3
        beta_values_alt = (2*beta_values_KL+beta_values_expectation)/3
        center_weighted_mean_batch_normalized = torch.matmul(moving_centers_multiple_gamma_alt.transpose(0,1), mu) / (beta_values_alt.unsqueeze(-1).repeat(1,self.K)+self.eps)#.detach()                            # size [C*M,K] 
        
        temp_coeff = (beta_values_alt < 1/self.M/100).int().view(-1,1).repeat(1,self.K)
        best_indices = (beta_values_alt.view(self.class_size,self.M).max(dim=1)[1]+ self.M * cuda(torch.arange(self.class_size),self.cuda)).view(-1,1).repeat(1,self.M).view(-1)
        center_weighted_mean_batch_normalized = (1-temp_coeff)*center_weighted_mean_batch_normalized + temp_coeff * center_weighted_mean_batch_normalized[best_indices,:]       
       
        if self.update_centers_coeff_mode == 'prop_M_b':
            coeff_old = 1 - self.moving_coefficient_mean * self.M * beta_values.unsqueeze(1).repeat(1,self.K)
            coeff_new = 1 - coeff_old

            coeff_old_alpha = coeff_old[:,0]
            coeff_new_alpha = coeff_new[:,0]
        elif self.update_centers_coeff_mode == 'prop_b':
            coeff_old = 1 - self.moving_coefficient_mean * beta_values.unsqueeze(1).repeat(1,self.K)
            coeff_new = 1-coeff_old 

            coeff_old_alpha = coeff_old[:,0]
            coeff_new_alpha = coeff_new[:,0]
        elif self.update_centers_coeff_mode == 'prop_M':
            coeff_old = 1 - self.moving_coefficient_mean * self.M 
            coeff_new = 1 - coeff_old  

            coeff_old_alpha = coeff_old
            coeff_new_alpha = coeff_new
        elif self.update_centers_coeff_mode == 'not_prop':
            coeff_old = 1 - self.moving_coefficient_mean *(beta_values.view(self.class_size,self.M).sum(-1)>4).int().view(self.class_size,1,1).repeat(1,self.M,self.K).view(self.class_size*self.M,self.K)#*(beta_values > 1/self.M/100).int().view(-1,1).repeat(1,self.K)
            coeff_new = 1 - coeff_old   

            coeff_old_alpha = 1 - self.moving_coefficient_alpha *(beta_values.view(self.class_size,self.M).sum(-1)>4).int().view(self.class_size,1).repeat(1,self.M).view(self.class_size*self.M) #*(beta_values > 1/self.M/100).int()
            coeff_new_alpha = 1 - coeff_old_alpha
        elif self.update_centers_coeff_mode == 'prop_b_logistic':
            coeff_logistic = 1/(1+(-self.temp_coeff*(beta_values.view(self.class_size,self.M).sum(-1)-self.batch_size/(3*self.class_size))).exp()) #1/(1+exp(-c*(x-x_0)))
            coeff_old = 1 - self.moving_coefficient_mean*coeff_logistic.unsqueeze(-1).unsqueeze(-1).repeat(1,self.M,self.K).view(self.class_size*self.M,self.K)
            coeff_new = 1-coeff_old 

            coeff_old_alpha = coeff_old[:,0]
            coeff_new_alpha = coeff_new[:,0]

        else:
            print("Invaid center coefficient update choice!")
            sys.exit()
        
        # This is like SGLD, adding noise to update process
        eps = self.noise_level * cuda(torch.randn(center_weighted_mean_batch_normalized.size()), self.cuda) / np.sqrt(self.K)
        self.moving_mean_multiple_tensor *= coeff_old
        self.moving_mean_multiple_tensor += coeff_new * center_weighted_mean_batch_normalized + eps
  
        if self.model_mode == 'stochastic':
            center_weighted_var = torch.matmul(moving_centers_multiple_gamma_KL.transpose(0,1), var)
            center_weighted_var_batch_normalized_alternative = center_weighted_var / (beta_values_KL.unsqueeze(-1).repeat(1,self.K)+self.eps)
            
            temp_coeff = (beta_values < 1/self.M/100).int().view(-1,1).repeat(1,self.K)
            best_indices = (beta_values_alt.view(self.class_size,self.M).max(dim=1)[1]+ self.M * cuda(torch.arange(self.class_size),self.cuda)).view(-1,1).repeat(1,self.M).view(-1)
         
            center_weighted_var_batch_normalized_alternative = (1-temp_coeff)*center_weighted_var_batch_normalized_alternative + temp_coeff * center_weighted_var_batch_normalized_alternative[best_indices,:]

            # Updating the variance
            coeff_var = self.moving_coefficient_var*(beta_values_KL.view(self.class_size,self.M).sum(-1)>4).int().view(self.class_size,1,1).repeat(1,self.M,self.K).view(self.class_size*self.M,self.K)
            self.moving_variance_multiple_tensor *= (1-coeff_var)
            self.moving_variance_multiple_tensor += coeff_var * center_weighted_var_batch_normalized_alternative
            
        self.moving_alpha_multiple_tensor  *= coeff_old_alpha
        self.moving_alpha_multiple_tensor  += coeff_new_alpha* (beta_values.view(self.class_size,self.M) / (beta_values.view(self.class_size,self.M).sum(-1).unsqueeze(-1)+self.eps)).view(self.class_size*self.M)
        # We normalize the updated alpha_r
        self.moving_alpha_multiple_tensor = (self.moving_alpha_multiple_tensor.view(self.class_size,self.M) / self.moving_alpha_multiple_tensor.view(self.class_size,self.M).sum(1).unsqueeze(-1)).view(self.class_size*self.M)
     
        return
                    

    def regularization(self, mu, std, y, idx, reduction = 'mean'):
        if idx == 0:   # no regularization
            info_loss = cuda(torch.tensor(0.0),self.cuda)
        elif idx == 1: # standard VIB regularization with Gaussian prior
            info_loss = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum().div(math.log(2))          
        elif idx == 2: # CDVIB
            if self.model_mode  == 'stochastic':
                var = std.pow(2)
            else:
                var = self.lossy_variance * cuda(torch.ones_like(mu),self.cuda)
            
            # select closest centers
            centers_mean_label = self.moving_mean_multiple_tensor.view(self.class_size,self.M,self.K)[y,:,:]     # size [B, M, K]
            centers_var_label = self.moving_variance_multiple_tensor.view(self.class_size,self.M,self.K)[y,:,:] # size [B, M, K]
            centers_selected_ind = KL_DG_DG(mu.unsqueeze(1).repeat(1,self.M,1),var.unsqueeze(1).repeat(1,self.M,1),\
                                                                    centers_mean_label,centers_var_label).argmin(dim=1) \
                                   +y*self.M      # size [B]
            center_mean_selected = self.moving_mean_multiple_tensor[centers_selected_ind,:]         # size [B, K]
            center_var_selected = self.moving_variance_multiple_tensor[centers_selected_ind,:]     # size [B, K]
                
            info_loss = KL_lossy_DG_DG(mu,var,center_mean_selected,center_var_selected).sum()  # size [1]

            # update centers
            if self.mode == 'train':
                centers_selected_hot_encoding = torch.nn.functional.one_hot(centers_selected_ind,self.class_size*self.M).type(torch.float).detach() # size [B,C*M]
                center_count = centers_selected_hot_encoding.sum(0)
                
                center_mean_batch = torch.matmul(centers_selected_hot_encoding.transpose(0,1), mu).detach()                            # size [C*M,K] 
                self.moving_mean_multiple_tensor     *= (1 - self.moving_coefficient_mean * self.M * center_count.unsqueeze(1).repeat(1,self.K))
                self.moving_mean_multiple_tensor     += self.moving_coefficient_mean * self.M * center_mean_batch       
                
                if self.model_mode  == 'stochastic':
                    center_var_batch = torch.matmul(centers_selected_hot_encoding.transpose(0,1), var).detach()                    # size [C*M,K]
                    self.moving_variance_multiple_tensor *= (1 - self.moving_coefficient_mean * self.M * center_count.unsqueeze(1).repeat(1,self.K))
                    self.moving_variance_multiple_tensor += self.moving_coefficient_mean * self.M * center_var_batch
        
        elif idx == 3: # GMVIB (Gaussian-Mixture VIB)
            # select corresponding components
            centers_mean_label = self.moving_mean_multiple_tensor.view(self.class_size,self.M,self.K)[y,:,:]     # size [B, M, K]
            centers_var_label = self.moving_variance_multiple_tensor.view(self.class_size,self.M,self.K)[y,:,:] # size [B, M, K]
            centers_alpha = self.moving_alpha_multiple_tensor.view(self.class_size,self.M)[y,:] # [B,M]
            
            ######### Adpative lossy
            std_expanded = std.unsqueeze(1).repeat(1,self.M,1)
            mu_expanded = mu.unsqueeze(1).repeat(1,self.M,1)
            c1 = 2*self.M/np.sqrt(2)
            c2 = 1/np.sqrt(2)
            q1 = 1
            q2 = 1
            lossy_variance_temp = (mu_expanded-centers_mean_label).pow(2).sum(-1).div(math.log(2)).std(dim=-1).mean().detach().div(2*c1)
            ratio_temp = (std_expanded.pow(2)+self.lossy_variance_var)/(centers_var_label+self.lossy_variance_var)
            std_std_avg = (-ratio_temp+ratio_temp.log()).sum(-1).div(2).div(math.log(2)).std(dim=-1).mean()
            lossy_variance_var_temp = self.lossy_variance_var-0.05 * (c2+std_std_avg.detach())

            self.lossy_variance= (1-q1) * lossy_variance_temp+ q1 *self.lossy_variance
            self.lossy_variance_var= (1-q2)* torch.max(lossy_variance_var_temp,lossy_variance_var_temp*0+1e-5)+q2*self.lossy_variance_var
            
            self.average_distance_centers_means *= 0.95
            self.average_distance_centers_means += 0.05 *  ((mu_expanded-centers_mean_label).pow(2).sum(-1).min(dim=-1)[0].sqrt()).mean().detach() 
            
            set_constants_utils(self.M,self.eps,self.lossy_variance,self.cuda)
            var_lossy = self.lossy_variance * cuda(torch.ones_like(mu),self.cuda)
            centers_var_label_lossy = self.lossy_variance * cuda(torch.ones_like(centers_var_label),self.cuda)
            var = std.pow(2)           
            
            # compute the regularizer term            
            info_prod = KL_DG_DGM_prod(mu, var_lossy, centers_alpha, centers_mean_label, centers_var_label_lossy).sum()
            info_lossy_var = KL_lossy_DG_DGM_var(mu, var+self.lossy_variance_var, centers_alpha, centers_mean_label, centers_var_label+self.lossy_variance_var).sum()
            info_loss = 25*(info_prod+info_lossy_var)/2

            #update centers
            if self.mode == 'train':
                # compute weights gamma
                gamma_values_expectation = contribution_Gaussian_to_GM(mu,var_lossy,centers_alpha,centers_mean_label,centers_var_label_lossy,'expectation') 
                gamma_values_KL_lossy = contribution_lossy_Gaussian_to_GM(mu,var+self.lossy_variance_var,centers_alpha,centers_mean_label,centers_var_label+self.lossy_variance_var,'D_KL') # [B,M]

                self.temp_beta_exp += gamma_values_expectation.std(dim=-1).mean().detach()*y.size(0)/50000
                self.temp_beta_KL += gamma_values_KL_lossy.std(dim=-1).mean().detach()*y.size(0)/50000

                self.update_centers_loss_3(mu.detach(),var.detach(),y.detach(),gamma_values_KL_lossy.detach(),gamma_values_expectation.detach())

        if reduction == 'sum':
            return info_loss
        elif reduction == 'mean':
            return info_loss.div(y.size(0))


    def test(self, dataloader_type):
        self.set_mode('eval')
        total_num, correct, cross_entropy, zx_complexity, reg_complexity = cuda(torch.tensor(0,dtype=torch.float64),self.cuda), cuda(torch.tensor(0,dtype=torch.float64),self.cuda), cuda(torch.tensor(0,dtype=torch.float64),self.cuda), cuda(torch.tensor(0,dtype=torch.float64),self.cuda),cuda(torch.tensor(0,dtype=torch.float64),self.cuda)
        for idx, (images,labels) in enumerate(self.data_loader[dataloader_type]):
            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))
            total_num += y.size(0)
            (mu, std), soft_logit = self.IBnet(x,self.num_avg)
            if self.num_avg > 1 and self.model_mode == 'stochastic':
                # cross entropy
                cross_entropy += sum(F.cross_entropy(soft_logit[j,:,:], y, reduction='sum').detach() for j in range(self.num_avg))
                # accuracy
                predictions = [soft_logit[j,:,:].max(1)[1] for j in range(self.num_avg)]
                correct += sum(torch.eq(predictions[j],y).float().sum().detach() for j in range(self.num_avg))
            else:
                # cross entropy
                cross_entropy += F.cross_entropy(soft_logit, y, reduction='sum').detach()
                # accuracy
                prediction = soft_logit.max(1)[1]
                correct += torch.eq(prediction,y).float().sum().detach()
            # complexity
            zx_complexity  += self.regularization(mu, std, y, 0, reduction = 'sum').detach()
            reg_complexity += self.regularization(mu, std, y, 0, reduction = 'sum').detach()
        # some statistics at the end of testing
        if self.model_mode == 'stochastic':
            const_den = self.num_avg
        else:
            const_den = 1

        self.accuracy      = correct/total_num/const_den
        self.log_likelihood = -cross_entropy/total_num/const_den
        self.izy_bound     = self.HY - cross_entropy/total_num/const_den/math.log(2)
        self.izx_bound     = zx_complexity/total_num
        self.reg_complexity = reg_complexity/total_num
        self.bootstrap_confidence_intervals(dataloader_type = dataloader_type + '_bootstrap', confidence = 0.95, sample_size=1000, repetitions=100)


    def initialize_centers(self,initialization_mode,dataset_name, dset_dir):
        if initialization_mode == 'fixed':
            self.moving_mean_multiple_tensor    = cuda(torch.zeros(self.class_size*self.M,self.K),self.cuda)
            self.moving_variance_multiple_tensor = cuda(torch.ones(self.class_size*self.M,self.K),self.cuda)
        elif initialization_mode == 'random':
            self.moving_mean_multiple_tensor    = 0.1*cuda(torch.randn(self.class_size*self.M,self.K),self.cuda)
            self.moving_variance_multiple_tensor = 0.8*cuda(torch.ones(self.class_size*self.M,self.K),self.cuda)+0.1*cuda(torch.randn(self.class_size*self.M,self.K).abs(),self.cuda)
        elif initialization_mode == 'scattered':
            self.moving_mean_multiple_tensor    = cuda(torch.zeros(self.class_size*self.M,self.K),self.cuda)
            self.moving_variance_multiple_tensor = cuda(torch.ones(self.class_size*self.M,self.K),self.cuda)
            
            # Create a new temporary dataloader with batch size B1=self.HY*200 for initializing the centers
            self.data_loader_temp = return_data(dataset_name,dset_dir, min(self.class_size*200,int(self.train_size/self.M/2),1000))

            counter_chosen_centers = 0

            temp_flag = True 
            classes_labels = cuda(torch.arange(self.class_size), self.cuda)

            while (counter_chosen_centers < self.M) and temp_flag == True:
                for idx, (images,labels) in enumerate(self.data_loader_temp['train']):
                    x = Variable(cuda(images, self.cuda))
                    y = Variable(cuda(labels, self.cuda))

                    if y.unique().size(0) != self.class_size:
                        break 

                    if counter_chosen_centers == 0:
                        running_M = 1
                    else:
                        running_M = counter_chosen_centers 

                    (mu, std), _ = self.IBnet(x) 

                    ###### Computing mean of the current batch and it's difference with average mean
                    temp_mean = cuda(torch.zeros(self.class_size,y.size(0),self.K),self.cuda)
                    temp_average_mean = temp_mean.scatter_(0,y.view(1,y.size(0),1).repeat(self.class_size,1,self.K),mu.unsqueeze(0).repeat(self.class_size,1,1)).sum(dim=1).squeeze(1)/(y.unique(return_counts=True)[1].unsqueeze(1)+self.eps)
                        
                    coeff_old_mean = 1 - self.coeff_running_mean_new *(y.unique(return_counts=True)[1]>4).int().view(self.class_size,1).repeat(1,self.K)
                
                    self.running_mean *= coeff_old_mean
                    self.running_mean += (1-coeff_old_mean) * temp_average_mean.detach()

                    centers_mean_label = self.moving_mean_multiple_tensor.view(self.class_size,self.M,self.K)[y,:running_M,:]     # size [B1, M, K]
                    centers_selected_ind = ((centers_mean_label-mu.unsqueeze(1).repeat(1,running_M,1)).pow(2)).sum(-1).argmin(dim=1) \
                                                +y*self.M      # size [B1]
                    center_mean_selected = self.moving_mean_multiple_tensor[centers_selected_ind,:]         # size [B1, K]
                    
                    min_distances = ((center_mean_selected-mu).pow(2)).sum(-1)  # size[B1]
                    
                    for ilabels in classes_labels:
                        index_i = (y==ilabels).nonzero(as_tuple=True)[0]
                        temp_i = index_i[torch.multinomial(min_distances[index_i],1)]
                        (self.moving_mean_multiple_tensor.view(self.class_size,self.M,self.K))[ilabels,counter_chosen_centers,:] = mu[temp_i].detach()
                        (self.moving_variance_multiple_tensor.view(self.class_size,self.M,self.K))[ilabels,counter_chosen_centers,:] = std.pow(2)[temp_i].detach()

                    counter_chosen_centers += 1
                    if counter_chosen_centers > self.M-1:
                        temp_flag = False
                        break
                
            del self.data_loader_temp

        # Gaussian Mixture parameters
        self.moving_alpha_multiple_tensor = cuda(torch.ones(self.class_size*self.M),self.cuda)/self.M

        ###### Update Centers
        self.first_iter_epoch_flag = True
   
        self.temp_beta_KL = 0
        self.temp_beta_exp = 0
        self.temp_inner_prod_std = 0
        self.temp_sigma_term_std = 0
        
        for idx, (images,labels) in enumerate(self.data_loader['train']):

            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))
            (mu, std), _ = self.IBnet(x)

            ###### Computing mean of the current batch and it's difference with average mean
            if y.unique().size(0) == self.class_size:
                temp_mean = cuda(torch.zeros(self.class_size,y.size(0),self.K),self.cuda)
                temp_average_mean = temp_mean.scatter_(0,y.view(1,y.size(0),1).repeat(self.class_size,1,self.K),mu.unsqueeze(0).repeat(self.class_size,1,1)).sum(dim=1).squeeze(1)/(y.unique(return_counts=True)[1].unsqueeze(1)+self.eps)
                coeff_old_mean = 1 - self.coeff_running_mean_new *(y.unique(return_counts=True)[1]>4).int().view(self.class_size,1).repeat(1,self.K)
        
                self.running_mean *= coeff_old_mean
                self.running_mean += (1-coeff_old_mean) * temp_average_mean.detach()
            
            centers_mean_label = self.moving_mean_multiple_tensor.view(self.class_size,self.M,self.K)[y,:,:]     # size [B, M, K]
            centers_var_label = self.moving_variance_multiple_tensor.view(self.class_size,self.M,self.K)[y,:,:] # size [B, M, K]
            centers_alpha = self.moving_alpha_multiple_tensor.view(self.class_size,self.M)[y,:] # [B,M]
            
            var_lossy = self.lossy_variance * cuda(torch.ones_like(mu),self.cuda)
            centers_var_label_lossy = self.lossy_variance * cuda(torch.ones_like(centers_var_label),self.cuda)
            var = std.pow(2)

            gamma_values_expectation = contribution_Gaussian_to_GM(mu,var_lossy,centers_alpha,centers_mean_label,centers_var_label_lossy,'expectation') 
           
            gamma_values_KL_lossy = contribution_lossy_Gaussian_to_GM(mu,var+self.lossy_variance_var,centers_alpha,centers_mean_label,centers_var_label+self.lossy_variance_var,'D_KL') # [B,M]
            self.update_centers_loss_3(mu.detach(),var.detach(),y.detach(),gamma_values_KL_lossy.detach(),gamma_values_expectation.detach())

            std_expanded = std.unsqueeze(1).repeat(1,self.M,1)
            mu_expanded = mu.unsqueeze(1).repeat(1,self.M,1)
            
            ratio_temp = (std_expanded.pow(2)+self.lossy_variance_var)/(centers_var_label+self.lossy_variance_var)
            std_std_avg = (-ratio_temp+ratio_temp.log()).sum(-1).div(2).div(math.log(2)).std(dim=-1).mean()
            
            self.temp_inner_prod_std += ((-(mu_expanded-centers_mean_label).pow(2)).sum(-1).div(math.log(2)).std(dim=-1).mean().detach().div(2*self.lossy_variance))*y.size(0)/50000
            self.temp_sigma_term_std += std_std_avg.detach() * y.size(0)/50000
            self.temp_beta_exp += gamma_values_expectation.std(dim=-1).mean().detach()*y.size(0)/50000
            self.temp_beta_KL += gamma_values_KL_lossy.std(dim=-1).mean().detach()*y.size(0)/50000

            self.first_iter_epoch_flag = False
         
        return

    
    def bootstrap_confidence_intervals(self, dataloader_type, confidence = 0.95, sample_size=1000, repetitions=100):
        accuracies, log_likelihoods_av = [], []
        # repeat repetitions time
        for rep in range(repetitions):
            total_num, correct, log_likelihood = cuda(torch.tensor(0,dtype=torch.float64),self.cuda),cuda(torch.tensor(0,dtype=torch.float64),self.cuda),cuda(torch.tensor(0,dtype=torch.float64),self.cuda)     
            # take randomly samples from the dataset
            for idx, (images,labels) in enumerate(self.data_loader[dataloader_type]):
                x = Variable(cuda(images, self.cuda))
                y = Variable(cuda(labels, self.cuda))
                total_num += y.size(0)
                _, soft_logit = self.IBnet(x, self.num_avg)
                if self.num_avg > 1 and self.model_mode == 'stochastic':
                    # log_likelihood
                    log_likelihood -= sum(F.cross_entropy(soft_logit[j,:,:], y, reduction='sum').detach() for j in range(self.num_avg))
                    # accuracy
                    predictions = [soft_logit[j,:,:].max(1)[1] for j in range(self.num_avg)]
                    correct += sum(torch.eq(predictions[j],y).float().sum().detach() for j in range(self.num_avg))
                else:
                    # log_likelihood
                    log_likelihood -= F.cross_entropy(soft_logit, y, reduction='sum').detach()
                    # accuracy
                    prediction = soft_logit.max(1)[1]
                    correct += torch.eq(prediction,y).float().sum().detach()
          
                # terminate if processed more than sample_size
                if idx*self.batch_size + 1 > sample_size: break
            # compute accuracy
            accuracy = correct/total_num/self.num_avg
            accuracies.append(accuracy)
            # compute average log_likelihood
            log_likelihood_av = log_likelihood/total_num/self.num_avg
            log_likelihoods_av.append(log_likelihood_av)

        # compute confidence intervals
        accuracy_confidence_intervals = st.norm.interval(alpha=confidence, loc=np.mean(torch.asarray(accuracies).cpu().numpy()), scale=st.sem(torch.asarray(accuracies).cpu().numpy()))
        log_likelihood_confidence_intervals = st.norm.interval(alpha=confidence, loc=np.mean(torch.asarray(log_likelihoods_av).cpu().numpy()), scale=st.sem(torch.asarray(log_likelihoods_av).cpu().numpy()))
        
        # output  
        self.accuracies_bootstrap = torch.asarray(accuracies)
        self.accuracy_confidence_intervals = accuracy_confidence_intervals
        self.log_likelihoods_bootstrap = torch.asarray(log_likelihoods_av)
        self.log_likelihood_confidence_intervals = log_likelihood_confidence_intervals