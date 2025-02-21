import numpy as np
import torch
from solver import Solver
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

global_parameters = {
    "repetitions":   5,                            # total number of repetitions
    "beta":          list(np.logspace(-5, 0, 11)), # regularization parameter
    "loss_id":       [1,2,3],                      # 0 - no regularizarion, 1 - VIB with standard Gaussian prior,
                                                   # 2 - lossy CDVIB,       3 - lossy GMVIB (ours)
    "centers_num":   [5],                          # number of centers in loss 2 and loss 3
    "mov_coeff_mul": [1e-2],                       # coefficient for smooth change of mean  of prior centers
    "mov_coeff_alpha": [1e-2],                     # coefficient for smooth change of alpha parameter 
    "mov_coeff_var": [5e-4],                       # coefficient for smooth change of variance of prior centers
    "results_dir":   'results',                    # dir to results
    "figures_dir":   'figures',                    # dir to figures
    "save_results":  True,                         # save training and testing results
    "save_model":    True                          # save trained models
}


solver_parameters = {
    "cuda":          True,
    "seed":          0,                   # used to re-initialize dataloaders 
    "epoch_num":     200,                 # number of training epochs                                     
    "lr":            1e-4,                # learning rate
    "K":             64,                  # dimension of encoding Z    
    "num_avg":       12,                  # number of samplings Z
    "batch_size":    128,                 # batch size
    "dataset":       'CIFAR10',           # dataset; options: "CIFAR10" and "INTEL"
    "dset_dir":      'datasets',          # dir with datasets
    "model_name":    'CNN4',              # encoder architectures; options: "CNN4", "Resnet18", "Resnet50" 
    "model_mode": 'stochastic',           # the model mode is either 'stochastic' or 'deterministic'
    "mean_normalization_flag": False,
    "std_normalization_flag": False,
    "perturbation_flag": True,
    "noise_level": 0.2,                   # the total noise added for updating the centers at each iteration
}

loss_list = []
beta_list = []
training_accuracy_list = []
test_accuracy_list = []

def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    # create a new folder to store our results and figures
    timestamp = time.strftime("%Y%m%d-%H%M%S")
        
    results_path = global_parameters["results_dir"] + '/results_' +  timestamp
    if not os.path.exists(results_path):
         os.makedirs(results_path)
    figures_path = global_parameters["figures_dir"] + '/figures_' +  timestamp
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    # save global and initial solver parameters to a file
    if not os.path.isfile(results_path + '/global_parameters.npz'): 
        np.savez(results_path + '/global_parameters', global_parameters = global_parameters)          

    # training of models with various parameters
    for rep in range(global_parameters["repetitions"]):
        for beta in global_parameters["beta"]:
            for loss_id in global_parameters["loss_id"]:
                for centers_num in global_parameters["centers_num"]:
                    for mov_coeff_mul in global_parameters["mov_coeff_mul"]: 
                        for mov_coeff_var in global_parameters["mov_coeff_var"]:    
                            for mov_coeff_alpha in global_parameters["mov_coeff_alpha"]:                 
                                # update selected solver parameters
                                solver_parameters["seed"]          = rep
                                solver_parameters["beta"]          = beta
                                solver_parameters["loss_id"]       = loss_id
                                solver_parameters["mov_coeff_mul"] = mov_coeff_mul
                                solver_parameters["mov_coeff_alpha"] = mov_coeff_alpha
                                solver_parameters["mov_coeff_var"] = mov_coeff_var
                                solver_parameters["centers_num"]   = centers_num
                                solver_parameters["timestamp"] = timestamp
                                
                                filename = '/_results_LossID_{}_Beta_{:,.0e}_NumCent_{}_MovCoeff_{}_rep_{}_.npz'.format(loss_id,beta,centers_num,mov_coeff_mul,rep)
                                if os.path.isfile(results_path+filename):
                                    continue

                                # create a model and train
                                net = Solver(solver_parameters)
                                net.train_full()

                                loss_list.append(loss_id)
                                beta_list.append(beta)
                                training_accuracy_list.append(net.train1_train_dataset["accuracy"] )
                                test_accuracy_list.append(net.train1_test_dataset["accuracy"] )

                                print('Dataset: {}'.format(solver_parameters["dataset"]))
                                print('Model: {}'.format(solver_parameters["model_name"]))
                                print('Loss_ID: {}'.format(loss_list))  
                                print('Betas: {:}'.format(beta_list))
                                print('Training_accuracies: {:}'.format(training_accuracy_list))
                                print('Test_accuracies: {:}'.format(test_accuracy_list))
                                
                                if global_parameters["save_results"]:
                                    # extract interesting statistics and save to a .npz file
                                    np.savez(results_path+filename, 
                                                solver_parameters    = solver_parameters,
                                                train1_train_dataset = net.train1_train_dataset, 
                                                train1_test_dataset  = net.train1_test_dataset,
                                                
                                                train_accuracy_list = torch.stack(net.train_accuracy_list).cpu().numpy(), 
                                                test_accuracy_list  = torch.stack(net.test_accuracy_list).cpu().numpy(),
                                                epoch_tested_list   = np.stack(net.epoch_tested_list),

                                                moving_average_mul   = net.moving_mean_multiple_tensor.cpu().numpy(),
                                                moving_variance_mul  = net.moving_variance_multiple_tensor.cpu().numpy(),
                                                moving_alpha_mul     = net.moving_alpha_multiple_tensor.cpu().numpy(),
                                                counter              = rep
                                        )
                                    
                                    if global_parameters["save_model"]: torch.save(net.IBnet, results_path+'/_trained_model_LossID_{}_Beta_{:,.0e}_NumCent_{}_MovCoeff_{}_rep_{}_.pth'.format(loss_id,beta,centers_num,mov_coeff_mul,rep))
                                del net 
                                
    print('Dataset: {}'.format(solver_parameters["dataset"]))
    print('Model: {}'.format(solver_parameters["model_name"]))
    print('Loss_ID: {}'.format(loss_list))  
    print('Betas: {:}'.format(beta_list))
    print('Training_accuracies: {:}'.format(training_accuracy_list))
    print('Test_accuracies: {:}'.format(test_accuracy_list))

if __name__ == "__main__":
    main()
