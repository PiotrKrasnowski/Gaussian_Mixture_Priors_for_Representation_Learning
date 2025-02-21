########################################################################################
#                                                                                      #
#                            Code for ICLR 2025 submission                             #
#               Generalization Guarantees for Representation Learning                  #
#                    via Data-Dependent Gaussian Mixture Priors                        #
#                                                                                      #
########################################################################################

This archive contains code and instructions for experiments in the submitted paper titled:
Generalization Guarantees for Representation Learning via Data-Dependent Gaussian Mixture Priors


#################
# Requirements  #
#################
The code is tested with Python 3.10.0 on Linux. Please see requirements.txt.


##################
# Code structure #
##################
 main.py            # File to run to reproduce presented experiments in the paper

 analyze_results.py # File which evaluates results and produces figures
 datasets.py        # File which specifies and creates data loaders 
 model.py           # File which specifies the prediction model
 solver.py          # File which specifies the training and testing algorithms
 utils.py           # File which defines some auxilary functions

 datasets/          # Directory with datasets
 results/           # Directory with training results
 figures/           # Directory with figures produced by the script analyze_results.py


#######################
# Running experiments #
#######################  
 Download CIFAR10 data from https://www.cs.toronto.edu/~kriz/cifar.html or INTEL data from https://www.kaggle.com/datasets/puneet6060/intel-image-classification.
 Extract the files and place under the right folder in the directory datasets/ .
 To reproduce results in the paper: 
   - Launch the script main.py . The selected training parameters and training results will appear in the right folder in the directory results/ .
   - Update the path to the results in the script analyze_results.py and then launch the script. The produced figures will appear in the right folder in the directory figures/ . 


#######################
# References #
####################### 
This code is based on the Variation Information Bottleneck implementation (VIB) of https://github.com/1Konny/VIB-pytorch.git. The code, however, has been modified substantially both for VIB and for our proposed objective function.