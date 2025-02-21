import os
import torch
import numpy as np
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from PIL import Image

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"
    
def load_dataset(dataset_dir, dataset_name):
    if dataset_name == "INTEL":
        dir = dataset_dir + "/INTEL/"
        # train data
        train_data, train_labels = [], []
        list_category = sorted(os.listdir(dir + "seg_train/seg_train/"))       
        for category_ind, category in enumerate(list_category):
            filelist = sorted(os.listdir(dir + "seg_train/seg_train/" + category + '/'))
            for file in filelist:
                image = np.array(Image.open(dir + "seg_train/seg_train/" + category + '/' + file))/255.
                if image.shape[0] != 150 or image.shape[1] != 150: 
                    continue
                train_data.append(np.moveaxis(image, -1, 0))
                train_labels.append(category_ind)

        train_data = np.stack(train_data)
        train_labels = np.array(train_labels, dtype = "int64")

        # test data
        test_data, test_labels = [], []
        list_category = sorted(os.listdir(dir + "seg_test/seg_test/"))       
        for category_ind, category in enumerate(list_category):
            filelist = sorted(os.listdir(dir + "seg_test/seg_test/" + category + '/'))
            for file in filelist:
                image = np.array(Image.open(dir + "seg_test/seg_test/" + category + '/' + file))/255.
                if image.shape[0] != 150 or image.shape[1] != 150: 
                    continue
                test_data.append(np.moveaxis(image, -1, 0))
                test_labels.append(category_ind)

        test_data = np.stack(test_data)
        test_labels = np.array(test_labels, dtype = "int64")
    return (train_data, train_labels), (test_data, test_labels)
        

class Intel_Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, transform):
        self.num_samples  = labels.size(0)
        self.labels       = labels
        self.dataset      = samples
        self.transform    = transform

    def __len__(self):
        return self.num_samples
    

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx]), self.labels[idx]


def return_data(name, dset_dir, batch_size):
    
    if 'CIFAR10' in name:
        transform = transforms.Compose([transforms.Resize((32,32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

        root = os.path.join(dset_dir,'CIFAR10')
        train_kwargs = {'root':root,'train':True,'transform':transform,'download':True}
        test_kwargs = {'root':root,'train':False,'transform':transform,'download':False}
        dset = CIFAR10
        train_data = dset(**train_kwargs)
        test_data = dset(**test_kwargs)

    elif 'INTEL' in name:
        
        (training_images, training_labels), (testing_images, testing_labels) = load_dataset(dset_dir, name)

        training_images = torch.tensor(training_images, dtype = torch.float32)
        training_labels = torch.tensor(training_labels, dtype = torch.int64) 

        testing_images = torch.tensor(testing_images, dtype = torch.float32)
        testing_labels = torch.tensor(testing_labels, dtype = torch.int64)

        transform = transforms.Compose([
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

        train_data = Intel_Dataset(training_images, training_labels, transform)
        test_data  = Intel_Dataset(testing_images, testing_labels, transform)

    else : raise UnknownDatasetError()

    data_loader = dict()
    
    test_loader = DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=1,
                                drop_last=False)
    
    test_bootstrap_loader = DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=1,
                                drop_last=True)

    train_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=1,
                                drop_last=True)
    
    train_bootstrap_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=1,
                                drop_last=True)

    data_loader['train'] = train_loader
    data_loader['train_bootstrap'] = train_bootstrap_loader
    data_loader['test'] = test_loader
    data_loader['test_bootstrap'] = test_bootstrap_loader

    return data_loader

if __name__ == '__main__' :
    import argparse
    os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--dset_dir', default='datasets', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    args = parser.parse_args()
