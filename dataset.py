import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
import glob
import torchvision.transforms as tvt


class ClassDataset(Dataset):

    #this class represents the dataset for just one class.
    def __init__(self, img_pathlist, class_idx, transforms=None):

        #   img_pathlist: list with the paths to all images of this class.
        #   class_idx: index/label for this class.
        #   transforms: transformations done to the images of this class.

        self.img_pathlist = img_pathlist
        self.class_idx = class_idx
        self.transforms = transforms

    def __getitem__(self, idx):
        #open the image corresponding to the given index.
        img = Image.open(self.img_pathlist[idx]).convert('RGB')
        #apply transformations
        if self.transforms is not None:
            img = self.transforms(img)
        
        #return image and label
        return img, self.class_idx
    
    def __len__(self):
        #the length of this dataset is just the number of images for this class.
        return len(self.img_pathlist)
    
class EpisodeSet(Dataset):

    #this class represents the dataset of all episodes. An object of this Class has a dataloader for each
    #classification class, so that when this dataset is indexed with and "idx", an image and a label from
    #the class number idx is retrieved.
    def __init__(self, root_path, train=True, k_shot=5, eval_img_num = 15, transforms=None):

        #   root_path = path where all images are stored
        #   train = wheter the episodes are for metatraining or metatesting
        #   k_shot = number of images of each class in each episode
        #   eval_img_num = number of images for evaluation in each episode
        #   transforms = transformations done to each image

        if train:
            #if the episodes are for training, get the images from the training directory
            root_path = os.path.join(root_path, "train")
        else:
            #if the episodes are for testing, get the images from the testing directory
            root_path = os.path.join(root_path, "test")

        #the names of the classes are the names of the folders inside root_path/train or root_path/test.
        self.classes = sorted(os.listdir(root_path))

        #Check that no hidden files are present in self.classes. Specially when using MACOS, the .DS_Store file might
        #be added causing errors.
        for clas in self.classes:
            #hidden files start with a "." character.
            if clas[0] == ".":
                #Filter out hidden files
                self.classes.remove(clas)
        
        #get the paths to all images of all classes
        images = []
        for clas in self.classes:
            images.append(glob.glob(os.path.join(root_path, clas, '*')))
        
        batch_size = k_shot + eval_img_num #total number of images in each episode

        self.episode_loader = [] #Array with dataloaders for each class. Each dataloader has a batch size of k_shot + eval_img_num
        for idx, clas in enumerate(self.classes):
            #Create a dataset for this class, with all its images
            class_dataset = ClassDataset(images[idx], class_idx=idx, transforms=transforms)
            #Create a dataloader from this dataset. num_workers=0 so that the main thread performs the dataloading. Otherwise, there
            #are daemon errors.
            class_dataloader = DataLoader(class_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            #add the dataloader for this class to the episode loader.
            self.episode_loader.append(class_dataloader)
    
    def __getitem__(self, idx):
        #depending on the index, iterate the dataloader for the corresponding class
        return next(iter(self.episode_loader[idx]))

    def __len__(self):
        return len(self.classes)



    
class EpisodeSampler(Sampler):

    #this class is a batch sampler for the final dataloader. It just indicates how many images and from which classes to retrieve. With this, the dataloader
    #samples the EpisodeSet dataset for images for these classes.
    def __init__(self, num_classes_total, num_classes_episode, num_episodes):

        #   num_classes_total: total number of classes in the dataset
        #   num_classes_episode: number of classes to be presented in each episode
        #   num_episodes: number of episodes to be sampled.
        self.num_classes_total = num_classes_total
        self.num_classes_episode = num_classes_episode
        self.num_episodes = num_episodes

    def __iter__(self):

        #return one random permutation of the classes for every episode.
        for i in range(self.num_episodes):
            yield torch.randperm(self.num_classes_total)[:self.num_classes_episode] #because the permutation is computed on the set of all classes, retrieve only the first "num_classes_episode" classes.
    
    def __len__(self):
        return self.num_episodes