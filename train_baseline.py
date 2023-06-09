import torch
import random
import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision.transforms as tvt
from learner import LearnerCNN
import dataset
import torch.nn as nn

if __name__ == "__main__":

    #Initializing random seeds for reproducibility
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmarks = False
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.cuda.empty_cache()

    print("Initializing models...\n")
    
    
    #Initialize Learner
    #-------------- USER PARAMS --------------------
    num_classes_episode = 10
    k_shot = 5
    eval_img_num = 15
    root_path = "./images/"
    num_workers = 8
    
    batch_size = k_shot*num_classes_episode
    num_episodes_train = 100
    num_episodes_test = 100

    learning_rate = 1e-3
    metaepochs = 5
    #-----------------------------------------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Initializing dataloaders...\n")

    #Initialize dataloader
    transforms = tvt.Compose([tvt.ToTensor()])

    test_set = dataset.EpisodeSet(root_path, train=False, k_shot=k_shot, eval_img_num=eval_img_num, transforms=transforms)
    test_sampler = dataset.EpisodeSampler(len(test_set), num_classes_episode=num_classes_episode, num_episodes=num_episodes_test)
    test_dataloader = DataLoader(test_set, num_workers=num_workers, batch_sampler=test_sampler)

    print("Starting training...\n")

    num_correct = 0
    num_imgs = 0
    for i, (episode, labeltensor) in enumerate(test_dataloader):
        
        #Initialize learner network
        learner = LearnerCNN(num_classes_episode).to(device)

        #Initialize optimization technique
        optimizer = torch.optim.Adam(learner.parameters(), lr=learning_rate)

        train_inputs = episode[:, :k_shot].reshape(-1, *episode.shape[-3:]).to(device)
        train_labels = torch.LongTensor(np.repeat(range(num_classes_episode), k_shot)).to(device)

        test_inputs = episode[:, k_shot:].reshape(-1, *episode.shape[-3:]).to(device)
        test_labels = torch.LongTensor(np.repeat(range(num_classes_episode), eval_img_num)).to(device)

        learner.train()

        for epoch in range(0,metaepochs):

            for j in range(0,len(train_inputs),batch_size):

                inputs = train_inputs[j:j+batch_size]
                label = train_labels[j:j+batch_size]

                prediction = learner(inputs)

                loss = learner.criterion(prediction, label)
                optimizer.zero_grad()
                #compute gradients
                loss.backward()
                optimizer.step()
        
        softmax = nn.Softmax()

        for j in range(0,len(test_inputs)):

            inputs = test_inputs[j].unsqueeze(0)
            label = test_labels[j].unsqueeze(0)

            output = learner(inputs).squeeze(0)

            prediction = torch.argmax(softmax(output))
            if int(prediction) == int(label):
                num_correct += 1
            
            num_imgs += 1

    baseline_acc = num_correct/num_imgs*100
    torch.save(baseline_acc, "baseline_acc.pth")
    print("Baseline accuracy: " + str(baseline_acc) + "%")
    