import torch
import random
import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import copy
from learner import LearnerCNN
import dataset
from metalearner import MetaLearnerNet
import torch.nn as nn

def process_loss_and_grad(x, p=10):

#The value of "p" is chosen as 10 as recommended in the papers.
#Originally, this function was calculated element-wise. However, this is extremely slow and I have changed it to tensor form using pytorch.

  rule = (x.abs() >= np.exp(-p)).to(torch.float32)

  # preproc1
  x_proc1 = rule * torch.log(x.abs() + 1e-8) / p + (1 - rule) * -1
  # preproc2
  x_proc2 = rule * torch.sign(x) + (1 - rule) * np.exp(p) * x
  return torch.stack((x_proc1, x_proc2), 1)

def forward_one_episode(learner_grad, metalearner, input_imgs, labels, metaepochs=1, p=10, batch_size=1, verbose=False):
    #get the values of the learner weights from the metalearner. These will be loaded into the learner later.
    ct = metalearner.metalstm_layer.ct.data
    #The initial hidden state for the metalearner is initialized inside the metalearner
    hidden_state = [None]

    #verbose for debugging purposes
    if verbose:
        print("\n\nNew episode")

    #go through metatraining images a number of times equal to metaepochs
    for metaepoch in range(0,metaepochs):

      #In the current implementation the batch_size is the whole metatrainig image set (k_shot*num_classes_episode).
      #This way, this for loop is only executed once per episode. However, in case the batch size was smaller, the for
      #loop is added.
      for i in range(0,len(input_imgs),batch_size):

        #get the input images
        inputs = input_imgs[i:i+batch_size]
        #get the corresponding labels
        label = labels[i:i+batch_size]

        #load the parameters that the metalearner predicts into the learner while keeping it attached to the computational graph.
        learner_grad.load_params_keep_grad(ct)

        #get the predictions of the learner
        prediction = learner_grad(inputs)

        #compute loss
        loss = learner_grad.criterion(prediction, label)
        #zero out the learner's gradients
        learner_grad.zero_grad()
        #compute the gradients
        loss.backward()
        #create the gradient vector that is then processed for the metalearner
        grad_data = []
        #get the gradient of each weight
        for param in learner_grad.parameters():
           #resize the weight so that it is a vector and add it to the list
           grad_data.append(param.grad.data.view(-1))
        
        #concatenate all elements of the list to get a single gradient vector.
        grad = torch.cat(grad_data, 0)


        #As per the paper, perform preprocessing on the loss and gradient
        processed_loss = process_loss_and_grad(loss.data.unsqueeze(0), p)
        processed_grad = process_loss_and_grad(grad, p)

        #the input to the metalearner are the processed loss and gradient and the unprocessed gradient
        metalearner_in = [processed_loss, processed_grad, grad.unsqueeze(1)]

        #get the new weights and the hidden state
        ct, h = metalearner(metalearner_in, hidden_state[-1])
        #add the new hidden state to the list so that it is used for the next image
        hidden_state.append(h)
        
        #feedback for debugging purposes
        if verbose:
            print("Labels shape: " + str(label.shape))
            print("Labels: " + str(label))
            print("Prediction shape: " + str(prediction.shape))
            print("Prediction: " + str(prediction))
            print("ct: " + str(ct))
            print("Loss: " + str(loss))
    
    return ct


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
    
    
    #USER DEFINED PARAMETERS
    #-------------- USER PARAMS --------------------
    num_classes_episode = 10    #number of classes per episode
    k_shot = 5                  #number of images of each class for metatraining
    eval_img_num = 15           #number of images of each class for metatesting
    root_path = "./images/"     #path where the "train" and "test" folders are
    num_workers = 2             #number of CPU workers for dataloading
    num_episodes_train = 1500   #number of training episodes

    input_size = 4              #LSTM input size. Should always be 4.
    hidden_size = 20            #LSTM hidden size.

    learning_rate = 1e-3        #Learning rate for Adam optimizer
    metaepochs = 5              #Number of metaepochs each episode
    #-----------------------------------------------
    batch_size = k_shot*num_classes_episode

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Initialize learner network
    learner_grad = LearnerCNN(num_classes_episode).to(device)

    #Let's now make a copy of this network for forward propagation during
    #training of the metalearner. A copy is necessary to have a version of the
    #learner that is not attached to the computational graph, so backpropagation is only done
    #to the meta-learner during its training. When deepcopying, the optimizer still
    #aims to optimize the original nets parameters unless it's explicitly told
    #to optimize the copy's parameters. We don't need to modify the copy's parameters,
    #so we don't point the optimizer to the new model.
    learner_nograd = copy.deepcopy(learner_grad)

    #Initialize Meta-Learner

    #get all learner parameters
    learner_params = learner_grad.get_params()
    #get the total number of parameters
    n_params = learner_params.size(0) #could also have used the other learner to get the number of params as they are identical.

    #create metalearner
    metalearner = MetaLearnerNet(n_params, input_size, hidden_size).to(device)
    #set the initial values of the metalearner output as the randomly initialized learner parameters.
    metalearner.metalstm_layer.set_ct(learner_params)

    print("Initializing dataloaders...\n")

    #Because the MNIST dataset is very simple, the only transformations to the images will be turning them to tensors.
    transforms = tvt.Compose([tvt.ToTensor()])

    #create the episode dataset
    train_set = dataset.EpisodeSet(root_path, train=True, k_shot=k_shot, eval_img_num=eval_img_num, transforms=transforms)
    #create the batch sampler for the dataloader
    train_sampler = dataset.EpisodeSampler(len(train_set), num_classes_episode=num_classes_episode, num_episodes=num_episodes_train)
    #create the dataloader of the episode dataset sampled by the batch sampler
    train_dataloader = DataLoader(train_set, num_workers=num_workers, batch_sampler=train_sampler)
    
    #Initialize optimization technique

    #Adam optimizer is used as suggested by the original paper. Only the metalearner parameters are optimized.
    optimizer = torch.optim.Adam(metalearner.parameters(), lr=learning_rate)
    #if the loss does not improve in 10 episodes, reduce the learning rate by a factor of 10 with the learning rate scheduler.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    print("Starting training...\n")

    time_print = 10 #print the running loss each 10 episodes
    loss_arr = [] #to save the loss for printing
    running_loss = 0

    #go through every episode
    for i, (episode, labeltensor) in enumerate(train_dataloader):
        
        #the first k_shot tensors are the metatraining images. Reshape them so that their dimension is [k_shot, C, H, W]
        # C = number of channels of each image
        # H = height in pixels
        # W = width in pixels
        train_inputs = episode[:, :k_shot].reshape(-1, *episode.shape[-3:]).to(device)

        #the labels are just sequences of the same label k_shot times. For example, for k_shot = 3 and num_classes_episode = 4, the label would be:
        #label = tensor([0,0,0,1,1,1,2,2,2,3,3,3])
        train_labels = torch.LongTensor(np.repeat(range(num_classes_episode), k_shot)).to(device)

        #same for the metatest images, from k_shot-th image to the end, all images are metatesting images.
        test_inputs = episode[:, k_shot:].reshape(-1, *episode.shape[-3:]).to(device)
        test_labels = torch.LongTensor(np.repeat(range(num_classes_episode), eval_img_num)).to(device)

        #compute the final weights after metatraining. the function below does not perform the backpropagation after metatesting
        ct = forward_one_episode(learner_grad, metalearner, train_inputs, train_labels, metaepochs=metaepochs, batch_size=batch_size)

        #Now that we have the final estimation of the weights, let's get the loss and gradient
        #from the meta-test images and backpropagate through the metalearner

        #First, update the weights of the learner copy, while detaching it from the computational graph
        learner_nograd.load_params_lose_grad(ct)
        #get the predictions of the learner on the metatest images
        output = learner_nograd(test_inputs)
        #compute the loss
        loss = learner_nograd.criterion(output, test_labels)
        #zero out the gradients associated to this optimizer (the metalearner parameter gradients)
        optimizer.zero_grad()
        #compute the gradients of the metalearner parameters (because the learner is detached from the computational graph)
        loss.backward()
        #the paper suggest that performance is better with gradient clipping.
        nn.utils.clip_grad_norm_(metalearner.parameters(), 0.25)
        #update the metalearner parameters.
        optimizer.step()

        #add the running loss
        running_loss += loss.item()
        
        #print the running loss every 10 episodes.
        if (i+1) % time_print == 0:
            print("\n[episode:%5d] loss: %.6f" %
                    (i + 1, running_loss / float(time_print)))
            loss_arr.append(running_loss/time_print)
            scheduler.step(running_loss)
            running_loss = 0.0        
            
    #save the model and its state dictionary
    torch.save(metalearner.state_dict(), "metalearner_dict.pth")
    torch.save(metalearner, "metalearner.pth")

    #plot the loss
    plt.plot(loss_arr)
    plt.title("Training Loss")
    plt.xlabel("iter")
    plt.ylabel("Loss")
    plt.savefig("loss.png")
    torch.save(loss_arr, "loss_arr.pth")