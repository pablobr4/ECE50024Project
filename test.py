import torch
import random
import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision.transforms as tvt
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

def forward_one_episode(learner_grad, metalearner, input_imgs, labels, criterion=nn.CrossEntropyLoss(), metaepochs=1, p=10, batch_size=1, verbose=False):
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
    num_episodes_test = 300     #number of testing episodes

    input_size = 4              #LSTM input size. Should always be 4.
    hidden_size = 20            #LSTM hidden size.

    learning_rate = 1e-3        #Learning rate for Adam optimizer
    metaepochs = 5              #Number of metaepochs each episode

    metalearner_model_path = "./metalearner.pth"
    #-----------------------------------------------
    batch_size = k_shot*num_classes_episode

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    location = "cuda" if torch.cuda.is_available() else "cpu"
    #Initialize learner network
    learner_grad = LearnerCNN(num_classes_episode).to(device)

    #Initialize Meta-Learner

    learner_params = learner_grad.get_params()
    n_params = learner_params.size(0) #could also have used the other learner to get the number of params as they are identical.

    metalearner = torch.load(metalearner_model_path, map_location=location)

    print("Initializing dataloaders...\n")

    #Initialize dataloader
    transforms = tvt.Compose([tvt.ToTensor()])

    test_set = dataset.EpisodeSet(root_path, train=False, k_shot=k_shot, eval_img_num=eval_img_num, transforms=transforms)
    test_sampler = dataset.EpisodeSampler(len(test_set), num_classes_episode=num_classes_episode, num_episodes=num_episodes_test)
    test_dataloader = DataLoader(test_set, num_workers=num_workers, batch_sampler=test_sampler)

    criterion = torch.nn.CrossEntropyLoss()

    print("Starting training...\n")

    softmax = nn.Softmax(0)

    accuracy = 0
    num_imgs = 0
    num_correct = 0
    for i, (episode, labeltensor) in enumerate(test_dataloader):
        
        train_inputs = episode[:, :k_shot].reshape(-1, *episode.shape[-3:]).to(device)
        train_labels = torch.LongTensor(np.repeat(range(num_classes_episode), k_shot)).to(device)

        test_inputs = episode[:, k_shot:].reshape(-1, *episode.shape[-3:]).to(device)
        test_labels = torch.LongTensor(np.repeat(range(num_classes_episode), eval_img_num)).to(device)

        learner_grad.train()

        ct = forward_one_episode(learner_grad, metalearner, train_inputs, train_labels, criterion=criterion, metaepochs=metaepochs, batch_size=batch_size)

        learner_grad.load_params_keep_grad(ct)
        softmax = nn.Softmax()

        for j in range(0,len(test_inputs)):
  
          inputs = test_inputs[j].unsqueeze(0)
          label = test_labels[j].unsqueeze(0)
          
          output = learner_grad(inputs).squeeze(0)

          prediction = torch.argmax(softmax(output))
          if int(prediction) == int(label):
              num_correct += 1
          
          num_imgs += 1

    print("Accuracy: " + str(num_correct/num_imgs*100) + "%")
    torch.save(num_correct/num_imgs*100, "accuracy.pth")
              