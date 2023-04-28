import torch
import torch.nn as nn
from collections import OrderedDict

class LearnerCNN(nn.Module):
    def __init__(self, num_classes):

        #   num_classes: number of classes that the learner will be classifying. This is the same as the number of classes in each episode.
        super(LearnerCNN, self).__init__()
        
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
           ('conv1', nn.Conv2d(3,16,5,padding=1)),
           ('relu1', nn.LeakyReLU()),
           ('pool1', nn.MaxPool2d(2)),
            
           ('conv2', nn.Conv2d(16, 32, 3, padding=1)),
           ('relu2', nn.LeakyReLU()),
            ('pool2', nn.MaxPool2d(2))
           ]))
        })

        self.num_classes = num_classes
        self.num_out_pixels = 1152
        self.model.update({('fc1', nn.Linear(self.num_out_pixels, num_classes))})

        #For multi-label classification, the cross-entropy loss is recommended. No softmax activation function is done after the fully connected layer because the CrossEntropyLoss takes logits, and not probabilities.
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        
        #convolutions
        x = self.model.features(x)
        #feature vector. Resize to (batch_size, -1)
        x = x.view(x.size(0), -1)
        #fully connected
        x = self.model.fc1(x)

        return x
    
    def load_params_keep_grad(self, new_params):
        #With this function, we update the values of all the parameters in the learner model. The copy_ function copies the
        #contents of the "new_params" tensor into the nn.Parameters of the learner. This is used during metatraining each episode,
        #because the gradient with respect to the learner weights (the output of the metalearner) must be computed.

        #   new_params: a vector with the values of the new parameters. These will be copied into the learner, keeping requires_grad=True.

        #index to keep count of the parameter set that we are modifying. Used when indexing new_params.
        idx = 0
        for param in self.model.parameters():
            #turn the tensor of the learner parameters into a vector and get the length so that we know how many parameters to take from new_params to copy into the learner.
            param_len = param.view(-1).size(0)
            #now get the new weights, reshape them to the shape of the parameter tensor and copy them.
            param.data.copy_(new_params[idx : idx+param_len].view_as(param))
            #update the index for the next iteration
            idx += param_len
    
    def load_params_lose_grad(self, new_params):
        #with this function, we update the parameters of the learner but cast them to tensors instead of nn.Parameters.
        #This makes these weights be used for forward propagation, but because they are tensors and not nn.Parameters, they are not optimized.
        #This is used when doing the forward to the test images, because the gradient
        #does not need to be computed with respect to the weights of the learner, only wrt the weights of the metalearner.
        #For an example of the difference between Tensor and nn.Parameter, see the second answer in this post:
        #https://stackoverflow.com/questions/56708367/difference-between-parameter-vs-tensor-in-pytorch

        #   new_params: a vector with the values of the new parameters. These will be copied into the learner as Tensors instead of parameters.

        idx = 0

        #run through every module in our model. In this learner, we have 3 modules whose weights will be updated: 2 convolutional layers and 1 fully connected layer.
        for module in self.model.modules():

            #check that the module that is being modified is one of the 3 modules that we care about.
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                #turn the tensor of the learner parameter weights into a vector and get the length so that we know how many parameters to take from new_params to copy into the learner.
                weight_len = module._parameters['weight'].view(-1).size(0)
                #clone the updated weights and assign them to the module weights. This turns them into tensors, instead of nn.Parameters, so they are not optimized.
                module._parameters['weight'] = new_params[idx: idx+weight_len].view_as(module._parameters['weight']).clone()
                #update index
                idx += weight_len

                #in case one of the layers has biases, copy them too.
                if module._parameters['bias'] is not None:
                    bias_len = module._parameters['bias'].view(-1).size(0)
                    module._parameters['bias'] = new_params[idx: idx+bias_len].view_as(module._parameters['bias']).clone()
                    idx += bias_len

    def get_params(self):

      #concatenate all parameters of all modules into a vector and return it.
      return torch.cat([param.view(-1) for param in self.model.parameters()], 0)