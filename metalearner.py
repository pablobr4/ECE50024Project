import torch
import torch.nn as nn

class MetaLearnerLSTMCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, n_params):
        
        super(MetaLearnerLSTMCell, self).__init__()
        
        #    input_size: input size of this LSTM cell. In our case, it is the hidden size of the LSTM layer that comes before (20 by default).
        #    hidden_size: hidden size of this LSTM cell. In our case, this should be 1 because our output has size (n_params, 1).
        #    n_params: number of total parameters of the learner.
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_params = n_params

        #Now, let's create the parametrized form. The tensors must be cast into nn.Parameter objects so that they are added to the computational graph.
        #we add the +2 because the matrix to which this is multiplied is [out_lstm_layer_1, f_prev, c_prev]
        #the size of out_lstm_layer_1 is [p_params, input_size] and the size of f_prev and c_prev is [n_params, 1]
        #input_size is the hidden_size of the previous LSTM layer, which is 20 by default.
        # 
        # Weights: 
        self.Wf = nn.Parameter(torch.Tensor(input_size + 2, hidden_size)) 
        self.Wi = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        # Biases:
        self.bi = nn.Parameter(torch.Tensor(1, hidden_size))
        self.bf = nn.Parameter(torch.Tensor(1, hidden_size))
        # Output:
        self.ct = nn.Parameter(torch.Tensor(n_params, 1))

        #Now initialize all paremeters to the values suggested in the paper.
        self.initialize_parameters()

    def initialize_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.01, 0.01) #initialization of all parameters with random small values as recommended by the paper

        nn.init.uniform_(self.bf, 4, 6) #initialize forget gate bias large so the forget gate is initialized close to 1
        nn.init.uniform_(self.bi, -5, -4) #initialize input gate bias small so the initial learning rate is small.

    def set_ct(self, new_ct):
       #This keeps the requires_grad = True
       self.ct.data.copy_(new_ct.unsqueeze(1))

    def forward(self, inputs, hidden_state=None):
       
        
            #inputs = [x, grad_noprocess]
            #    x = output from previous LSTM, size (n_params, input_size) = (n_params, 20) by default. (our input size is the hidden size of the previous LSTM).
            #    grad_noprocess = gradient computed from the learner evaluation, without preprocessing. Used only for the update of c_next.
            
            #hidden_state = [f_prev, i_prev, c_prev]:
            #    f_prev: previous forget gate, size [n_params, 1]
            #    i_prev: previous input gate, size [n_params, 1]
            #    c_prev: previous learner parameters, size [n_params, 1]
        
        #get the output from the input LSTM cell, the unprocessed gradient and the batch size.
        x = inputs[0]
        grad_noprocess = inputs[1]
        batch_size, dummy = x.size()

        #if there is no hidden state yet, initialize it
        if hidden_state==None:
            f_prev = torch.zeros((batch_size, self.hidden_size)).to(self.Wf.device)
            i_prev = torch.zeros((batch_size, self.hidden_size)).to(self.Wi.device)
            c_prev = self.ct

            hidden_state = [f_prev, i_prev, c_prev]
        
        #get gate values and previous output from hidden state
        f_prev = hidden_state[0]
        i_prev = hidden_state[1]
        c_prev = hidden_state[2]

        #compute LSTM cell update for metalearner
        f_next = torch.sigmoid(torch.mm(torch.cat((x, c_prev, f_prev), 1), self.Wf) + self.bf.expand_as(f_prev))

        i_next = torch.sigmoid(torch.mm(torch.cat((x, c_prev, i_prev), 1), self.Wi) + self.bi.expand_as(i_prev))

        c_next = f_next.mul(c_prev) - i_next.mul(grad_noprocess)

        new_hidden_state = [f_next, i_next, c_next]

        #return new parameters for the learner and the new hidden state.
        return c_next, new_hidden_state


class MetaLearnerNet(nn.Module):

    def __init__(self, n_params, input_size=4, hidden_size=20):
        
        super(MetaLearnerNet, self).__init__()
        
        #    n_params: number of total parameters of the learner.
        #    input_size: input size of first LSTM layer. In the MetaLearner LSTM case it should be 4, because this is the width of the processed array [loss, grad].
        #    hidden_size: hidden size of first LSTM layer. User-defined, 20 by default.
        #
        #    The input_size of the Metalearner LSTM layer is "hidden_size=20".
        #    The hidden_size of the Metalearner LSTM layer is 1, because the output must be of size [n_params, 1] (a vector of length n_params)
        
        #Two LSTM cells are used. The first one, the input cell, is just a standard LSTM cell. The second one, the output cell, is our custom METALSTM cell.
        self.lstm_layer = nn.LSTMCell(input_size = input_size, hidden_size = hidden_size)
        self.metalstm_layer = MetaLearnerLSTMCell(input_size = hidden_size, hidden_size=1, n_params=n_params)

    def forward(self, inputs, hidden_state=None):
       
        
        #    inputs = [loss, grad, grad_noprocessed]
        #        loss: preprocessed, size [1,2]
        #        grad: preprocessed, size [n_params, 2]
        #        grad_noprocessed: NOT preprocessed, size [n_params]        
        #
        #    hidden_state = [(lstm_hidden, lstm_ct)], [metalstm_ft, metalstm_it, metalstm_ct]
        

        #retrieve processed loss and grad and unprocessed grad
        loss = inputs[0]
        grad = inputs[1]
        grad_noprocessed = inputs[2]

        #expand loss as many times as grad
        loss = loss.expand_as(grad)
        #combine inputs for lstm input cell. final size = (n_params, 4)
        inputs = torch.cat((loss, grad), 1)

        #initialize hidden states for input and output LSTM cells
        if hidden_state is None:
           hidden_state = [None, None]
        
        #forward through first LSTM cell layer and get the hidden state and the next cellstate for this layer.
        lstm_next_hidden, lstm_next_cellstate = self.lstm_layer(inputs, hidden_state[0])

        #forward throught metaLSTM cell layer, passing the hidden state of the first layer, the unprocessed gradient and the previous hidden state of the metaLSTM cell.
        ct, metalstm_hidden = self.metalstm_layer([lstm_next_hidden, grad_noprocessed], hidden_state[1])

        #return the learner weights, and the hidden states for both layers for the next iteration.
        return ct.squeeze(), [(lstm_next_hidden, lstm_next_cellstate), metalstm_hidden]