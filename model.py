import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        ''' Initialize the layers of this model.'''
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # embedding layer thst turns words into a vector of a specified size
        self.embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specidefed size) as inputs 
        # and outpus hidden states of size, hidden_size
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
    

        # the linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.hidden2score = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        """ Define the feedforward behavior of the model """            
        # captions[:, :-1] --> discard the <end> word to avoid prediction as <end> as input
        # create word embedding vector for each word in the captions
        embeds = self.embeddings(captions[:,:-1]) # embeds.shape: (batch_size, caption,shape[1]- 1, embed_size)
        
        # Stack features and captions
        inputs = torch.cat((features.unsqueeze(1), embeds), 1) # input.shape: (batch_size, caption,shape[1], embed_size)
        
        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, self.hidden = self.lstm(inputs)
                            
        
        # get the scores for the most likely word 
        outputs = self.hidden2score(lstm_out);     
        
        return outputs  

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []
        for index in range(max_len):
            lstm_out, states = self.lstm(inputs, states) # lstm_out.shape:(1,1,hidden_size)
            outputs = self.hidden2score(lstm_out) # outputs.shape:(1,1,vocab_size)
            outputs = outputs.squeeze(1) # outputs.shape:(1,vocab_size)
            target = outputs.argmax(dim=1) # most likely next word. target.shape: (1)
            
            # break the loop if it it <end> word
            if target.item() == 1:
                break
            
            # store predicted words
            caption.append(target.item()) # item(): get value from a tensor containing a single value
            
            # embed the last predicted word to be the new input of the lstm
            inputs = self.embeddings(target).unsqueeze(1) # inputs.shape : (1, embed_size)
            
        return caption
            
