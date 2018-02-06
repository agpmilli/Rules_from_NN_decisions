import torch
import torch.nn as nn
import torch.nn.functional as F

class  twitter_Text(nn.Module):
    
    def __init__(self, args):
        super(twitter_Text,self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K//2) for K in Ks])

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(Co, C, bias=False)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


    def forward(self, x):
        x = self.embed(x) # (N,W,D)
        
        if self.args.static:
            x = Variable(x)

        #x = x.unsqueeze(1) # (N,Ci,W,D)
        x = x.permute(0,2,1)

        x = [F.relu(conv(x)) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)

        
        #x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        #x = torch.cat(x, 1)
        
        x = torch.cat(x, 2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        
        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        return logit
        
    def ngram_forward(self, x, K):
        # K is the kernel size
        # Ki is the index in the list of Ks
        Ks = self.args.kernel_sizes
        Ki = Ks.index(K)
               
        x = self.embed(x) # (1,W,D)
        
        if self.args.static:
            x = Variable(x)

        x = x.permute(0,2,1)
        
        conv = self.convs1[Ki]
        conv.padding=0
        x = conv(x)
        
        x = x.squeeze(0)
        
        x = F.relu(x) #(1,Co,W)
        
        x = x.permute(1,0)
        
        logit = self.fc1(x) # (N,C)
        
        logit = F.softmax(logit)
        return logit, x
        
    def tweet_forward(self, x):
        x = self.embed(x) # (N,W,D)
        
        if self.args.static:
            x = Variable(x)

        #x = x.unsqueeze(1) # (N,Ci,W,D)
        x = x.permute(0,2,1)

        x = [F.relu(conv(x)) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)

        
        #x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        #x = torch.cat(x, 1)
        
        x = torch.cat(x, 2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        
        y = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(y) # (N,C)
        return logit, x
        
    def tweet_forward_indices(self, x):
        x = self.embed(x) # (N,W,D)
        
        if self.args.static:
            x = Variable(x)

        #x = x.unsqueeze(1) # (N,Ci,W,D)
        x = x.permute(0,2,1)

        x = [F.relu(conv(x)) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)

        
        #x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        #x = torch.cat(x, 1)
        x = torch.cat(x, 2)
        (x, indices) = F.max_pool1d(x, x.size(2), return_indices=True)
        
        x = x.squeeze(2)
        
        y = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(y) # (N,C)
        return logit, x, indices