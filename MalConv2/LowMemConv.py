from collections import deque

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def drop_zeros_hook(module, grad_input, grad_out):
    """
    This function is used to replace gradients that are all zeros with None
    In pyTorch None will not get back-propogated
    So we use this as a approximation to saprse BP to avoid redundant and useless work
    """
    grads = []
    with torch.no_grad():
        for g in grad_input:
            if torch.nonzero(g).shape[0] == 0:#ITS ALL EMPTY!
                grads.append(g.to_sparse())
            else:
                grads.append(g)
                
    return tuple(grads)

class CatMod(torch.nn.Module):
    def __init__(self):
        super(CatMod, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim=2)
    
    

class LowMemConvBase(nn.Module):
    
    def __init__(self, chunk_size=65536, overlap=512, min_chunk_size=1024):
        """
        chunk_size: how many bytes at a time to process. Increasing may improve compute efficent, but use more memory. Total memory use will be a function of chunk_size, and not of the length of the input sequence L
        
        overlap: how many bytes of overlap to use between chunks
        
        """
        super(LowMemConvBase, self).__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
            
        #Used for pooling over time in a meory efficent way
        self.pooling = nn.AdaptiveMaxPool1d(1)
    #   self.pooling.register_backward_hook(drop_zeros_hook)
        self.cat = CatMod()
        self.cat.register_backward_hook(drop_zeros_hook)
        self.receptive_field = None
        
        #Used to force checkpoint code to behave correctly due to poor design https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
    
    def processRange(self, x, **kwargs):
        """
        This method does the work to convert an LongTensor input x of shape (B, L) , where B is the batch size and L is the length of the input. The output of this functoin should be a tensor of (B, C, L), where C is the number of channels, and L is again the input length (though its OK if it got a little shorter due to convs without padding or something). 
        
        """
        pass
    
    def determinRF(self):
        """
        Lets determine the receptive field & stride of our sub-network
        """
        
        if self.receptive_field is not None:
            return self.receptive_field, self.stride, self.out_channels
        #else, figure this out! 
        
        if not hasattr(self, "device_ids"):
            #We are training with just one device. Lets find out where we should move the data
            cur_device = next(self.embd.parameters()).device
        else:
            cur_device = "cpu"
            
        #Lets do a simple binary search to figure out how large our RF is. 
        #It can't be larger than our chunk size! So use that as upper bound
        min_rf = 1
        max_rf = self.chunk_size
        
        with torch.no_grad():
            
            tmp = torch.zeros((1,max_rf)).long().to(cur_device)
            
            while True:
                test_size = (min_rf+max_rf)//2
                is_valid = True
                try:
                    self.processRange(tmp[:,0:test_size])
                except:
                    is_valid = False
                
                if is_valid:
                    max_rf = test_size
                else:
                    min_rf = test_size+1
                    
                #print(is_valid, test_size, min_rf, max_rf)
                    
                if max_rf == min_rf:
                    self.receptive_field = min_rf
                    out_shape = self.processRange(tmp).shape
                    self.stride = self.chunk_size//out_shape[2]
                    self.out_channels = out_shape[1]
                    break
                    
                
        return self.receptive_field, self.stride, self.out_channels
                
    
    def pool_group(self, *args):
        #x = torch.cat(args[0:-1], dim=2)
        x = self.cat(args)
        x = self.pooling(x)
        return x
    
    def seq2fix(self, x, pr_args={}):
        """
        Takes in an input LongTensor of (B, L) that will be converted to a fixed length representation (B, C), where C is the number of channels provided by the base_network  given at construction. 
        """
        
        receptive_window, stride, out_channels = self.determinRF()
        
        if x.shape[1] < receptive_window: #This is a tiny input! pad it out please
            x = F.pad(x, (0, receptive_window-x.shape[1]), value=0)#0 is the pad value we use 
        
        batch_size = x.shape[0]
        length = x.shape[1]
        
        

        #Lets go through the input data without gradients first, and find the positions that "win"
        #the max-pooling. Most of the gradients will be zero, and we don't want to waste valuable
        #memory and time computing them. 
        #Once we know the winners, we will go back and compute the forward activations on JUST
        #the subset of positions that won!
        winner_values = np.zeros((batch_size, out_channels))-1.0
        winner_indices = np.zeros((batch_size, out_channels), dtype=np.int64)
            
        if not hasattr(self, "device_ids"):
            #We are training with just one device. Lets find out where we should move the data
            cur_device = next(self.embd.parameters()).device
        else:
            cur_device = None

        step = self.chunk_size #- self.overlap
        #step = length
        start = 0
        end = start+step
        
        
        #TODO, I'm being a little sloppy on picking exact range, and selecting more context than i need
        #Future, should figure out precisely which bytes won and only include that range

        #print("Starting Search")
        with torch.no_grad():
            while start < end and (end-start) >= max(self.min_chunk_size, receptive_window):
                #print("Range {}:{}/{}".format(start,end,length))
                x_sub = x[:,start:end]
                if cur_device is not None:
                    x_sub = x_sub.to(cur_device)
                activs = self.processRange(x_sub.long(), **pr_args)
                activ_win, activ_indx = F.max_pool1d(activs, kernel_size=activs.shape[2], return_indices=True)
                #print(activ_win.shape)
                #Python for this code loop is WAY too slow! Numpy it!
                #for b in range(batch_size):
                #    for c in range(out_channels):
                #        if winner_values[b,c] < activ_win[b,c]:
                #            winner_indices[b, c] = activ_indx[b, c]*stride + start + receptive_window//2
                #            winner_values[b,c]   = activ_win[b,c]
                #We want to remove only last dimension, but if batch size is 1, np.squeeze
                #will screw us up and remove first dime too. 
                #activ_win = np.squeeze(activ_win.cpu().numpy())
                #activ_indx = np.squeeze(activ_indx.cpu().numpy())
                activ_win = activ_win.cpu().numpy()[:,:,0]
                activ_indx = activ_indx.cpu().numpy()[:,:,0]
                selected = winner_values < activ_win
                winner_indices[selected] = activ_indx[selected]*stride + start 
                winner_values[selected]  = activ_win[selected]
                start = end
                end = min(start+step, length)

        #Now we know every index that won, we need to compute values and with gradients! 

        #Find unique winners for every batch
        final_indices = [np.unique(winner_indices[b,:]) for b in range(batch_size)]
        
        #Collect inputs that won for each batch
        chunk_list = [[x[b:b+1,max(i-receptive_window,0):min(i+receptive_window,length)] for i in final_indices[b]] for b in range(batch_size)]
        #Convert to a torch tensor of the bytes
        chunk_list = [torch.cat(c, dim=1)[0,:] for c in chunk_list]
        
        #Padd out shorter sequences to the longest one
        x_selected = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True)
        
        #Shape is not (B, L) Lets compute!
        
        if cur_device is not None:
            x_selected = x_selected.to(cur_device)
        x_selected = self.processRange(x_selected.long(), **pr_args)
        x_selected = self.pooling(x_selected)
        x_selected = x_selected.view(x_selected.size(0), -1)
            
        return x_selected
        
