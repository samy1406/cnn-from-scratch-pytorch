import numpy as np
import torch

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        # weights and bias will go here
        fan_in = in_channels * kernel_size * kernel_size
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2/fan_in)
        self.bias = np.zeros(out_channels)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        
    
    def forward(self, x):
        # convolution will go here
        # output dimension
        batch_size, in_channel, height, width = x.shape
        out_width = width - self.kernel_size + 1 + (2*self.padding)
        out_height = height - self.kernel_size + 1 + (2*self.padding)
        
        # step 2 applying padding
        x_padded = np.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))
        self.x_padded = x_padded
        # step 3 initialized output
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        

        # step 4 patch extraction
        for i in range(out_height):
            for j in range(out_width):
                patch = x_padded[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                output[:, :, i, j] = np.tensordot(patch, self.weights, axes=((1, 2, 3), (1, 2, 3)))
                output[:, :, i, j] += self.bias

                
        return output

    
    def backward(self, upstream_grad):
        # backprop will go here
        pass


class ReLU:
    def forward(self, x):
        # one line
        self.x = x
        output = np.maximum(x, 0) 
        return output
    
    def backward(self, upstream_grad):
        # one line
        mask = (self.x > 0)
        return mask * upstream_grad


class MaxPool:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out_height = height // self.pool_size
        out_width  = width  // self.pool_size
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.mask = np.zeros_like(x)  # saves where the max was
        self.x = x
        for i in range(out_height):
            for j in range(out_width):
                # extract 2x2 window
                window = x[:, :, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
                # take max
                output[:, :, i, j] = np.max(window, axis=(2, 3))
                # save mask — we'll do this after forward works
                window_mask = np.max(window, axis=(2, 3), keepdims=True)
                mask = (window == window_mask)
                self.mask[:, :, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size] = mask

        return output
    
    def backward(self, upstream_grad):
        grad_input = np.zeros_like(self.x)
        
        out_height = self.x.shape[2] // self.pool_size
        out_width  = self.x.shape[3] // self.pool_size
        
        for i in range(out_height):
            for j in range(out_width):
                # get the upstream gradient for this window — shape (batch, channels)
                grad = upstream_grad[:, :, i, j]
                # expand to window shape for broadcasting — shape (batch, channels, 1, 1)
                grad_expanded = grad[:, :, np.newaxis, np.newaxis]
                # place gradient only where mask is True
                grad_input[:, :, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size] = grad_expanded * self.mask[:, :, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
        
        return grad_input

class Flatten:
    def forward(self, x):
        # save shape for backward
        self.input_shape = x.shape
        batch_size, in_channel, height, width = x.shape
        return x.reshape(batch_size, in_channel * height * width)# reshape x to 2D
    
    def backward(self, upstream_grad):
        batch_size, in_channel, height, width = self.input_shape
        return upstream_grad.reshape(batch_size, in_channel, height, width)  # reshape back to original shape