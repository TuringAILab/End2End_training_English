import mxnet as mx
from mxnet import gluon, nd, autograd, gpu
import numpy as np


class U_Net (gluon.nn.HybridBlock):

    def __init__ (self, number_pre, pre_channels, output_size, number_downsampling, list_channels, list_number_convs, activation, prefix, norm=False, dropout=0):

        super (U_Net, self).__init__ (prefix=prefix)

        self.number_pre = number_pre
        self.pre_channels = pre_channels
        self.output_size = output_size
        self.number_downsampling = number_downsampling
        self.list_channels = list_channels
        self.list_number_convs = list_number_convs
        self.activation = activation
        self.dropout = dropout

        self.pre_layers = [] 
        self.down_layers = []       
        self.up_layers = []
 
        with self.name_scope ():       

            for i in xrange (self.number_pre):
		
		layer = gluon.nn.Dense (self.pre_channels, activation=self.activation, flatten=False, prefix='pre_layer_'+str(i)+'_')
		self.pre_layers.append (layer)
		self.register_child (layer)

            for i in xrange (self.number_downsampling):

		down_layer = []

		for j in xrange (self.list_number_convs[i]):

		    layer = gluon.nn.Conv1D (2 * self.list_channels[i], 3, padding=1, activation=None, prefix='downconv_'+str(i)+'_'+str(j)+'_')
		    down_layer.append (layer)
		    self.register_child (layer) 

		if norm:
		    continue

		down_layer.append (gluon.nn.AvgPool1D (strides=2, prefix='Avg_pooling_'+str(i)+'_'))

       	        self.down_layers.append (down_layer) 

            for i in reversed (xrange (self.number_downsampling)):

		up_layer = []

		layer = gluon.nn.Conv2DTranspose (self.list_channels[i], (2, 1), strides=(2, 1), activation=None, prefix='upsampling_'+str(i)+'_')  
	        up_layer.append (layer)
		self.register_child (layer)

		for j in xrange (self.list_number_convs[i]):

 		    layer = gluon.nn.Conv1D (2 * self.list_channels[i], 3, padding=1, activation=None, prefix='upconv_'+str(i)+'_'+str(j)+'_')
		    up_layer.append (layer)
		    self.register_child (layer)

	        if norm:
		    continue

		self.up_layers.append (up_layer)

	    #self.final_layer = gluon.nn.Dense (self.output_size, activation=None, flatten=False, prefix='final_layer_')
            self.final_layer = gluon.nn.Conv1D (512, 3, padding=1, activation='tanh', prefix='final_layer_')
	    self.final_dense = gluon.nn.Dense (self.output_size, activation=None, flatten=False, prefix='final_dense_')
	    self.register_child (self.final_layer)
	    self.register_child (self.final_dense)


    def hybrid_forward (self, F, x):

        for item in self.pre_layers:

	    x = item (x)

        x  = F.swapaxes (x, dim1=1, dim2=2)

        down_outs = []

        for i in xrange (self.number_downsampling):

            for j in xrange (len (self.down_layers[i]) - 1):

                x = self.down_layers[i][j] (x)
		x1, x2 = F.split (x, num_outputs=2, axis=1)

		x = F.sigmoid (x1) * F.tanh (x2)
	    
            down_outs.append (x)

            x = self.down_layers[i][-1] (x)

        for i in xrange (self.number_downsampling):

            x = F.expand_dims (x, axis=3)

            x = self.up_layers[i][0] (x)

            x = F.split (x, num_outputs=1, axis=3, squeeze_axis=True)

	    x1, x2 = F.split (x, num_outputs=2, axis=1)

            x = x + down_outs[self.number_downsampling - i - 1]

            if self.dropout > 0:

                x = F.Dropout(x, p=self.dropout)

            for j in range (1, len (self.up_layers[i])):

                x = self.up_layers[i][j] (x)
		x1, x2 = F.split (x, num_outputs=2, axis=1)

		x = F.sigmoid (x1) * F.tanh (x2)

        x = self.final_layer (x)
	    
        x = F.swapaxes (x, dim1=1, dim2=2)

        return self.final_dense (x)

 
	  






		
						

		
		

 	    
