#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
# In[2]:


class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
                                            images=images,
                                            sizes=[1, self.patch_size, self.patch_size, 1],
                                            strides=[1, self.patch_size, self.patch_size, 1],
                                            rates=[1, 1, 1, 1],
                                            padding="VALID",
                                          )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [tf.shape(patches)[0], patches.shape[1] * patches.shape[2], patch_dims])

        return patches
     
    def get_config(self):
        config = super(Patches, self).get_config()
        config.update({"patch_size": self.patch_size})
        return config


# In[3]:



class PatchEncoder(Layer):
    """
    The PatchEncoder layer will linearly transform a patch by projecting it into a vector of size projection_dim. 
    In addition, it adds a learnable position embedding to the projected vector.

    """
    def __init__(self, num_patches, num_neurons):
        super(PatchEncoder, self).__init__()

        self.num_patches = num_patches
        self.num_neurons = num_neurons


        self.projection  = layers.Dense(units=self.num_neurons)
        self.position_embedding = layers.Embedding(input_dim=self.num_patches,
                                                   output_dim=self.num_neurons)


    def call(self, patch):

        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        enc_positions = self.position_embedding(positions) 
        encoded = self.projection(patch) + enc_positions 

        return encoded

    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({"num_patches": self.num_patches})
        config.update({"num_neurons": self.num_neurons})
        return config


# In[4]:


class MultiHeadAttentionLayer(Layer):
    def __init__(self, num_neurons, num_heads, drop_rate=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_neurons = num_neurons
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        
        self.multi_head_attention_layer =  layers.MultiHeadAttention(num_heads = self.num_heads,
                                                                     key_dim = self.num_neurons,
                                                                      dropout= self.drop_rate, 
                                                                     
                                                                      )
                                                                   
    def call(self, q, k, v): 
        return self.multi_head_attention_layer(q, k, v, return_attention_scores=True)
   
    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({"num_neurons": self.num_neurons})
        config.update({"num_heads": self.num_heads})
        config.update({"drop_rate": self.drop_rate})
        return config


class DropoutLayer(Layer):
    def __init__(self, drop_rate=0.1):
        super(DropoutLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dropout =  layers.Dropout(self.drop_rate )
                                                                       
    def call(self, x): 
        return self.dropout(x)
   
    def get_config(self):
        config = super(DropoutLayer, self).get_config()
        config.update({"drop_rate": self.drop_rate})
        return config


class FeedForwardLayers(Layer):
    def __init__(self, num_neurons, num_hidden_neurons, drop_rate=0.1):
        super(FeedForwardLayers, self).__init__()
        self.num_neurons = num_neurons
        self.num_hidden_neurons = num_hidden_neurons
        self.drop_rate = drop_rate

        self.dense1 = layers.Dense(self.num_neurons, activation='gelu')
        self.drop1 = layers.Dropout(self.drop_rate)
        self.dense2 = layers.Dense(self.num_hidden_neurons, activation='gelu')
        self.drop2 = layers.Dropout(self.drop_rate)
        self.dense3 = layers.Dense(self.num_neurons)
              
    def call(self, inputs): 
        x = self.dense1(inputs)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        x = self.dense3(x)
        return x
   
    def get_config(self):
        config = super(FeedForwardLayers, self).get_config()
        config.update({"num_neurons": self.num_neurons})
        config.update({"num_hidden_neurons": self.num_hidden_neurons})
        config.update({"drop_rate": self.drop_rate})
        return config


# In[7]:


class NormalizationLayer(Layer):
    def __init__(self, epsilon = 0.01):
        super(NormalizationLayer, self).__init__()
        self.epsilon = epsilon
        self.normal = layers.LayerNormalization(epsilon=self.epsilon)

    def call(self, inputs): 
        return self.normal (inputs)
   
    def get_config(self):
        config = super(NormalizationLayer, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

    
class CustomNormalizationLayer(Layer):

    def __init__(self, **kwargs):
        super(CustomNormalizationLayer, self).__init__(**kwargs)
        self.epsilon = K.epsilon() * K.epsilon()
        self.gamma = None
        self.beta = None

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        shape = input_shape[-1:]
        
        self.gamma = self.add_weight(
                shape=input_shape[-2:],
                initializer='zeros',#tf.initializers.GlorotUniform(),
                constraint=lambda t: tf.clip_by_value(t, clip_value_min=-1, clip_value_max=1),
                name='gamma',
                        )
        self.beta = self.add_weight(
                shape=shape,
                initializer=tf.initializers.GlorotUniform(),
                constraint=lambda t: tf.clip_by_value(t, clip_value_min=-1, clip_value_max=1),
                name='beta',
            )
        super(CustomNormalizationLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        
        mean = K.mean(inputs, axis=[0], keepdims=False)
        variance = K.mean(K.square(inputs - mean), axis=[0], keepdims=False)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        mean_coeff  = K.mean(outputs, axis=[0,1,2], keepdims=False)
        
        #output  =  (outputs * (self.gamma * mean_coeff)) + self.beta # 256
        #output  =  (outputs * (self.gamma * mean_coeff)) + self.beta# to 256*256

        #output  =  (outputs * self.gamma ) + self.beta     1.41    
        #output  =  ((mean_coeff * inputs)  + self.beta) #1.41#/ inputs
        #output  =  (((mean_coeff  * inputs)/ self.gamma) + self.beta) #/ inputs
        output  =  ((outputs/16) * self.gamma ) + self.beta #1.43

        #output  =  (((outputs)/ self.gamma) + self.beta) #/ inputs 1.36228728 1.43353343
        
        #print('coeff * inputs', tf.reduce_sum( mean_coeff * inputs, [0,1,2]))
        #print('self.beta * inputs', tf.reduce_sum( mean_coeff * inputs, [0,1,2]))

        return   output
        



    

class TransformerLayer(Layer):
    def __init__(self,num_neurons, num_hidden_neurons, num_heads):
        
        super(TransformerLayer, self).__init__()
        self.num_neurons = num_neurons
             
        self.multi_head_attention_layer1 = MultiHeadAttentionLayer(num_neurons, num_heads, drop_rate=0.1)
        self.dropout1 = DropoutLayer(drop_rate=0.0001)
        
        self.feed_forward_layer1  = FeedForwardLayers(num_neurons, num_hidden_neurons, drop_rate=0.1)
        self.feed_forward_dropout1 = DropoutLayer(drop_rate=0.0001)
        
       
               

    def call(self, sequence):
        

        attnention_output, attnention_weight = self.multi_head_attention_layer1(sequence,
                                                                                sequence,
                                                                                sequence, 
                                                                                )
        #print('attnention_output Before',tf.reduce_sum(attnention_output, [0,1,2]))
        attnention_output =  CustomNormalizationLayer()(attnention_output)
        #print('attnention_output After',tf.reduce_sum(attnention_output, [0,1,2]))
        attnention_output = sequence + self.dropout1(attnention_output)

        ff_a_output = self.feed_forward_layer1(attnention_output)
        #print('ff_a_output Before',tf.reduce_sum(ff_a_output, [0,1,2]))
        ff_a_output =  CustomNormalizationLayer()(ff_a_output)
        #print('ff_a_output After',tf.reduce_sum(ff_a_output, [0,1,2]))

        ff_a_output = attnention_output + self.feed_forward_dropout1(ff_a_output)

        
        return ff_a_output 


# In[9]:


class Encoder(Layer):
    
    def __init__(self, num_neurons ,num_hidden_neurons,
                       num_heads, num_enc_layers ):
        
        super(Encoder, self).__init__()
        
        
        self.num_enc_layers = num_enc_layers
        self.encoder_layers = [TransformerLayer(num_neurons,
                                                num_hidden_neurons,
                                                num_heads) for _ in range(self.num_enc_layers)]

    def call(self, sequence):
        for i in range(self.num_enc_layers):
            sequence  = self.encoder_layers[i](sequence)
        return sequence

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"num_enc_layers": self.num_enc_layers})
        return config


# In[10]:


class MaskTransformer(Layer):

    def __init__(self, num_neurons, num_hidden_neurons, num_heads, num_classes, num_transformer_layers):
        super(MaskTransformer, self).__init__()
        
        self.num_neurons = num_neurons
        self.num_hidden_neurons = num_hidden_neurons
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_transformer_layers =  num_transformer_layers

        
        self.cls_tokens = tf.random.normal((1, self.num_classes, self.num_neurons))
        self.transformerLayer = [TransformerLayer(num_neurons,
                                                  num_hidden_neurons,
                                                  num_heads) for _ in range(self.num_transformer_layers)]

                
    def call(self, sequence):
        
        
        n_batches = tf.shape(sequence)[0]#x.shape[0]
        cls_tokens = tf.repeat(self.cls_tokens,[n_batches], axis=0) 
        sequence = layers.concatenate([cls_tokens, sequence ], axis=1)
        
        for i in range(self.num_transformer_layers):
            sequence  = self.transformerLayer[i](sequence)
            
        c = sequence[:, :self.num_classes]
        z = sequence[:, self.num_classes:]
        return  z, c
    

    def get_config(self):
        config = super(MaskTransformer, self).get_config()
        config.update({"num_classes": self.num_classes})
        config.update({"num_transformer_layers": self.num_transformer_layers})
        config.update({"num_neurons": self.num_neurons})
        config.update({"num_hidden_neurons": self.num_hidden_neurons})
        config.update({"num_heads": self.num_heads})

        return config


# In[11]:


class Upsample(Layer):

    def __init__(self, image_size, patch_size):
        super(Upsample, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.p = self.image_size // self.patch_size
        self.upsample = layers.UpSampling2D( size=(self.patch_size, self.patch_size),
                                            interpolation='bilinear')

        
    def call(self, x):
        x = tf.reshape(x, (tf.shape(x)[0], self.p,  self.p, x.shape[2]))
        x = self.upsample(x)
        return x
    
    def get_config(self):
        config = super(Upsample, self).get_config()
        config.update({"image_size": self.image_size})
        config.update({"patch_size": self.patch_size})
        config.update({"p": self.p})
        return config


# In[12]:


def create_model(input_shape, num_patches, patch_size, num_neurons,
                 num_hidden_neurons, num_heads, num_classes, 
                 num_enc_layers, num_maskTransformer_layers):
    
    enc_inputs = layers.Input(shape=input_shape)
    print('Encoder Input Shape :',enc_inputs.shape)
    
    
    patching = Patches(patch_size)(enc_inputs)
    print('Patches Shape: ',patching.shape)

    patchEncoder = PatchEncoder(num_patches, num_neurons)(patching)
    print('Patch encoder Shape: ', patchEncoder.shape)


    encoder = Encoder(num_neurons ,
                      num_hidden_neurons,
                      num_heads, num_enc_layers )
    
    encoder_output = encoder(patchEncoder)
    print('Encoder output Shape: ', encoder_output.shape)
    
    
    decoder = MaskTransformer(num_neurons, num_hidden_neurons, num_heads, num_classes, num_maskTransformer_layers)
  
    z, c = decoder(encoder_output)
    print('z, c output Shape: ', z.shape, c.shape)
    
    
    masks = z @ tf.transpose(c, perm=[0, 2, 1])
    masks = tf.nn.softmax(masks , axis=-1)#
    print('Masks output Shapes: ', masks.shape) 


    upsapled_masks = Upsample(input_shape[0], patch_size)(masks)
    #upsapled_masks = tf.nn.relu(upsapled_masks )#,/ self.scale , axis=-1
    print('Upsapled masks output Shape: ', upsapled_masks.shape)

    model = Model(inputs=enc_inputs, outputs=upsapled_masks)
    return model


# In[ ]:




