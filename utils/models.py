
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Lambda, Activation, Conv2D, Flatten
from keras.models import Model
from keras.layers.merge import concatenate, Add

class Critic:
    def __init__(self, state, action, state_dims, action_dims):
         
        self.state = state_dims
        self.action = action_dims
        self.state_dims = np.prod(state_dims) 
        self.action_dims = np.prod(action_dims)
        
        inputs = Input(tensor = state)
        inputs_action = Input(tensor = action)

        x = Dense(400, activation = 'relu')(inputs)
        x_a = Dense(300)(x)
        x_b = Dense(300)(inputs_action)
        out = Add()([x_a, x_b])
        out = Activation('relu')(out)
        self.output = Dense(1)(out)

        self.model = Model(inputs = [inputs, inputs_action], outputs  = self.output)
        self.network_params = self.model.weights
        self.action_grads = tf.gradients(self.model.output, inputs_action)

    def train_step(self, target_Q):
         
        learning_rate = 0.001

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.loss = tf.losses.mean_squared_error(target_Q, self.model.output)
        self.total_loss = self.loss

        train_step = self.optimizer.minimize(self.total_loss, var_list=self.network_params)
        return train_step        

class Actor:
    def __init__(self, state, state_dims, action_dims, action_bound_low, action_bound_high):
         
        self.state = state_dims
        self.state_dims = np.prod(state_dims)  
        self.action_dims = np.prod(action_dims)
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
                            
        inputs = Input(tensor = state)
        x = Dense(500, activation = 'relu')(inputs)
        x = Dense(400, activation = 'relu')(x)
        self.output = Dense(self.action_dims,activation = 'tanh')(x)
        self.output = Lambda(lambda i: tf.multiply(
            0.5, tf.multiply(
                i, (self.action_bound_high-self.action_bound_low)) + (self.action_bound_high+self.action_bound_low)))(self.output)

        self.model = Model(inputs = inputs, outputs  = self.output)
        self.network_params = self.model.weights
            
    def train_step(self, action_grads):

        learning_rate = 0.0001

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.grads = tf.gradients(self.model.output, self.network_params, -action_grads)                  
        train_step = self.optimizer.apply_gradients(zip(self.grads, self.network_params))

        return train_step

def generic_model(state_ph,
                  action_n,
                  ouput_activation = None,
                  activation = 'tanh'
                  ):

    inputs = tf.keras.layers.Input(tensor = state_ph)
    input_shape = state_ph.shape.as_list()

    if len(input_shape) > 2:   
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4
                                ,activation=activation
                                ,padding='valid'
                                ,kernel_initializer=tf.variance_scaling_initializer(scale=2))(inputs) 
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2
                                ,activation=activation
                                ,padding='valid'
                                ,kernel_initializer=tf.variance_scaling_initializer(scale=2))(x) 
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1
                                ,activation=activation
                                ,padding='valid'
                                ,kernel_initializer=tf.variance_scaling_initializer(scale=2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu'
        ,kernel_initializer=tf.variance_scaling_initializer(scale=2))(x) 
        x = tf.keras.layers.Dense(action_n, activation = ouput_activation
        ,kernel_initializer=tf.variance_scaling_initializer(scale=2))(x)
        return inputs, x

    x = tf.keras.layers.Dense(100, activation = activation)(inputs)
    x = tf.keras.layers.Dense(100, activation = activation)(x)
    x = tf.keras.layers.Dense(100, activation = activation)(x)
    x = tf.keras.layers.Dense(action_n, activation = ouput_activation)(x)
    return inputs, x

