import tensorflow as tf
import numpy as np
from config import CFG

def res_block_A(x,channels):
    """
    residual block A to obtain a large receptive field from the increased model depth and 
    avoid the gradients problem
    """

    x_shortcut = x

    # residual component 1
    x = tf.keras.layers.Conv2D(channels,(3,3),padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # residual componet 2
    x = tf.keras.layers.Conv2D(channels,(3,3),padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # final
    x = tf.keras.layers.Add()([x_shortcut,x])
    x = tf.keras.layers.ReLU()(x)

    return x


def res_block_B(x,channels):
    """
    residual block B to reduce the feature map and double the quantity of filters to 
    increase the model capabilites and preser the model complexity
    """

    x_shortcut = x

    # residual component 1
    x = tf.keras.layers.Conv2D(channels,(3,3),padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # residual componet 2
    x = tf.keras.layers.Conv2D(channels,(3,3),(2,2))(x)
    x = tf.keras.layers.BatchNormalization()(x) 

    # shortcut path
    x_shortcut = tf.keras.layers.Conv2D(channels,(3,3),(2,2))(x_shortcut)
    x_shortcut = tf.keras.layers.BatchNormalization()(x_shortcut)

    # final
    x = tf.keras.layers.Add()([x_shortcut,x])
    # x = tf.keras.layers.ReLU(x)
    x = tf.keras.activations.relu(x)

    return x



class FCSRN():
    def __init__(self):
        super().__init__()
        self.create_model()

    def create_model(self):
        # FCN backbone
        # entrance
        inputs = tf.keras.Input(shape=(CFG.FCSRN.INPUTSHAPE))
        x = tf.keras.layers.Conv2D(16,(3,3))(inputs)        
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        # stack 3 res block A,channels 16
        for i in range(3):
            x = res_block_A(x,channels=16)
        
        # stack 1 res block b, channels 24
        x = res_block_B(x,channels=24)

        # stack 3 res block A,channels 24
        for i in range(3):
            x = res_block_A(x,channels=24)

        # stack 1 res block b, channels 32
        x = res_block_B(x,channels=32)

        # stack 3 res block A,channels 32
        for i in range(3):
            x = res_block_A(x,channels=32)

        # stack 1 res block b, channels 48
        x = res_block_B(x,channels=48)

        # stack 3 res block a, channels 48
        for i in range(3):
            x = res_block_A(x,channels=48)

        
        # TEMPORAL MAPPER
        # The mapper outputs K(total charactor classes plus blank)channel feature maps
        x = tf.keras.layers.Conv2D(CFG.FCSRN.CLASSES,(3,3),strides=(1,1),padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.AveragePooling2D((2,1))(x)
        
        # Transcription Layer
        # Using CTC

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Softmax())(x)
        x = tf.squeeze(x)
        
        self.model = tf.keras.Model(inputs = inputs,outputs = x)




if __name__ == '__main__':
    mymodel = FCSRN()
    mymodel.model.summary()

        

