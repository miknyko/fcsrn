import tensorflow as tf
import numpy as np

from config import CFG
from fcsrn_model import FCSRN
from fcsrn_dataset import FcsrnDataset

trainset = FcsrnDataset()
model = FCSRN()
optimizer = tf.keras.optimizers.Adam(lr = 1e-4)
global_steps = 1


def train_step(image_data,y_true,batch_label_length,batch_logit_length):
    with tf.GradientTape() as tape:
        pred_result = model.model(image_data)
       
        loss = tf.nn.ctc_loss(y_true,pred_result,batch_label_length,batch_logit_length,logits_time_major=False,blank_index=20)
        cost = tf.reduce_mean(loss)
        
        gradients = tape.gradient(cost,model.model.trainable_variables)
        
        optimizer.apply_gradients(zip(gradients,model.model.trainable_variables))
        global global_steps
        print(f"[TRAINING INFO] STEP {global_steps} LOSS {cost}")
        global_steps += 1

if __name__ == '__main__':
    for epoch in range(CFG.FCSRN.EPOCHS):
        for image_data,y_true,batch_label_length,batch_logit_length in trainset:
            train_step(image_data,y_true,batch_label_length,batch_logit_length)

   


