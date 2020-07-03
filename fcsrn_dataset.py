import numpy as np
import tensorflow as tf
import os
import cv2
import tensorflow as tf
from config import CFG
from fcsrn_utils import label_to_array,sparse_tuple_from



class FcsrnDataset():
    def __init__(self):
        self.input_height = CFG.FCSRN.INPUTSHAPE[0]
        self.input_width = CFG.FCSRN.INPUTSHAPE[1]
        self.batch_size = CFG.FCSRN.BATCHSIZE
        self.max_length = CFG.FCSRN.MAXLENGTH
        self.classes = CFG.FCSRN.CLASSES
        self.images_path = CFG.FCSRN.TRAINSETPATH
        
        self.imgs_dirs = os.listdir(self.images_path)
        self.num_samples = len(self.imgs_dirs)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        
        
    def __iter__(self):
        return self
    
    def __next__(self):
        with tf.device('/cpu:0'):
            batch_image = np.zeros((self.batch_size,self.input_height,self.input_width,1),dtype=np.float32)
            labels = np.zeros((self.batch_size,self.max_length,self.classes))
            num = 0
            if self.batch_count < self.num_batches:
                batch_labels = []
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    image,label = self.load_image(self.imgs_dirs[index])
                    batch_image[num,:,:,:] = image
                    batch_labels.append(label)
                    num += 1
                batch_labels_sparsetensor = sparse_tuple_from(batch_labels)
                batch_label_length = np.array([None] * self.batch_size).reshape(-1,1)
                batch_logit_length = tf.convert_to_tensor([CFG.FCSRN.MAXTIMESTEP] * self.batch_size)
                self.batch_count += 1

                return batch_image,batch_labels_sparsetensor,batch_label_length,batch_logit_length
            else:
                self.batch_count = 0
                np.random.shuffle(self.imgs_dirs)
                raise StopIteration

    def load_image(self,image_path):
        image_path = os.path.join(self.images_path,image_path)
        img = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),-1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img,(self.input_width,self.input_height))
        img = np.expand_dims(img,axis = -1)
        # trasform the label(images name) to a array of code according to the charlist
        labels = image_path.split('\\')[-1].split('.')[0].split(',')
        label_array = label_to_array(labels,CFG.FCSRN.CHARLIST)

        return img,label_array


if __name__ == '__main__':
    test_path = r'0,0,0,3,4,18.jpg'
    data = FcsrnDataset()
    test_image = data.load_image(test_path)[0]
    print(test_image.shape)
    cv2.imshow('testimage',test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        