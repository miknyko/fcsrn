import os
import random
import xml.etree.ElementTree as ET
import shutil
 
trainval_percent = 0.9
train_percent = 0.8
imagepath = r'.\YOLOV3\data\dataset\jpgs'
# txtsavepath = ''
total_img = os.listdir(imagepath)
 
num = len(total_img)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
 
ftrainval = open('./YOLOV3/data/dataset/trainval.txt', 'w')
ftest = open('./YOLOV3/data/dataset/test.txt', 'w')
ftrain = open('./YOLOV3/data/dataset/train.txt', 'w')
fval = open('./YOLOV3/data/dataset/val.txt', 'w')

for i in list:
    name = total_img[i] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
 
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

# convert the xml to yolo3 annotation

classes = ['indicator']
sets=['train','val','test']

cwd = os.getcwd()
def convert_annotation(image_id, list_file):
    xml_path = os.path.join(cwd,'YOLOV3','data','dataset','xml_annotations',f'{os.path.splitext(image_id)[0]}.xml')
    in_file = open(xml_path,encoding='gb18030',errors='ignore')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


for image_set in sets:
    path = os.path.join(cwd,'YOLOV3','data','dataset',f'{image_set}.txt')
    image_ids = open(path,encoding = 'gb18030',errors = 'ignore').read().strip().split()
    list_file = open(os.path.join(cwd,'YOLOV3',f'{image_set}.txt'),'w',encoding = 'gb18030',errors = 'ignore')
    for image_id in image_ids:
        image_path = os.path.join(cwd,'YOLOV3','data','dataset','jpgs',image_id)
        list_file.write(image_path)
        convert_annotation(image_id,list_file)
        list_file.write('\n')
    list_file.close()