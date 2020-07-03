from easydict import EasyDict as edict


CFG                      = edict()

CFG.FCSRN                = edict()


# FCSRN config
CFG.FCSRN.CLASSES        = 21
CFG.FCSRN.INPUTSHAPE = (48,160,1)
CFG.FCSRN.BATCHSIZE = 4
CFG.FCSRN.MAXLENGTH = 6

CFG.FCSRN.EPOCHS = 12
CFG.FCSRN.TRAINSETPATH = './fcsrn_train_images'
CFG.FCSRN.CHARLIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
CFG.FCSRN.MAXTIMESTEP = 16  # TUNE THIS ACCORDING TO THE SHAPE OF THE OUTPUT OF THE NET



if __name__ == '__main__':
    print(CFG)