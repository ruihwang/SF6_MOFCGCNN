from mof import MOF_CGCNN
import csv

##read data
with open('./dataset_label/SF6_train.csv',encoding='utf-8-sig') as f:
    readerv = csv.reader(f)
    train = [row for row in readerv]
with open('./dataset_label/SF6_test.csv',encoding='utf-8-sig') as f:
    readerv = csv.reader(f)
    test = [row for row in readerv]
with open('./dataset_label/SF6_val.csv',encoding='utf-8-sig') as f:
    readerv = csv.reader(f)
    val = [row for row in readerv]

with open('./Data-Driven-Screening.csv',encoding='utf-8-sig') as f:
    readerv = csv.reader(f)
    pred = [row for row in readerv]


root = './SF6cif'
predroot = '../../visc/hmof1'
#create a class
import time
time_start=time.time()
mof = MOF_CGCNN(works=40, root_file=root,trainset = train[:16], valset=val[:16],testset=test[:16],epoch = 2,lr=0.002,optim='Adam',batch_size=24,h_fea_len=580,n_conv=5,lr_milestones=[200],weight_decay=1e-7,dropout=0.15)





mof.pred_MOF(predroot,pred,'./model_best.pth.tar')

