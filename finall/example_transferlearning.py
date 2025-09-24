from mof import MOF_CGCNN
import csv,os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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



root = './SF6cif'
mof = MOF_CGCNN(cuda=True, root_file=root,trainset = train, valset=val,testset=test,epoch = 3000,lr=0.002,optim='Adam',batch_size=24,h_fea_len=580,n_conv=5,lr_milestones=[200],weight_decay=1e-7,dropout=0.15)

mof.transfer_learning(modelpath='./model_best.pth.tar',fix_layer_lr = 0.0008, flex_layer_lr = 0.001)

