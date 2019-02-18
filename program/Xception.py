
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import time 
import shutil
import random


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import PIL.Image as Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings(action='once')



# ## Pytorch to be backend
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pretrainedmodels import xception



np.random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed_all(666)
random.seed(666)


# ## Load dataset

path_to_train_metadata = '../input/train/'
path_to_train_info = '../input/train.csv'
path_to_test_metadata = '../input/test/'
path_to_test_info = '../input/sample_submission.csv'



# ### Should know some dataset information


data_info,data_set_total_num = get_data_info(path_to_train_info)



# ### Get the summary of dataset class number


name_with_cnt_dict= convert_class_to_name(data_info,data_set_total_num)
class_name = list(name_with_cnt_dict.keys())
class_num = len(class_name)



# ### Representing the dataset 



plt.figure(figsize=(15,15))
sns.barplot(y=list(name_with_cnt_dict.keys()), x=list(name_with_cnt_dict.values()))



# ## Split dataset as its distribution

# In[14]:


val_ratio = 0.12
train_data_list, val_data_list = train_test_split(data_info, test_size=val_ratio, random_state=666,stratify=data_info['Target'].map(lambda x: x[:3] if '27' not in x else '0'))
train_data_num = train_data_list.shape[0]
print('Train dataset has {}'.format(train_data_num))
val_data_num = val_data_list.shape[0]
print('Val dataset has {}'.format(val_data_num))


# ## Let's splite datasets 


train_class_with_cnt_dict = dataset_cnt_dict(train_data_list,train_data_num)
train_name_with_cnt_dict = convert_class_to_name(train_class_with_cnt_dict)


val_class_with_cnt_dict = dataset_cnt_dict(val_data_list,val_data_num)
val_name_with_cnt_dict = convert_class_to_name(val_class_with_cnt_dict)


class_weighted = get_class_weighted(train_class_with_cnt_dict,class_num)

train_dataset = DatasetGenerate(path_to_train_metadata,train_data_list)


# 获取当前文件名，用于创建模型及结果文件的目录
file_name = os.path.basename(__file__).split('.')[0]
# 创建保存模型和结果的文件夹
if not os.path.exists('./model/%s' % file_name):
    os.makedirs('./model/%s' % file_name)
if not os.path.exists('./result/%s' % file_name):
    os.makedirs('./result/%s' % file_name)
# 创建日志文件
if not os.path.exists('./result/%s.txt' % file_name):
    with open('./result/%s.txt' % file_name, 'w') as acc_file:
        pass
with open('./result/%s.txt' % file_name, 'a') as acc_file:
    acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), file_name))




class FixedRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        return fixed_rotate(img, self.angles)

def fixed_rotate(img, angles):
    angles = list(angles)
    angles_num = len(angles)
    index = random.randint(0, angles_num - 1)
    return img.rotate(angles[index])



class F1Loss(nn.Module):
    def __init__(self):
        super(F1Loss, self).__init__()
        
    def forward(self,y_pred,y_true):
        m = nn.Sigmoid()
        y_pred = m(y_pred)
        tp = torch.sum(y_true*y_pred,dim=0)
        tn = torch.sum((1-y_true)*(1-y_pred),dim=0)
        fp = torch.sum((1-y_true)*y_pred,dim=0)
        fn = torch.sum(y_true*(1-y_pred),dim=0)
        precision = tp / (tp+fp+1e-8)
        recall = tp / (tp+fn+1e-8)
        
        f1 = 2*precision*recall / (precision+recall+1e-8)
        
        return 1 - torch.mean(f1)
        







os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
# 小数据集上，batch size不易过大。如出现out of memory，应调小batch size
batch_size = 24
# 进程数量，最好不要超过电脑最大进程数。windows下报错可以改为workers=0
workers = 0


# predict threshold

threshold = 0.5
# epoch数量，分stage进行，跑完一个stage后降低学习率进入下一个stage
stage_epochs = [1, 2,4,6,10]  
# 初始学习率
lr = 1e-4
# 学习率衰减系数 (new_lr = lr / lr_decay)
lr_decay = 0.2
# 正则化系数
weight_decay = 1e-6

# 参数初始化
stage = 0
start_epoch = 0
total_epochs = sum(stage_epochs)
best_precision = -1
lowest_loss = 100

# 设定打印频率，即多少step打印一次，用于观察loss和acc的实时变化
# 打印结果中，括号前面为实时loss和acc，括号内部为epoch内平均loss和acc
print_freq = 1
# 验证集比例
val_ratio = 0.12
# 是否只验证，不训练
evaluate = False
# 是否从断点继续跑
resume = False
# 创建Xception模型
model = xception()
model.last_linear=nn.Linear(2048,28)
model = torch.nn.DataParallel(model).cuda()



# optionally resume from a checkpoint
if resume:
    checkpoint_path = './model/%s/checkpoint.pth.tar' % file_name
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        best_precision = checkpoint['best_precision']
        lowest_loss = checkpoint['lowest_loss']
        stage = checkpoint['stage']
        lr = checkpoint['lr']
        model.load_state_dict(checkpoint['state_dict'])
        # 如果中断点恰好为转换stage的点，需要特殊处理
        if start_epoch in np.cumsum(stage_epochs)[:-1]:
            stage += 1
#             optimizer = adjust_learning_rate()
            model.load_state_dict(torch.load('./model/%s/model_best.pth.tar' % file_name)['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

# 读取训练图片列表
# all_data = pd.read_csv('data/label.csv')
# # 分离训练集和测试集，stratify参数用于分层抽样
# train_data_list, val_data_list = train_test_split(all_data, test_size=val_ratio, random_state=666, stratify=all_data['label'])
# 读取测试图片列表
test_data_list = pd.read_csv(path_to_test_info)

# 图片归一化，由于采用ImageNet预训练网络，因此这里直接采用ImageNet网络的参数
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 训练集图片变换，输入网络的尺寸为299*299
train_data = DatasetGenerate(train_data_list,
                          transform=transforms.Compose([
                              transforms.Resize((320, 320)),
                              transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
                              transforms.RandomHorizontalFlip(),
#                               transforms.RandomGrayscale(),
                              # transforms.RandomRotation(20),
                              FixedRotation([0, 90, 180, 270]),
                              transforms.RandomCrop(299),
                              transforms.ToTensor(),
                              normalize,
                          ]))

# 验证集图片变换
val_data = DatasetGenerate(val_data_list,
                      transform=transforms.Compose([
                          transforms.Resize((320, 320)),
                          transforms.CenterCrop(299),
                          transforms.ToTensor(),
                          normalize,
                      ]))

# 测试集图片变换
test_data = TestDataset(test_data_list,
                        transform=transforms.Compose([
                            transforms.Resize((320, 320)),
                            transforms.CenterCrop(299),
                            transforms.ToTensor(),
                            normalize,
                        ]))

# 生成图片迭代器
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)
val_loader = DataLoader(val_data, batch_size=batch_size*2, shuffle=False, pin_memory=False, num_workers=workers)
test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=False, pin_memory=False, num_workers=workers)

# 使用MultiLabelSoftMarginLoss
# weighted = torch.tensor(class_weighted).cuda(async=True).view(1,28)
# weighted = torch.log(weighted)
# criterion = nn.MultiLabelSoftMarginLoss(weight=torch.tensor(weighted)).cuda()
criterion = F1Loss().cuda()
# criterion = nn.MultiLabelSoftMarginLoss().cuda()
# criterion = nn.CrossEntropyLoss().cuda()

# 优化器，使用带amsgrad的Adam
optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=stage_epochs, gamma=lr_decay)

if evaluate:
#     pass
    _,_,p,l=validate(val_loader, model, criterion)
    fig = draw_cm(p,l,prob_threshold=threshold)
    fig.savefig('./result/'+file_name+'/epochs_'+str(start_epoch)+'.png')
else:
    # 开始训练
    for epoch in range(start_epoch, total_epochs):
        scheduler.step()
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        precision, avg_loss,p,l = validate(val_loader, model, criterion)
        fig = draw_cm(p,l,prob_threshold=threshold)
        fig.savefig('./result/'+file_name+'/epochs_'+str(epoch)+'.png')

        # 在日志文件中记录每个epoch的精度和loss
        with open('./result/%s.txt' % file_name, 'a') as acc_file:
            acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, precision, avg_loss))

        # 记录最高精度与最低loss，保存最新模型与最佳模型
        is_best = precision > best_precision
        is_lowest_loss = avg_loss < lowest_loss
        best_precision = max(precision, best_precision)
        lowest_loss = min(avg_loss, lowest_loss)
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_precision': best_precision,
            'lowest_loss': lowest_loss,
            'stage': stage,
            'lr': lr,
        }
        save_checkpoint(state, is_best, is_lowest_loss)

        # 判断是否进行下一个stage
        if (epoch + 1) in np.cumsum(stage_epochs)[:-1]:
            stage += 1
#             optimizer = adjust_learning_rate()
            model.load_state_dict(torch.load('./model/%s/model_best.pth.tar' % file_name)['state_dict'])
            print('Step into next stage')
            with open('./result/%s.txt' % file_name, 'a') as acc_file:
                acc_file.write('---------------Step into next stage----------------\n')

# 记录线下最佳分数
with open('./result/%s.txt' % file_name, 'a') as acc_file:
    acc_file.write('* best acc: %.8f  %s\n' % (best_precision, file_name))
with open('./result/best_acc.txt', 'a') as acc_file:
    acc_file.write('%s  * best acc: %.8f  %s\n' % (
    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), best_precision, file_name))

# 读取最佳模型，预测测试集，并生成可直接提交的结果文件
best_model = torch.load('./model/%s/checkpoint.pth.tar' % file_name)
model.load_state_dict(best_model['state_dict'])
test(test_loader=test_loader, model=model,thresholder=threshold)

# 释放GPU缓存
torch.cuda.empty_cache()

