import torch
import torchvision.datasets as datasets
import numpy as np
import os
# turn head
# from model_1 import ProtoNetAGAM
# motion 4
from model_PCA_MSST import ProtoNetAGAM  # regular
from torch.autograd import Variable
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ['CUDA_VISIBLE_DEVICES']='0'

# Jason
# GQY
# Finesse
# 然
# Samishikude
# molly
# 陈昌忻
# 熊宇轩

# participants = ['Jason', 'GQY', 'Finesse', '然', 'Samishikude', 'molly', '陈昌忻', '熊宇轩']
participants = ['陈昌忻']

for parti in participants:

    participant = 'P_' + parti

    # single road type
    # road_type = 'turnR_offpeak'

    # all road type of a participant
    msstdir = './DataAug/'+participant+'/MSSTFeature'
    road_type_list = os.listdir(msstdir)
    for road_type in road_type_list:
        print(road_type)

        if (road_type.find("distractMotion")) == -1:
            channel = 21
            total_classes = 2
        else:
            continue
            channel = 2
            total_classes = 4

        traindatadir = './DataAug/' + participant + '/MSSTFeature/' + road_type
        testdatadir = './DataAug/' + participant + '/MSSTFeature/' + road_type+'/test'


    # side experiment
    # road_type_list = os.listdir('./'+participant+'/MSSTFeature_'+'turn'+'/')  # distractMotion ;; turn ;; changeLane ;; roundabout
    # for i in range(len(road_type_list)):
    #     road_type = str(road_type_list[i])
    #     print(road_type)
    # datadir = './'+participant+'/MSSTFeature_'+'turn'+'/'+road_type  # distractMotion ;; turn ;; roundabout

        log_path = 'Log/onlyChannel_'+participant+'_'+road_type+'.txt'
    
        with open(log_path, 'a') as f:
            text = 'road_type:{} \n'.format(road_type)
            f.write(text)

        # orig_set = datasets.ImageFolder(datadir)
        train_set = datasets.ImageFolder(traindatadir)
        test_set = datasets.ImageFolder(testdatadir)
        # k_fold = 3
        epochs = 60
        # # k_fold
        # kf = KFold(n_splits=k_fold, shuffle=True, random_state=2)
        # k_index = 0
        # for train_index, test_index in kf.split(orig_set):
        #
        #     train_subset = torch.utils.data.dataset.Subset(orig_set, train_index)
        #     test_subset = torch.utils.data.dataset.Subset(orig_set, test_index)
        #
        #     train_loader = torch.utils.data.DataLoader(train_subset, shuffle=True, batch_size=5, num_workers=0, pin_memory=True)
        #     test_loader = torch.utils.data.DataLoader(test_subset, shuffle=True, batch_size=5, num_workers=0, pin_memory=True)

        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=5, num_workers=0,pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=5, num_workers=0, pin_memory=True)


        model = ProtoNetAGAM(total_class=total_classes, channel=channel).cuda()
        optimizer = torch.optim.Adam(model.parameters())  # weight_decay
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.5)
        loss_func = torch.nn.CrossEntropyLoss().cuda()  # multiclass
        test_round = 0
        # ==============train==============
        for epoch in range(epochs):
            print('epoch {}'.format(epoch + 1))
            model.train()
            train_loss = 0.
            train_acc = 0.

            for i, (batch_x, batch_y) in enumerate(train_loader):
                # 对lr进行调整
                batch_x, batch_y = Variable(batch_x).float().cuda(), Variable(batch_y).cuda()
                # norm L1
                regularization_loss = 0
                lamda = 0.00015
                for param in model.parameters():
                    regularization_loss += torch.sum(abs(param))
                result = model(batch_x)
                loss = loss_func(result, batch_y) + (lamda * regularization_loss)
                train_loss += loss.item()
                pred = torch.max(result, 1)[1]
                train_correct = (pred == batch_y).sum()
                train_acc += train_correct.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss , train_acc ))

            # with open(log_path, 'a') as f:
            #     text = 'Around:{} Train Loss: {:.6f}, Acc: {:.6f} \n'.format(test_round, train_loss / (len(train_index)), train_acc / (len(train_index)))
            #     f.write(text)

            # ==============evaluation====================
            model.eval()
            eval_loss = 0.
            total_pred = []
            total_true = []


            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x, batch_y = Variable(batch_x).float().cuda(), Variable(batch_y).cuda()
                    result = model(batch_x)
                    loss = loss_func(result, batch_y)
                    eval_loss += loss.item()
                    pred = torch.max(result, 1)[1]
                    total_pred.extend(pred.tolist())
                    total_true.extend(batch_y.tolist())

                # precision
                p = precision_score(total_true, total_pred, average='macro').item()
                # recall
                r = recall_score(total_true, total_pred, average='macro').item()
                # F score
                F1 = f1_score(total_true, total_pred, average='macro').item()
                # acc
                acc = accuracy_score(total_true, total_pred).item()

                print('precision: {:.6f}  recall: {:.6f}  F score: {:.6f}  acc: {:.6f}    \n'.format(p, r, F1, acc))

                if epoch+1 == epochs:
                    with open(log_path, 'a') as f:
                        text = 'precision: {:.6f}  recall: {:.6f}  F score: {:.6f}  acc: {:.6f}    \n'.format(p, r, F1, acc)
                        f.write(text)

                test_round = test_round + 1

            # k_index = k_index + 1

