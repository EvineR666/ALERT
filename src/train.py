import torch
import torchvision.datasets as datasets
import numpy as np
import os
import sys
# turn head
# from model_1 import ProtoNetAGAM
# motion 4
from model_PCA_MSST import ProtoNetAGAM  # regular
from torch.autograd import Variable
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sklearn



# Jason
# GQY
# Finesse
# 然
# Samishikude
# molly
# 陈昌忻
# 熊宇轩

def main(participants,road_type,model_type=3):

    # participants = ['Jason', 'GQY', 'Finesse', '然', 'Samishikude', 'molly', '陈昌忻', '熊宇轩']
    # participants = ['Finesse', '然', 'molly']
    a_acc_all={}
    a_TP={}
    a_TN={}
    a_FP={}
    a_FN={}
    a_result={}

    for parti in participants:

        # participant = 'P_' + parti
        participant = 'P_' + parti
        root = 'result3/result_'+parti+'/'
        # root = 'result3/result_' + parti + '/' + str(model_type)
        if not os.path.exists(root):
            os.makedirs(root)

        # all road type of a participant
        msstdir = './'+participant+'/MSSTFeature_new'
        road_type_list=[]
        road_type_list.append(road_type)
        # road_type_list = os.listdir(msstdir)

        acc_all = {}
        TP = {}
        TN = {}
        FP = {}
        FN = {}
        result2 = {}
        for road_type in road_type_list:
            print(road_type)
            acc_t=[]
            result_pred=[]
            result_true=[]

            if (road_type.find("distractMotion")) == -1:
                channel = 21
                total_classes = 2
            else:
                continue
                channel = 2
                total_classes = 4

            datadir = './' + participant + '/MSSTFeature_new/' + road_type


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

            orig_set = datasets.ImageFolder(datadir)

            k_fold = 3
            epochs = 5
            # k_fold
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=2)
            k_index = 0


            for train_index, test_index in kf.split(orig_set):

                train_subset = torch.utils.data.dataset.Subset(orig_set, train_index)
                test_subset = torch.utils.data.dataset.Subset(orig_set, test_index)

                train_loader = torch.utils.data.DataLoader(train_subset, shuffle=True, batch_size=5, num_workers=8, pin_memory=True)
                test_loader = torch.utils.data.DataLoader(test_subset, shuffle=True, batch_size=5, num_workers=8, pin_memory=True)

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

                    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
                        train_index)), train_acc / (len(train_index))))

                    # with open(log_path, 'a') as f:
                    #     text = 'Around:{} Train Loss: {:.6f}, Acc: {:.6f} \n'.format(test_round, train_loss / (len(train_index)), train_acc / (len(train_index)))
                    #     f.write(text)

                    # ==============evaluation====================
                    model.eval()
                    eval_loss = 0.
                    total_pred = []
                    total_true = []
                    total_score = []


                    with torch.no_grad():
                        for i, (batch_x, batch_y) in enumerate(test_loader):
                            batch_x, batch_y = Variable(batch_x).float().cuda(), Variable(batch_y).cuda()
                            result = model(batch_x)
                            loss = loss_func(result, batch_y)
                            eval_loss += loss.item()
                            pred = torch.max(result, 1)[1]
                            total_pred.extend(pred.tolist())
                            total_true.extend(batch_y.tolist())
                            # for y in result.tolist():
                            #     total_score.append(y[1])
                        # fpr, tpr, thresholds = sklearn.metrics.roc_curve(total_true, total_score, pos_label=None,
                        #                                                  sample_weight=None, drop_intermediate=True)

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
                            acc_t.append(acc)
                            result_pred.append(total_pred)
                            result_true.append(total_true)
                            # for pr, t in zip(total_pred, total_true):
                            #     if t == 1 and pr == 1:
                            #         TP_t += 1
                            #     elif t == 1 and pr == 0:
                            #         FN_t += 1
                            #     elif t == 0 and pr == 0:
                            #         TN_t += 1
                            #     elif t == 0 and pr == 1:
                            #         FP_t += 1
                            with open(log_path, 'a') as f:
                                text = 'precision: {:.6f}  recall: {:.6f}  F score: {:.6f}  acc: {:.6f}    \n'.format(p, r, F1, acc)
                                f.write(text)

                        test_round = test_round + 1

                k_index = k_index + 1
            acc_all[road_type]=acc_t
            # TP[road_type]=TP_t
            # TN[road_type]=TN_t
            # FP[road_type]=FP_t
            # FN[road_type]=FN_t
            result2[road_type]= {'pred':result_pred,
                                'true':result_true}

            a_acc_all[parti]=acc_all
            a_TP[parti]=TP
            a_TN[parti]=TN
            a_FP[parti]=FP
            a_FN[parti]=FN
            a_result[parti]=result2

            with open(root+'/{}_acc.txt'.format(road_type),'w') as f:
                f.write(str(a_acc_all))
            with open(root+'/{}_result.txt'.format(road_type),'w') as f:
                f.write(str(a_result))

if __name__=='__main__':
    # participants = ['Jason', 'GQY', 'Finesse', '然', 'Samishikude', 'molly', '陈昌忻', '熊宇轩']
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    participants = ['GQY']
    road_type=''
    model_type=3
    # if len(sys.argv)>3:
    #     model_type=int(sys.argv[4])
    if len(sys.argv)>1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[2])
        participants = []
        participants.append(sys.argv[1])
        road_type=sys.argv[3]
    main(participants,road_type,model_type)