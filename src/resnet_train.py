import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import resnet as resnet

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":resnet.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

def main(participants,road_type,model_type=3):
    # participants = ['Jason', 'GQY', 'Finesse', '然', 'Samishikude', 'molly', '陈昌忻', '熊宇轩']
    # participants = ['Jason', 'GQY', 'Samishikude', '陈昌忻', '熊宇轩']
    # participants = ['Finesse', '然', 'molly']
    # participants = 'GQY'

    a_acc_all={}
    a_result={}
    a_recall={}
    a_presicion={}
    a_f1={}

    for parti in participants:

        # participant = 'P_' + parti + '_side'
        participant = 'P_' + parti
        root = 'result3/result_distract_' + parti + '/' + str(model_type)
        if not os.path.exists(root):
            os.mkdir(root)
        # single road type
        # road_type = 'turnR_offpeak'

        # all road type of a participant
        msstdir = './'+participant+'/MSSTFeature_new'
        # road_type_list = os.listdir(msstdir)
        road_type_list=[]
        road_type_list.append(road_type)

        acc_all = {}
        result2 = {}
        recall_all={}
        precision_all={}
        f1_all={}

        for road_type in road_type_list:
            print(road_type)

            acc_t = []
            result_pred = []
            result_true = []
            recall_t=[]
            precision_t=[]
            f1_t=[]
            channel = 2
            total_classes = 4
            if (road_type.find("distractMotion")) == -1:
                continue
                channel = 21
                total_classes = 2
            else:
                channel = 2
                total_classes = 4

            datadir = './' + participant + '/MSSTFeature_new/' + road_type


        # side experiment
        # road_type_list = os.listdir('./'+participant+'/MSSTFeature_'+'turn'+'/')  # distractMotion ;; turn ;; changeLane ;; roundabout
        # for i in range(len(road_type_list)):
        #     road_type = str(road_type_list[i])
        #     print(road_type)
        # datadir = './'+participant+'/MSSTFeature_'+'turn'+'/'+road_type  # distractMotion ;; turn ;; roundabout

            log_path = 'Log/OnlySpatial'+participant+'_'+road_type+'.txt'

            with open(log_path, 'a') as f:
                text = 'road_type:{} \n'.format(road_type)
                f.write(text)

            orig_set = datasets.ImageFolder(datadir)

            k_fold = 5
            epochs = 10
            # k_fold
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=2)
            k_index = 0
            for train_index, test_index in kf.split(orig_set):

                train_subset = torch.utils.data.dataset.Subset(orig_set, train_index)
                test_subset = torch.utils.data.dataset.Subset(orig_set, test_index)

                #train_loader = torch.utils.data.DataLoader(train_subset, shuffle=True, batch_size=5, num_workers=0, pin_memory=False)
                #test_loader = torch.utils.data.DataLoader(test_subset, shuffle=True, batch_size=5, num_workers=0, pin_memory=False)
                train_loader = torch.utils.data.DataLoader(train_subset, shuffle=True, batch_size=5, num_workers=10, pin_memory=True)
                test_loader = torch.utils.data.DataLoader(test_subset, shuffle=True, batch_size=5, num_workers=10, pin_memory=True)

                # model = ProtoNetAGAM(total_class=total_classes, channel=channel).cuda()
                model = resnet(num_classes=total_classes).cuda()

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
                        #batch_x, batch_y = Variable(batch_x).float(), Variable(batch_y)
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


                    with torch.no_grad():
                        for i, (batch_x, batch_y) in enumerate(test_loader):
                            batch_x, batch_y = Variable(batch_x).float().cuda(), Variable(batch_y).cuda()
                            #batch_x, batch_y = Variable(batch_x).float(), Variable(batch_y)
                            result = model(batch_x)
                            loss = loss_func(result, batch_y)
                            eval_loss += loss.item()
                            pred = torch.max(result, 1)[1]
                            total_pred.extend(pred.tolist())
                            total_true.extend(batch_y.tolist())

                        # # precision
                        p = precision_score(total_true, total_pred, average='macro').item()
                        # # recall
                        r = recall_score(total_true, total_pred, average='macro').item()
                        # # F score
                        F1 = f1_score(total_true, total_pred, average='macro').item()
                        # # acc
                        acc = accuracy_score(total_true, total_pred).item()

                        # classification report
                        target_names = orig_set.classes
                        report = classification_report(total_true, total_pred, target_names=target_names)

                        print(report)

                        if epoch+1 == epochs:
                            acc_t.append(acc)
                            precision_t.append(p)
                            recall_t.append(r)
                            f1_t.append(F1)
                            result_pred.append(total_pred)
                            result_true.append(total_true)
                            with open(log_path, 'a') as f:
                                text = report
                                f.write(text)
                        test_round = test_round + 1

                k_index = k_index + 1
            acc_all[road_type] = acc_t
            precision_all[road_type]=precision_t
            recall_all[road_type]=recall_t
            f1_all[road_type]=f1_t
            result2[road_type] = {'pred': result_pred,
                                  'true': result_true}


            a_acc_all[parti] = acc_all
            a_presicion[parti] = precision_all
            a_f1[parti] = f1_all
            a_recall[parti] = recall_all
            a_result[parti] = result2

            with open(root + '/{}_acc.txt'.format(road_type), 'w') as f:
                f.write(str(a_acc_all))
            with open(root + '/{}_result.txt'.format(road_type), 'w') as f:
                f.write(str(a_result))
