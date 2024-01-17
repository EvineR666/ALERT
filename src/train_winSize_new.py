import torch
import torchvision.datasets as datasets
import numpy as np
import os
import sys
import statistics
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

def main(participants, road_type):

    # ====== config =======
    k_fold = 5
    epochs = 50
    winSize_number = 13

    for parti in participants:
        participant = 'P_' + parti

        # ============ config =============
        root = 'result/winSize' + parti + '/'
        if not os.path.exists(root):
            os.makedirs(root)

        road_type_list = [road_type]
        for road_type in road_type_list:
            print(road_type)

            if (road_type.find("distractMotion")) == -1:
                channel = 21
                total_classes = 2
            else:
                channel = 2
                total_classes = 4

            # record performance
            datadir = './' + participant + '/MSSTFeature_new/' + road_type


            orig_set = datasets.ImageFolder(datadir)

            with open(root + '/{}_{}_winSize_acc.txt'.format(parti, road_type), 'w') as f:
            # =========================================== each win size ===============================================
                for winSize in range(1, winSize_number):

                    # acc_t = []

                    kf = KFold(n_splits=k_fold, shuffle=True, random_state=2)
                    k_index = 0
                    # ========================= k fold ===========================
                    for train_index, test_index in kf.split(orig_set):

                        train_subset = torch.utils.data.dataset.Subset(orig_set, train_index)
                        test_subset = torch.utils.data.dataset.Subset(orig_set, test_index)
                        train_loader = torch.utils.data.DataLoader(train_subset, shuffle=True, batch_size=5, num_workers=8,
                                                                   pin_memory=True)
                        test_loader = torch.utils.data.DataLoader(test_subset, shuffle=True, batch_size=5, num_workers=8,
                                                                  pin_memory=True)
                        model = ProtoNetAGAM(total_class=total_classes, channel=channel).cuda()
                        optimizer = torch.optim.Adam(model.parameters())  # weight_decay
                        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.5)
                        loss_func = torch.nn.CrossEntropyLoss().cuda() # multiclass
                        test_round = 0
                        best_acc = 0
                        # ==============train==============
                        for epoch in range(epochs):
                            print('epoch {}'.format(epoch + 1))
                            model.train()
                            train_loss = 0.
                            train_acc = 0.
                            # =========================each batch================================
                            for i, (batch_x, batch_y) in enumerate(train_loader):
                                # 对lr进行调整
                                # mask
                                batch_x[:, :, :, 5 * winSize:] = torch.zeros(
                                    (batch_x.shape[0], batch_x.shape[1], batch_x.shape[2],
                                     batch_x.shape[3] - (5 * winSize)))
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
                            # ==================evaluation====================
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
                                print('precision: {:.6f}  recall: {:.6f}  F score: {:.6f}  acc: {:.6f}\n'
                                      .format(p, r, F1, acc))
                                if acc > best_acc:
                                    best_acc = acc
                                if epoch + 1 == epochs:
                                    f.write("WinSize = " + str(winSize) + ";  fold = " + str(k_index) + ";  Acc = " + str(best_acc) + "\n")

                            test_round = test_round + 1
                            # ======================end echo=========================

                        k_index = k_index + 1
                        # =============================end k fold=================================

                    # f.write("WinSize = "+str(winSize)+";  Acc = "+str(statistics.mean(acc_t))+"\n")


if __name__ == '__main__':
    # participants = ['Jason', 'GQY', 'Finesse', '然', 'Samishikude', 'molly', '陈昌忻', '熊宇轩']
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    participants = ['GQY']
    road_type = 'changeLaneL'
    if len(sys.argv) > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[2])
        participants = []
        participants.append(sys.argv[1])
        road_type = sys.argv[3]
    main(participants, road_type)
