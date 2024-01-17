import torch
import torchvision.datasets as datasets
import os
from model_PCA_MSST import ProtoNetAGAM
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import ConcatDataset
import handle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# road_types = ['changeLaneL', 'changeLaneR', 'turnL', 'turnR', 'roundabout', 'distractMotion']
road_types = ['changeLaneR']
# participants = ['GQY', 'Jason', 'Samishikude', 'molly', '陈昌忻', '熊宇轩']
participants = ['陈昌忻']
a_acc_all = {}
a_result = {}
a_recall = {}
a_presicion = {}
a_f1 = {}

for rtype in road_types:
    acc_all = {}
    result2 = {}
    recall_all = {}
    precision_all = {}
    f1_all = {}

    road_type = rtype
    # create log txt
    log_path = 'Log/dataSize_DM' + road_type + '.txt'

    with open(log_path, 'a') as f:
        text = 'road_type:{} \n'.format(road_type)
        f.write(text)

    if (road_type.find("distractMotion")) == -1:
        channel = 21
        total_classes = 2
    else:
        continue
        channel = 2
        total_classes = 4
    parti=road_type
    print('--{}--'.format(parti))
    root = 'result_train_size2/result_distract_' + parti
    if not os.path.exists(root):
        os.mkdir(root)
    datadir = './P_' + participants[0] + '/MSSTFeature_new/' + road_type
    all_set = datasets.ImageFolder(datadir)
    # read .mat through all participants
    for parti_index in range(1, len(participants)):
        datadir = './P_' + participants[parti_index] + '/MSSTFeature_new/' + road_type
        if os.path.isdir(datadir):
            temp_set = datasets.ImageFolder(datadir)
            all_set = ConcatDataset([all_set, temp_set])
        else:
            continue

    n_val = int(len(all_set) * 0.2)
    n_train = len(all_set) - n_val
    trian_set, test_set = torch.utils.data.random_split(all_set, [n_train, n_val])

    # data size
    # sizes = [i * 25 for i in range(1, len(trian_set) // 25 + 1)]
    sizes = [i * 25 for i in range(1, 1000 // 25 + 1)]
    for size in sizes:

        acc_t = []
        result_pred = []
        result_true = []
        recall_t = []
        precision_t = []
        f1_t = []
        # write txt
        with open(log_path, 'a') as f:
            text = 'size:{} \n'.format(str(size))
            f.write(text)

        concat_data_cutted, _ = torch.utils.data.random_split(all_set, [size, len(all_set) - size])

        # if len(concat_data_cutted) < 100:
        #     epochs = 50
        #
        # else:
        #     epochs = 100
        epochs = 10

        batch_size = len(concat_data_cutted) // 10
        if batch_size >= 48:
            batch_size = 48

        train_loader = torch.utils.data.DataLoader(concat_data_cutted, shuffle=True, batch_size=batch_size, num_workers=20,
                                                   pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=20,
                                                  pin_memory=True)
        model = ProtoNetAGAM(total_class=total_classes, channel=channel).cuda()
        optimizer = torch.optim.Adam(model.parameters())  # weight_decay
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)
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
                # batch_x, batch_y = Variable(batch_x).float(), Variable(batch_y)
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

            #print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
            #   train_subset)), train_acc / (len(train_subset))))
            # ==============evaluation====================
            model.eval()
            eval_loss = 0.
            total_pred = []
            total_true = []

            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x, batch_y = Variable(batch_x).float().cuda(), Variable(batch_y).cuda()
                    # batch_x, batch_y = Variable(batch_x).float(), Variable(batch_y)
                    result = model(batch_x)
                    loss = loss_func(result, batch_y)
                    eval_loss += loss.item()
                    pred = torch.max(result, 1)[1]
                    total_pred.extend(pred.tolist())
                    total_true.extend(batch_y.tolist())

                # classification report
                # target_names = temp_set .classes
                # report = classification_report(total_true, total_pred, target_names=target_names)
                #
                # print(report)
                # # precision
                p = precision_score(total_true, total_pred, average='macro').item()
                # # recall
                r = recall_score(total_true, total_pred, average='macro').item()
                # # F score
                F1 = f1_score(total_true, total_pred, average='macro').item()
                # # acc
                acc = accuracy_score(total_true, total_pred).item()

                if epoch + 1 == epochs:
                    acc_t.append(acc)
                    precision_t.append(p)
                    recall_t.append(r)
                    f1_t.append(F1)
                    result_pred.append(total_pred)
                    result_true.append(total_true)
                    # with open(log_path, 'a') as f:
                    #     text = report
                    #     f.write(text)
        acc_all[size] = acc_t
        precision_all[size] = precision_t
        recall_all[size] = recall_t
        f1_all[size] = f1_t
        result2[size] = {'pred': result_pred,
                              'true': result_true}
    a_acc_all[parti] = acc_all
    a_presicion[parti] = precision_all
    a_f1[parti] = f1_all
    a_recall[parti] = recall_all
    a_result[parti] = result2

    with open(root + '/acc.txt', 'w') as f:
        f.write(str(a_acc_all))
    with open(root + '/result.txt', 'w') as f:
        f.write(str(a_result))
    with open(root + '/precision.txt', 'w') as f:
        f.write(str(a_presicion))
    with open(root + '/recall.txt', 'w') as f:
        f.write(str(a_recall))
    with open(root + '/f1.txt', 'w') as f:
        f.write(str(a_f1))

root = 'result_train_size'
handle.result4(root)