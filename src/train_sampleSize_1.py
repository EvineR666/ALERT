import torch
import torchvision.datasets as datasets
import os
from model_PCA_MSST import ProtoNetAGAM
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import ConcatDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# road_types = ['changeLaneL', 'changeLaneR', 'turnL', 'turnR', 'roundabout', 'distractMotion']
road_types = ['head_turn']
# participants = ['GQY', 'Jason', 'Finesse', '然', 'Samishikude', 'molly', '陈昌忻', '熊宇轩']
participants = ['trainSize']

for rtype in road_types:
    road_type = rtype
    # create log txt
    log_path = 'Log/' + road_type + '.txt'

    with open(log_path, 'a') as f:
        text = 'road_type:{} \n'.format(road_type)
        f.write(text)

    if (road_type.find("distractMotion")) == -1:
        channel = 21
        total_classes = 2
    else:
        channel = 2
        total_classes = 4

    datadir = './P_' + participants[0] + '/MSSTFeature/' + road_type
    all_set = datasets.ImageFolder(datadir)

    n_val = int(len(all_set) * 0.2)
    n_train = len(all_set) - n_val
    trian_set, test_set = torch.utils.data.random_split(all_set, [n_train, n_val])

    # train size
    sizes = [i * 10 for i in range(1, len(trian_set) // 10 + 1)]
    for size in sizes:
        # write txt
        with open(log_path, 'a') as f:
            text = 'size:{} \n'.format(str(size))
            f.write(text)
        cutted_set, _ = torch.utils.data.random_split(trian_set, [size, len(trian_set) - size])

        if len(cutted_set) < 100:
            epochs = 50

        else:
            epochs = 100
        # epochs = 1

        batch_size = len(cutted_set) // 10

        train_loader = torch.utils.data.DataLoader(cutted_set, shuffle=True, batch_size=batch_size, num_workers=20,
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

            # print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
            #     train_subset)), train_acc / (len(train_subset))))
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

                # precision
                p = precision_score(total_true, total_pred, average='macro').item()
                # recall
                r = recall_score(total_true, total_pred, average='macro').item()
                # F score
                F1 = f1_score(total_true, total_pred, average='macro').item()
                # acc
                acc = accuracy_score(total_true, total_pred).item()

                # classification report
                target_names = all_set.classes
                report = classification_report(total_true, total_pred, target_names=target_names)

                print(report)

                if epoch + 1 == epochs:
                    with open(log_path, 'a') as f:
                        text = report
                        f.write(text)
