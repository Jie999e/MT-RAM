import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import time
import shutil
import pickle
import os
import xml.etree.ElementTree as ET

import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import skimage.feature

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from model import RecurrentAttention

from utils import plot_images
from utils import AverageMeter
from utils import denormalize
from utils import f_loss

#import cv2
from skimage import color


class CustomizedDataset(data.Dataset):
    def __init__(self, X, Y, MAPS):
        self.X = X
        self.Y = Y
        self.MAPS = MAPS

    def __getitem__(self,index):
        return self.X[index],self.Y[index],self.MAPS[index]

    def __len__(self):
        return len(self.Y)


def my_collate(batch):
    data = torch.cat([item[0].unsqueeze(0) for item in batch],axis=0)
    target = torch.Tensor([item[1] for item in batch])
    target = target.long()
    defect_map = torch.from_numpy(np.stack([item[2] for item in batch],axis=0))
    return [data, target, defect_map]


def get_train_valid_loader(
        dataset,
        batch_size,
        random_seed,
        fold=5,
        num_workers=0,
        pin_memory=True,
        collate_fn=my_collate
):
    train_loaders = []
    valid_loaders = []

    num_train = len(dataset)
    unit_size = num_train // fold
    indices = list(range(num_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    for i in range(fold):
        valid_idx = indices[i * unit_size: (i + 1) * unit_size]
        train_idx = indices[:i * unit_size]
        train_idx.extend(indices[(i + 1) * unit_size:])
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=my_collate,
        )

        valid_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=my_collate,
        )

        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)

    return (train_loaders, valid_loaders, num_train - unit_size)


def reset(batchsize, device, hidden_size):
    h1_t = torch.zeros(batchsize, hidden_size, dtype=torch.float, device=device, requires_grad=True)
    h2_t = torch.zeros(batchsize, hidden_size, dtype=torch.float, device=device, requires_grad=True)

    c1_t = torch.zeros(batchsize, hidden_size, dtype=torch.float, device=device, requires_grad=True)
    c2_t = torch.zeros(batchsize, hidden_size, dtype=torch.float, device=device, requires_grad=True)

    l_t = torch.FloatTensor(batchsize, 2).uniform_(-1, 1).to(device)
    l_t.requires_grad = True

    return h1_t, h2_t, c1_t, c2_t, l_t


# customized cmap
c=['#3F1005','#5A1708','#6C1C0A','#8B260F','#A12D13','#BD4024','#D85538','#E26B50','#E57E67','#E58F7B','#ECA291','#EFB7AA','#F4D1C9','#FEE5DF','#FCEEEB','darkcyan','c','turquoise','paleturquoise','lightcyan']

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

img_size = 100
num_of_types = 3
g=15
hidden_size = 256
max_num_glimpses=4
M = 1
dt_thres = 0.5
beta = 0.15

loaded = np.load('../data/fourier_bg.npz')
X = loaded['X']
y = loaded['y']
defect_map = loaded['defect_maps']

unique_labels = np.unique(y)
unique_labels = dict(zip(unique_labels, range(len(unique_labels))))
# y = np.array([unique_labels[i] for i in y])

num_of_types = len(unique_labels.keys())

unique_labels

transform_batch = transforms.Compose([
    transforms.ToTensor(),
    ])

imgs = [transform_batch(img).type(torch.FloatTensor) for img in X]
labels = y
defect_maps = defect_map

X_train, X_test, y_train, y_test, map_train, map_test = train_test_split(imgs, labels, defect_maps, test_size=0.3, random_state=66, stratify=labels)

# normalize the image data
X_mean = [np.mean(np.array([X_train[j][i].numpy() for j in range(len(X_train))])) for i in range(X_train[0].shape[0])]
X_std = [np.std(np.array([X_train[j][i].numpy() for j in range(len(X_train))])) for i in range(X_train[0].shape[0])]
transform_batch_2 = transforms.Compose([
    transforms.Normalize(X_mean, X_std),
    ])
X_train = [transform_batch_2(img) for img in X_train]
X_test = [transform_batch_2(img) for img in X_test]

train_dataset = CustomizedDataset(X_train, y_train, map_train)
test_dataset = CustomizedDataset(X_test, y_test, map_test)
train_loaders, valid_loaders, num_train = get_train_valid_loader(train_dataset,64,66,collate_fn=my_collate)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=False,num_workers=0,pin_memory=True,collate_fn=my_collate)

cv_acc = []
cv_fscore = []
cv_precision = []
cv_recall = []
cv_fscore_normal = []
cv_precision_normal = []
cv_recall_normal = []
cv_false_positive=[]
cv_glimpse_precision = []
cv_false_alarm_rate = []

for cv in range(len(train_loaders)):
    # initialize variables
    model = RecurrentAttention(g, 1, 2, 1, hidden_size//2, hidden_size//2, 0.05, hidden_size, num_of_types, [6, 12, 6], [3, 6], 3, 1, [img_size, img_size])
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-4
    )
    scheduler = ReduceLROnPlateau(
        optimizer, "min", patience = 20
    )

    losses_lst = []
    accs_lst = []
    precisions_lst = []
    glimpse_precisions_lst = []
    ps_lst = []
    rs_lst = []
    fps_lst= []

    for epoch in range(0,10):

        model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        precisions = AverageMeter()
        total_glimpse_precision = 0
        count_abnormal = 0

        tic = time.time()
        correct_dict = dict(zip(range(num_of_types),[0] * num_of_types))
        total_dict = dict(zip(range(num_of_types),[0] * num_of_types))
        wrong_matrix = np.zeros((num_of_types,num_of_types))

        with tqdm(total = num_train) as pbar:
            for i, (x, y, defect_map) in enumerate(train_loaders[cv]):
                optimizer.zero_grad()
                glimpse_precision_sum = 0
                x, y, defect_map = x.to(device), y.to(device), defect_map.to(device)

                # initialize location vector and hidden state
                batch_size = x.shape[0]
                h1_t, h2_t, c1_t, c2_t, l_t = reset(batch_size, device, hidden_size)
                log_pi = []
                baselines = []
                glimpse_precision1=torch.zeros((batch_size, max_num_glimpses), device=torch.device('cuda'))
                R_glimpse=torch.zeros((batch_size, max_num_glimpses), device=torch.device('cuda'))

                for t in range(max_num_glimpses - 1):
                    h1_t, c1_t, h2_t, c2_t, l_t, b_t, log_pi_l, glimpse_precision = model(x, l_t, h1_t, c1_t, h2_t, c2_t, defect_map)
                    #glimpse_precision_sum += torch.sum(glimpse_precision)
                    glimpse_precision_mean1 = torch.true_divide(glimpse_precision, g * g)
                    glimpse_precision1[:,t]=glimpse_precision_mean1
                    log_pi.append(log_pi_l)
                    baselines.append(b_t)

                h1_t, c1_t, h2_t, c2_t, l_t, b_t, log_probas, defect_map_prime, log_pi_l, glimpse_precision = model(x, l_t, h1_t, c1_t, h2_t, c2_t, defect_map, last = True)
                #glimpse_precision_sum += torch.sum(glimpse_precision)
                glimpse_precision_mean1 = torch.true_divide(glimpse_precision, g * g)
                glimpse_precision1[:, max_num_glimpses - 1] = glimpse_precision_mean1

                log_pi.append(log_pi_l)
                baselines.append(b_t)

                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)

                predicted = torch.max(log_probas, 1)[1]
                defect_map_predicted = (defect_map_prime > dt_thres).int()

                PRECISION, _, _, _ = f_loss(defect_map, defect_map_predicted.detach(), beta)

                for t in range(max_num_glimpses):
                    if t==0:
                        R_glimpse[:,t] = glimpse_precision1[:,t]
                    else:
                        R_glimpse[:,t]=R_glimpse[:,t-1]+glimpse_precision1[:,t]
                # calculate reward
                R = (predicted.detach() == y).float()+PRECISION
                R = R.unsqueeze(1).repeat(1, max_num_glimpses)+R_glimpse

                # compute losses for differentiable modules
                loss_action = F.nll_loss(log_probas, y)
                loss_detection, _, _, _ = f_loss(defect_map.float(), defect_map_prime.float(), beta)
                loss_detection = 1 - torch.mean(loss_detection)
                loss_baseline = F.mse_loss(baselines, R)

                # compute reinforce loss
                # summed over timesteps and averaged across batch
                adjusted_reward = R - baselines.detach()
                loss_reinforce = torch.sum(-log_pi[:, 0:-1] * adjusted_reward[:, 1:], dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)

                PRECISION = torch.mean(PRECISION, dim = 0)
                #glimpse_precision_mean = torch.true_divide(glimpse_precision_sum, batch_size * max_num_glimpses * g * g)
                if torch.sum(sum(y == j for j in [1,2]).bool()) > 0:
                    glimpse_precision_mean = torch.sum(torch.mean(
                            torch.squeeze(glimpse_precision1[torch.cat([(y == j).nonzero() for j in [1, 2]]), 1:]), -1))
                    total_glimpse_precision += glimpse_precision_mean
                    count_abnormal += torch.squeeze(glimpse_precision1[torch.cat([(y == j).nonzero() for j in [1, 2]]), 1:]).size(0)
                # sum up into a hybrid loss
                loss = loss_action+loss_baseline + loss_detection + loss_reinforce * 0.01

                # compute accuracy
                correct = torch.sum((predicted == y).float())
                acc = 100 * (correct/batch_size)

                for j in range(y.data.size()[0]):
                    total_dict[y.data[j].item()] = total_dict[y.data[j].item()] + 1
                    if (predicted[j]==y.data[j]).item():
                        correct_dict[y.data[j].item()] = correct_dict[y.data[j].item()] + 1
                    else:
                        wrong_matrix[y.data[j].item(),predicted[j].item()] += 1

                # store
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])
                precisions.update(PRECISION.item(), x.size()[0])
                #glimpse_precisions.update(glimpse_precision_mean.item(), x.size()[0])
                epsilon = 1e-7
                glimpse_precisions=total_glimpse_precision/(count_abnormal+epsilon)
                loss.backward()
                optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:2d}cv - {:.1f}s - loss: {:.3f} - acc: {:.3f} - precision: {:.3f} - glimpse precision: {:.3f}".format(
                            (cv + 1), (toc - tic), losses.avg, accs.avg, precisions.avg, glimpse_precisions.item()
                        )
                    )
                )
                pbar.update(batch_size)

                torch.cuda.empty_cache()

            if epoch == 299:
                cv_false_alarm_rate.append((wrong_matrix[0][1]+wrong_matrix[0][2])/total_dict[0])

        with torch.no_grad():
            losses = AverageMeter()
            accs = AverageMeter()
            precisions = AverageMeter()
            total_glimpse_precision = 0
            count_abnormal=0
            ps = AverageMeter()
            rs = AverageMeter()
            fps= AverageMeter()

            for i, (x, y, defect_map) in enumerate(valid_loaders[cv]):

                optimizer.zero_grad()
                glimpse_precision_sum = 0
                x, y, defect_map = x.to(device), y.to(device), defect_map.to(device)

                # initialize location vector and hidden state
                batch_size = x.shape[0]
                h1_t, h2_t, c1_t, c2_t, l_t = reset(batch_size, device, hidden_size)
                log_pi = []
                baselines = []
                glimpse_precision1=torch.zeros((batch_size, max_num_glimpses), device=torch.device('cuda'))
                R_glimpse=torch.zeros((batch_size, max_num_glimpses), device=torch.device('cuda'))

                for t in range(max_num_glimpses - 1):
                    h1_t, c1_t, h2_t, c2_t, l_t, b_t, log_pi_l, glimpse_precision = model(x, l_t, h1_t, c1_t, h2_t, c2_t, defect_map)
                    #glimpse_precision_sum += torch.sum(glimpse_precision)
                    glimpse_precision_mean1 = torch.true_divide(glimpse_precision, g * g)
                    glimpse_precision1[:,t]=glimpse_precision_mean1
                    log_pi.append(log_pi_l)
                    baselines.append(b_t)

                h1_t, c1_t, h2_t, c2_t, l_t, b_t, log_probas, defect_map_prime, log_pi_l, glimpse_precision = model(x, l_t, h1_t, c1_t, h2_t, c2_t, defect_map, last = True)
                #glimpse_precision_sum += torch.sum(glimpse_precision)
                glimpse_precision_mean1 = torch.true_divide(glimpse_precision, g * g)
                glimpse_precision1[:, max_num_glimpses - 1] = glimpse_precision_mean1

                log_pi.append(log_pi_l)
                baselines.append(b_t)

                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)

                predicted = torch.max(log_probas, 1)[1]
                defect_map_predicted = (defect_map_prime > dt_thres).int()

                PRECISION, p, r, fp = f_loss(defect_map, defect_map_predicted.detach(), beta)

                for t in range(max_num_glimpses):
                    if t==0:
                        R_glimpse[:,t] = glimpse_precision1[:,t]
                    else:
                        R_glimpse[:,t]=R_glimpse[:,t-1]+glimpse_precision1[:,t]
                # calculate reward
                R =  (predicted.detach() == y).float()+PRECISION
                R = R.unsqueeze(1).repeat(1, max_num_glimpses)+R_glimpse

                # compute losses for differentiable modules
                loss_action = F.nll_loss(log_probas, y)
                loss_detection, _, _, _ = f_loss(defect_map.float(), defect_map_prime.float(), beta)
                loss_detection = 1 - torch.mean(loss_detection)
                loss_baseline = F.mse_loss(baselines, R)

                # compute reinforce loss
                # summed over timesteps and averaged across batch
                adjusted_reward = R - baselines.detach()
                loss_reinforce = torch.sum(-log_pi[:, 0:-1] * adjusted_reward[:, 1:], dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)

                PRECISION = torch.mean(PRECISION, dim = 0)
                p = torch.mean(p, dim = 0)
                r = torch.mean(r, dim = 0)
                fp= torch.mean(fp, dim = 0)
                #glimpse_precision_mean = torch.true_divide(glimpse_precision_sum, batch_size * max_num_glimpses * g * g)
                if torch.sum(sum(y == j for j in [1, 2]).bool()) > 0:
                    glimpse_precision_mean = torch.sum(torch.mean(
                            torch.squeeze(glimpse_precision1[torch.cat([(y == j).nonzero() for j in [1, 2]]), 1:]), -1))
                    total_glimpse_precision += glimpse_precision_mean
                    count_abnormal += torch.squeeze(glimpse_precision1[torch.cat([(y == j).nonzero() for j in [1, 2]]), 1:]).size(0)
                # sum up into a hybrid loss
                loss = loss_action+loss_baseline + loss_detection + loss_reinforce * 0.01

                # compute accuracy
                correct = torch.sum((predicted == y).float())
                acc = 100 * (correct/batch_size)

                # store
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])
                precisions.update(PRECISION.item(), x.size()[0])
                ps.update(p.item(), x.size()[0])
                rs.update(r.item(), x.size()[0])
                fps.update(fp.item(), x.size()[0])
                #glimpse_precisions.update(glimpse_precision_mean.item(), x.size()[0])

            glimpse_precisions = total_glimpse_precision/count_abnormal
            losses_lst.append(losses.avg)
            accs_lst.append(accs.avg)
            precisions_lst.append(precisions.avg)
            glimpse_precisions_lst.append(glimpse_precisions.item())
            ps_lst.append(ps.avg)
            rs_lst.append(rs.avg)
            fps_lst.append(fps.avg)

            print("validation losses avg. " + str(losses.avg))
            print("validation accs avg. " + str(accs.avg))
            print("validation fscore avg. " + str(precisions.avg) + "/ precision avg. " + str(ps.avg) + "/ recall avg. " + str(rs.avg)+"/ false positive avg."+str(fps.avg))
            print("validation glimpse precisions avg." + str(glimpse_precisions.item()))
            print("epoch: " + str(epoch + 1))

            torch.cuda.empty_cache()

        scheduler.step(-precisions.avg)

#     PATH = "s50n5_cv" + str(cv + 1) + ".pt"
#     torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict':scheduler.state_dict(),
#             }, PATH)
    '''
    fig, axes = plt.subplots(2,2,figsize=(10,8))
    for i, ax in enumerate(axes.flat):
        if i==0:
            ax.plot(range(epoch+1),accs_lst)
            ax.set_xlabel("epoch")
            ax.set_ylabel("validation accuracy")
        if i==1:
            ax.plot(range(epoch+1),precisions_lst)
            ax.set_ylabel("validation precision")
            ax.set_xlabel("epoch")
        if i==2:
            ax.plot(range(epoch+1),glimpse_precisions_lst)
            ax.set_ylabel("validation glimpse precision")
            ax.set_xlabel("epoch")
        if i==3:
            ax.plot(range(epoch+1),losses_lst)
            ax.set_ylabel("validation loss")
            ax.set_xlabel("epoch")
    plt.show()
    '''
    # testing ...
    with torch.no_grad():
        num_test = len(test_dataset)
        num_normal = len(np.in1d(y_test, 0).nonzero()[0])
        num_abnormal = len(np.in1d(y_test, [1,2]).nonzero()[0])
        correct = 0
        test_precision = 0
        test_glimpse_precision = 0
        test_p = 0
        test_r = 0
        test_fp = 0
        test_precision1 = 0
        test_p1 = 0
        test_r1 = 0
        test_fp1 = 0

        correct_dict = dict(zip(range(num_of_types),[0] * num_of_types))
        total_dict = dict(zip(range(num_of_types),[0] * num_of_types))
        wrong_matrix = np.zeros((num_of_types,num_of_types))


        for i, (x, y, defect_map) in enumerate(test_loader):
            optimizer.zero_grad()

            x, y, defect_map = x.to(device), y.to(device), defect_map.to(device)

            # initialize location vector and hidden state
            batch_size = x.shape[0]
            h1_t, h2_t, c1_t, c2_t, l_t = reset(batch_size, device, hidden_size)
            log_pi = []
            baselines = []
            glimpse_precision1 = torch.zeros((batch_size, max_num_glimpses), device=torch.device('cuda'))

            for t in range(max_num_glimpses - 1):
                h1_t, c1_t, h2_t, c2_t, l_t, b_t, log_pi_l, glimpse_precision = model(x, l_t, h1_t, c1_t, h2_t, c2_t, defect_map)
                #test_glimpse_precision += torch.sum(glimpse_precision)
                glimpse_precision_mean1 = torch.true_divide(glimpse_precision, g * g)
                glimpse_precision1[:, t] = glimpse_precision_mean1
                log_pi.append(log_pi_l)
                baselines.append(b_t)

            h1_t, c1_t, h2_t, c2_t, l_t, b_t, log_probas, defect_map_prime, log_pi_l, glimpse_precision = model(x, l_t, h1_t, c1_t, h2_t, c2_t, defect_map, last = True)
            #test_glimpse_precision += torch.sum(glimpse_precision)
            glimpse_precision_mean1 = torch.true_divide(glimpse_precision, g * g)
            glimpse_precision1[:, max_num_glimpses - 1] = glimpse_precision_mean1

            log_pi.append(log_pi_l)
            baselines.append(b_t)

            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            predicted = torch.max(log_probas, 1)[1]
            defect_map_predicted = (defect_map_prime > dt_thres).int()

            if torch.sum(sum(y == j for j in [1, 2]).bool()) > 0:
                abnormal_defect_map = torch.squeeze(defect_map[torch.cat([(y == j).nonzero() for j in [1,2]]), :, :])
                abnormal_defect_map_predicted=torch.squeeze(defect_map_predicted[torch.cat([(y == j).nonzero() for j in [1,2]]), :, :])
                PRECISION, p, r, fp = f_loss(abnormal_defect_map, abnormal_defect_map_predicted.detach(), beta)
                PRECISION = torch.sum(PRECISION, dim=0)
                test_precision += PRECISION
                p = torch.sum(p, dim=0)
                test_p += p
                r = torch.sum(r, dim=0)
                test_r += r
                fp = torch.sum(fp, dim=0)
                test_fp += fp
            if torch.sum(sum(y == j for j in [0]).bool()) > 0:
                normal_defect_map = torch.squeeze(defect_map[(y == 0).nonzero(), :, :])
                normal_defect_map_predicted=torch.squeeze(defect_map_predicted[(y == 0).nonzero(), :, :])
                PRECISION1, p1, r1, fp1 = f_loss(normal_defect_map, normal_defect_map_predicted.detach(), beta)
                PRECISION1 = torch.sum(PRECISION1, dim=0)
                test_precision1 += PRECISION1
                p1 = torch.sum(p1, dim=0)
                test_p1 += p1
                r1 = torch.sum(r1, dim=0)
                test_r1 += r1
                fp1 = torch.sum(fp1, dim=0)
                test_fp1 += fp1

            if torch.sum(sum(y == j for j in [1, 2]).bool()) > 0:
                glimpse_precision_mean = torch.sum(torch.mean(
                        torch.squeeze(glimpse_precision1[torch.cat([(y == j).nonzero() for j in [1, 2]]), 1:]), -1))
                test_glimpse_precision+=glimpse_precision_mean


            correct += torch.sum((predicted == y).float())


            for j in range(y.data.size()[0]):
                total_dict[y.data[j].item()] = total_dict[y.data[j].item()] + 1
                if (predicted[j]==y.data[j]).item():
                    correct_dict[y.data[j].item()] = correct_dict[y.data[j].item()] + 1
                else:
                    wrong_matrix[y.data[j].item(),predicted[j].item()] += 1


        perc = (100.0 * correct) / (num_test)
        error = 100 - perc
        print("testing cv" + str(cv + 1) + " ... ")
        print(
            "[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)".format(
                correct, num_test, perc, error
            )
        )
        print("[*] Test fscore: {:.2f}% / Test precision: {:.2f}% / Test recall: {:.2f}% / Test false positive: {:.2f}%".format(100 * test_precision / num_abnormal, 100 * test_p / num_abnormal, 100 * test_r / num_abnormal, 100*test_fp/num_abnormal))

        print(
            "[*] Normal Test fscore: {:.2f}% / Normal Test precision: {:.2f}% / Normal Test recall: {:.2f}% / Normal Test false positive: {:.2f}%".format(
                100 * test_precision1 / num_normal, 100 * test_p1 / num_normal, 100 * test_r1/ num_normal, 100*test_fp1/num_normal))

        print("[*] Test GLIMPSE PRECISION: {:.2f}%".format(100 * torch.true_divide(test_glimpse_precision,  num_abnormal)))
        for label in unique_labels.keys():
            print("Type: " + str(label) + "\t Total: " +str(total_dict[unique_labels[label]]) + "\t Correct Ratio: " + str(round(correct_dict[unique_labels[label]]/total_dict[unique_labels[label]], 2)))

        cv_acc.append((perc / 100).item())
        cv_fscore.append((test_precision / num_abnormal).item())
        cv_precision.append((test_p / num_abnormal).item())
        cv_recall.append((test_r / num_abnormal).item())
        cv_fscore_normal.append((test_precision1 / num_normal).item())
        cv_precision_normal.append((test_p1 / num_normal).item())
        cv_recall_normal.append((test_r1 / num_normal).item())
        cv_glimpse_precision.append(torch.true_divide(test_glimpse_precision,  num_abnormal).item())

r_accs = np.zeros((5, 8))
r_accs[:,0] = np.array(cv_acc)
r_accs[:,1] = np.array(cv_fscore)
r_accs[:,2] = np.array(cv_precision)
r_accs[:,3] = np.array(cv_recall)
r_accs[:,4] = np.array(cv_fscore_normal)
r_accs[:,5] = np.array(cv_precision_normal)
r_accs[:,6] = np.array(cv_recall_normal)
r_accs[:,7] = np.array(cv_glimpse_precision)
np.savetxt("results/result.csv", np.stack([np.mean(r_accs,axis=0),np.std(r_accs,axis=0)],axis=0), delimiter=',')

# r_accs = np.zeros((5, 1))
# r_accs[:,0] = np.array(cv_false_alarm_rate)
#
# np.savetxt("false_alarm_rate/far.csv", np.mean(r_accs,axis=0), delimiter=',')

sample_loader = DataLoader(
    test_dataset,
    batch_size=9,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=my_collate,
)

data_iter = iter(sample_loader)
images, ys, defect_map = data_iter.__next__()
X = images.numpy()
X = np.transpose(X, [0, 2, 3, 1])
plot_images(X, ys, defect_map, c)
X = torch.from_numpy(np.transpose(X, [0, 3, 1, 2]))

x = X.to(device)
defect_map = defect_map.to(device)
y = ys

# initialize location vector and hidden state
batch_size = x.shape[0]
h1_t, h2_t, c1_t, c2_t, l_t = reset(batch_size, device, hidden_size)

glimpses = torch.zeros((batch_size, max_num_glimpses, 4), dtype=torch.long, device=device)
glimpses[:, 0, :2] = denormalize(x.shape[-1], l_t).long()
glimpses[:, 0, 2:] = (denormalize(x.shape[-1], l_t) + g - 1).long()

for t in range(max_num_glimpses - 1):
    h1_t, c1_t, h2_t, c2_t, l_t, b_t, log_pi_l, _ = model(x, l_t, h1_t, c1_t, h2_t, c2_t, defect_map)
    glimpses[:, t + 1, :2] = denormalize(x.shape[-1], l_t).long()
    glimpses[:, t + 1, 2:] = (denormalize(x.shape[-1], l_t) + g - 1).long()

h1_t, c1_t, h2_t, c2_t, l_t, b_t, log_probas, defect_map_prime, log_pi_l, _ = model(x, l_t, h1_t, c1_t, h2_t, c2_t,
                                                                                    defect_map, last=True)

predicted = torch.max(log_probas, 1)[1]
defect_map_predicted = (defect_map_prime > dt_thres).int()

x = x.cpu().numpy()
x = np.transpose(x, [0, 2, 3, 1])

plot_images(x, ys, defect_map.cpu(), c)

glimpses = glimpses.cpu()
fig, axes = plt.subplots(nrows=batch_size, ncols=3, figsize=(10, 30))

for i, ax in enumerate(axes.flat):
    if (i % 3) == 0:
        ax.imshow(defect_map[i // 3].cpu().squeeze(), cmap='gray')
        ax.set_xlabel("{}".format(int(y[i // 3])))
        ax.set_xticks([])
        ax.set_yticks([])
    if (i % 3) == 1:
        ax.imshow(defect_map_predicted.cpu()[i // 3].squeeze(), cmap='gray')
        ax.set_xlabel("{}".format(int(predicted[i // 3])))
        ax.set_xticks([])
        ax.set_yticks([])
    if (i % 3) == 2:
        ax.imshow(((x[i // 3] - np.min(x[i // 3])) / (np.max(x[i // 3]) - np.min(x[i // 3]))).squeeze(), cmap='gray')
        ax.set_xlabel("{}".format(int(predicted[i // 3])))
        ax.set_xticks([])
        ax.set_yticks([])
        for ii in range(max_num_glimpses):
            rect = patches.Rectangle((glimpses[i // 3][ii][0], glimpses[i // 3][ii][1]),
                                     glimpses[i // 3][ii][2] - glimpses[i // 3][ii][0],
                                     glimpses[i // 3][ii][3] - glimpses[i // 3][ii][1], linewidth=1, edgecolor=c[ii],
                                     facecolor='none')
            ax.add_patch(rect)

plt.show()
