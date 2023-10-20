import torch
from sonar_loader import *
from sklearn.model_selection import KFold
from model import *
from eval import *
import numpy as np
import tqdm
import csv
import os
import math
from loss import FocalLoss, CBFocalLoss
from torch.utils.data import ConcatDataset
import sys
import segmentation_models_pytorch as smp

import warnings
warnings.filterwarnings(action='ignore')

pre_path = './checkpoints/'
save_csv = True
csv_path = './results/'

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def train_function(data, model, optimizer, loss_function, scheduler, device):
    model.train()
    epoch_loss = 0

    for index, sample_batch in enumerate(tqdm.tqdm(data)):
        imgs = sample_batch['image']
        gt_mask = sample_batch['mask']

        imgs = imgs.to(device)
        true_masks = gt_mask.to(device)

        outputs = model(imgs)

        # prediction vis
        probs = torch.softmax(outputs, dim=1)
        masks_pred = torch.argmax(probs, dim=1)

        loss = loss_function(outputs, true_masks)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch finished ! Loss: {epoch_loss / index:.4f}, lr:{scheduler.get_last_lr()}')

def validation_epoch(model, val_loader, num_class, device, epoch, model_name, fold):
    global history
    class_iou, mean_iou, cf = eval_net_loader(model, val_loader, num_class, device, epoch)
    if epoch == 1:
        history = np.expand_dims(class_iou.copy(),axis=1)
    else:
        history = np.concatenate((history, np.expand_dims(class_iou.copy(),axis=1)))

    print('Class IoU:', ' '.join(f'{x:.4f}' for x in class_iou), f'  |  Mean IoU: {mean_iou:.4f}')
    if save_csv and epoch == 'test':
        createFolder(f'{csv_path}/{model_name}')
        with open(f'{csv_path}/{model_name}/{dataset}_gpu{gpu}_{iter}_{fold+1}.csv', 'w', newline='') as f:
            w = csv.writer(f, delimiter='\n')
            w.writerow(class_iou)
            w.writerow([mean_iou])
            w.writerow(['history'])
            w = csv.writer(f, delimiter=',')
            w.writerows(history)
            w.writerow(['Confusion Matrix'])
            w.writerows(cf)
    return mean_iou

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))

def main(mode='', gpu_id=0, num_epoch=31, train_batch_size=2, test_batch_size=1, classes=[], pretrained=False, save_path='', model_name = 'unet', loss_fn = torch.nn.CrossEntropyLoss(), dataset = '', iter=0):
    lr = 0.001
    save_term = 10
    fold_num = 5
    dir_checkpoint = './checkpoints/'

    device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {str(device)}\n')
    data_path = './data/' + dataset

    print(f'model: {model_name}')
    print(f'dataset: {dataset}')
    print(f'iter: {iter}')

    total_dataset = sonarDataset('./data/total', classes)
    num_val = len(total_dataset) // 5
    new_dataset = [] if dataset == 'real' else sonarDataset(data_path, classes)
    kf = KFold(n_splits=fold_num, shuffle=True, random_state=iter)

    for fold, (train_idx, val_idx) in enumerate(kf.split(total_dataset)):
        print(f'fold: {fold+1}')
        # trainset과 validation set을 분리
        dataset_train = torch.utils.data.Subset(total_dataset, train_idx[num_val:]) # 1316 - 200 = 1116
        dataset_val = torch.utils.data.Subset(total_dataset, train_idx[:num_val])
        dataset_test = torch.utils.data.Subset(total_dataset, val_idx)
        
        # 만약 new_data를 추가할 경우
        dataset_train = torch.utils.data.ConcatDataset([dataset_train, new_dataset])

        data_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=0,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=test_batch_size, shuffle=True, num_workers=0
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=0
        )

        if model_name == 'resnet18':
            model = ResNetUNet(in_channels=1, n_classes=len(classes), encoder=models.resnet18).to(device).train() 
        elif model_name == 'resnet34':
            model = ResNetUNet(in_channels=1, n_classes=len(classes), encoder=models.resnet34).to(device).train() 
        elif model_name == 'resnet50':
            model = DeepResUnet(in_channels=1, n_classes=len(classes), encoder=models.resnet50).to(device).train()
        elif model_name == 'resnet101':
            model = DeepResUnet(in_channels=1, n_classes=len(classes), encoder=models.resnet101).to(device).train()
        elif model_name == 'resnet152':
            model = DeepResUnet(in_channels=1, n_classes=len(classes), encoder=models.resnet152).to(device).train()
        elif model_name == 'vgg16':
            model = VGGUnet(in_channels=1, n_classes=len(classes), encoder=models.vgg16).to(device).train()
        elif model_name == 'vgg19':
            model = VGGUnet(in_channels=1, n_classes=len(classes), encoder=models.vgg19).to(device).train()
        elif model_name == 'unet':
            model = UNet(in_channels=1, n_classes=len(classes)).to(device).train()

        # model = smp.PAN(encoder_name='resnet18',
        #                  encoder_weights=None,
        #                  in_channels=1,
        #                  classes=len(classes)).to(device).train()
        model.apply(init_weights)
        
        if 'train' in mode:
            if pretrained:
                model.load_state_dict(torch.load(pre_path+f'best_model{gpu_id}.pth'))
                print('Model loaded from {}'.format(pre_path+f'best_model{gpu_id}.pth'))

            print('Starting training:\n'
                f'Epochs: {num_epoch}\n'
                f'Batch size: {train_batch_size}\n'
                f'Learning rate: {lr}\n'
                f'Training size: {len(data_loader.dataset)}\n')


            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            loss_function = loss_fn   # torch.nn.CrossEntropyLoss()

            max_score = 0
            max_score_epoch = 0
            for epoch in range(1, num_epoch+1):
                print('*** Starting epoch {}/{}. ***'.format(epoch, num_epoch))

                train_function(data_loader, model, optimizer, loss_function, lr_scheduler, device)
                lr_scheduler.step()

                mean_iou = validation_epoch(model, data_loader_val, len(classes), device, epoch, model_name, fold)

                state_dict = model.state_dict()
                if device == "cuda":
                    state_dict = model.module.state_dict()
                if epoch % save_term == 0:
                    state_dict = model.state_dict()
                    if device == "cuda":
                        state_dict = model.module.state_dict()
                    torch.save(state_dict, dir_checkpoint + f'{epoch}.pth')
                    print('Checkpoint epoch: {} saved !'.format(epoch))
                if max_score < mean_iou:
                    max_score = mean_iou
                    max_score_epoch = epoch
                    print('Best Model saved!')
                    torch.save(state_dict, dir_checkpoint + f'best_model{gpu_id}.pth')
                print('****************************')
            print('*** Test ***')
            model.load_state_dict(torch.load(dir_checkpoint + f'best_model{gpu_id}.pth'))
            validation_epoch(model, data_loader_test, len(classes), device, 'test', model_name, fold)
            print()

if __name__ =="__main__":

    gpu = '0' #sys.argv[1] if len(sys.argv) == 2 else 0
    if gpu == '0':
        datasets = ['real']
    elif gpu == '1':
        datasets = ['heatmap_unbalance_wo']
    elif gpu == '2':
        datasets = ['obj_unbalance_Simple_Crop']
    elif gpu == '3':
        datasets = ['obj_unbalance_Simple']
        
    CLASSES = ['background', 'bottle', 'can', 'chain',
                'drink-carton', 'hook', 'propeller', 'shampoo-bottle',
                'standing-bottle', 'tire', 'valve', 'wall']

    for iter in range(1, 10+1):
        for model_name in ['resnet18']:#['resnet18','resnet34','resnet50', 'unet','vgg16','vgg19','resnet101','resnet152']: #['resnet101','resnet152','resnet18','resnet34','resnet50','vgg16','vgg19']:
            batch_size = 4
            if model_name == 'resnet18':
                batch_size = 16
            elif model_name == 'resnet34':
                batch_size = 16
            elif model_name == 'resnet50':
                batch_size = 8
            elif model_name == 'resnet101':
                batch_size = 4
            elif model_name == 'resnet152':
                batch_size = 4
            elif model_name == 'vgg16':
                batch_size = 4
            elif model_name == 'vgg19':
                batch_size = 4
            elif model_name == 'unet':
                batch_size = 4
            batch_size = 16
            for dataset in datasets:
                history = np.array([])
                main(mode='train', gpu_id=gpu, num_epoch=50,
                    train_batch_size=batch_size, test_batch_size=1, classes=CLASSES,
                    pretrained=False, save_path='', loss_fn=torch.nn.CrossEntropyLoss(),
                    model_name=model_name, dataset=dataset, iter=iter)
