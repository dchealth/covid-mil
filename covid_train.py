import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
from covid_data import COVIDDataset
import pickle
import mil_tool
from augment_data import default_2D_augmentation_params as aug_params
from augment_data import get_patch_augmentation

best_acc = 0


def main(args):
    global best_acc
    # result path basename
    csv_path = f'convergence.csv'
    model_path = f'checkpoint_best.pth'

    # cnn
    # model = models.resnet34(True)
    print('CNN network: ', args.network)
    model = torch.hub.load(
        'pytorch/vision:v0.3.0', args.network, pretrained=False, num_classes=2)
    # model.fc = nn.Linear(model.fc.in_features, 2)

    try:
        fname = os.path.join(args.output, model_path)
        model.load_state_dict(torch.load(fname)['state_dict'])
        print(f'load parameters: {fname}')
    except Exception as e:
        print(f'fail load parameters: {e}')
    model.cuda()

    if args.weights == 0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1 - args.weights, args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    cudnn.benchmark = True

    with open(args.plan, 'rb') as f:
        plan = pickle.load(f)
    # The image folder is assumed to be the same as plan's folder
    data_folder = os.path.dirname(args.plan)

    # normalization
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    train_trans = transforms.Compose([
        # get_patch_augmentation(plan['tile'][1:], aug_params),
        torch.FloatTensor,
        normalize
    ])
    valid_trans = transforms.Compose([torch.FloatTensor, normalize])

    # load data
    # patch = (np.array(plan['tile']) * [1, 1.45, 1.45]).astype(np.int)
    patch = (np.array(plan['tile']) * [1, 1, 1]).astype(np.int)

    train_dset = COVIDDataset(
        plan=plan,
        data_folder=data_folder,
        target_key=args.target_key,
        transform=train_trans,
        data_type='train',
        fold=args.fold,
        patch=patch)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False)

    if args.val_lib:
        val_dset = COVIDDataset(
            plan=plan,
            data_folder=data_folder,
            transform=valid_trans,
            target_key=args.target_key,
            data_type='valid',
            fold=args.fold)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False)

    # open output file
    fconv = open(os.path.join(args.output, csv_path), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    print('Begin to train')
    # loop throuh epochs
    for epoch in range(args.nepochs):
        train_dset.setmode(1)
        print(f'Inference\tBatch: [{len(train_loader)}]\t', end='')
        probs = mil_tool.inference(epoch, train_loader, model, args)
        with open(os.path.join(args.output, 'tile_probs_train.pkl'),
                  'wb') as f:
            pickle.dump(probs, f)
        topk = mil_tool.group_argtopk(
            np.array(train_dset.slideIDX), probs, args.k)
        print(f'TopK: [{len(topk)}]')
        train_dset.maketraindata(topk)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        print(f'Training\tEpoch: [{epoch + 1}/{args.nepochs}]\t', end='')
        loss = mil_tool.train(epoch, train_loader, model, criterion, optimizer)
        print(f'Loss: {loss}')
        fconv = open(os.path.join(args.output, csv_path), 'a')
        fconv.write('{},loss,{}\n'.format(epoch + 1, loss))
        fconv.close()

        # Validation
        if args.val_lib and (epoch + 1) % args.test_every == 0:
            val_dset.setmode(1)
            print('Validation\t', end='')
            probs = mil_tool.inference(epoch, val_loader, model, args)
            with open(os.path.join(args.output, 'tile_probs_val.pkl'),
                      'wb') as f:
                pickle.dump(probs, f)
            maxs = mil_tool.group_max(
                np.array(val_dset.slideIDX), probs, len(val_dset.targets))
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            err, fpr, fnr = mil_tool.calc_err(pred, val_dset.targets)
            acc = np.sum(
                np.array(pred) == np.array(val_dset.targets)) / len(pred)
            print('Epoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}\tAccuracy: {}'.
                  format(epoch + 1, args.nepochs, err, fpr, fnr, acc))
            fconv = open(os.path.join(args.output, csv_path), 'a')
            fconv.write('{},error,{}\n'.format(epoch + 1, err))
            fconv.write('{},fpr,{}\n'.format(epoch + 1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch + 1, fnr))
            fconv.write('{},accuracy,{}\n'.format(epoch + 1, acc))
            fconv.close()
            # Save best model
            err = (fpr + fnr) / 2.
            if 1 - err >= best_acc:
                print('Save model...')
                best_acc = 1 - err
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output, model_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MIL-nature-medicine-2019 tile classifier training script')
    parser.add_argument(
        '--train_lib',
        type=str,
        default='yes',
        help='path to train MIL library binary')
    parser.add_argument(
        '--val_lib',
        type=str,
        default='yes',
        help='path to validation MIL library binary. If present.')
    parser.add_argument(
        '--output', type=str, default='.', help='name of output file')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='mini-batch size (default: 512)')
    parser.add_argument(
        '--nepochs', type=int, default=100, help='number of epochs')
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        help='number of data loading workers (default: 4)')
    parser.add_argument(
        '--test_every',
        default=10,
        type=int,
        help='test on val every (default: 10)')
    parser.add_argument(
        '--weights',
        default=0.5,
        type=float,
        help='unbalanced positive class weight (default: 0.5, balanced classes)'
    )
    parser.add_argument(
        '--k',
        default=1,
        type=int,
        help=
        'top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)'
    )
    parser.add_argument('--plan', type=str, default='', help='plan file')
    parser.add_argument(
        '--balance_class',
        default=False,
        type=bool,
        help='balance class by copy case')
    parser.add_argument('--fold', default=0, type=int, help='fold to train')
    parser.add_argument(
        '--target_key',
        default='ICU',
        type=str,
        help='target key, ICU/Stay/Transfer')
    parser.add_argument('--network', default='resnet34', type=str, help='which model to use, defaut: resnet34')


    args = parser.parse_args()

    # print(f'train_num:{args.train_num}')
    main(args)
