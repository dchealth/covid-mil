import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
import pickle
import mil_tool

from covid_data import COVIDDataset


def main(args):
    # load model
    # model = models.resnet34(True)
    print('CNN network: ', args.network)
    model = torch.hub.load(
        'pytorch/vision:v0.3.0', args.network, pretrained=False, num_classes=2)
    # model.fc = nn.Linear(model.fc.in_features, 2)
    ch = torch.load(args.model)
    model.load_state_dict(ch['state_dict'])
    model = model.cuda()
    cudnn.benchmark = True

    with open(args.plan, 'rb') as f:
        plan = pickle.load(f)
    data_folder = os.path.dirname(args.plan)

    # normalization
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    trans = transforms.Compose([torch.FloatTensor, normalize])

    # load data
    dset = COVIDDataset(
        plan=plan,
        data_folder=data_folder,
        target_key=args.target_key,
        transform=trans,
        data_type=args.dataset)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False)

    dset.setmode(1)
    probs = mil_tool.inference(0, loader, model, args)
    with open(
            os.path.join(args.output, f'tile_probs_{args.dataset}.pkl'),
            'wb') as f:
        pickle.dump(probs, f)

    maxs = mil_tool.group_max(
        np.array(dset.slideIDX), probs, len(dset.targets))
    pred = [1 if x >= 0.5 else 0 for x in maxs]
    real = dset.targets
    pred = np.array(pred)
    real = np.array(real)
    err, fpr, fnr = mil_tool.calc_err(pred, real)
    neq = np.not_equal(pred, real)

    fp = open(
        os.path.join(args.output, f'predictions_{args.dataset}.csv'), 'w')
    fp.write('file,target,prediction,probability,mark\n')
    for name, target, prob in zip(dset.slide_names, dset.targets, maxs):
        if target != int(prob >= 0.5):
            if target == 1:
                fp.write('{},{},{},{},{}\n'.format(name, target,
                                                   int(prob >= 0.5), prob,
                                                   'fn'))
            else:
                fp.write('{},{},{},{},{}\n'.format(name, target,
                                                   int(prob >= 0.5), prob,
                                                   'fp'))
        else:
            fp.write('{},{},{},{}\n'.format(name, target, int(prob >= 0.5),
                                            prob))
    fp.write(
        f'error,{err},{neq.sum()}/{pred.shape[0]}\n'
        f'fpr,{fpr},{np.logical_and(pred==1,neq).sum()}/{(real==0).sum()}'
        f'\nfnr,{fnr},{np.logical_and(pred==0,neq).sum()}/{(real==1).sum()}')
    fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='inference result for the best model')
    parser.add_argument(
        '--plan', type=str, default='plan.pkl', help='path to plan file')
    parser.add_argument(
        '--output', type=str, default='.', help='name of output directory')
    parser.add_argument(
        '--model', type=str, default='', help='path to pretrained model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='how many images to sample per slide (default: 256)')
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        help='number of data loading workers (default: 4)')
    parser.add_argument(
        '--dataset',
        type=str,
        default='valid',
        help='dataset to test: one of (train, valid, test)')
    parser.add_argument(
        '--target_key',
        default='ICU',
        type=str,
        help='target key, ICU/Stay/Transfer')
    parser.add_argument('--fold', default=0, type=int, help='fold to use.')
    parser.add_argument('--network', default='resnet34', type=str, help='which model to use, defaut: resnet34')
    args = parser.parse_args()
    main(args)
