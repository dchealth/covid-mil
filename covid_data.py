import numpy as np
import random
import torch.utils.data as data
from os import path
import torch
import misc


class COVIDDataset(data.Dataset):
    def __init__(self,
                 plan,
                 data_folder,
                 target_key,
                 transform=None,
                 data_type='train',
                 fold=0,
                 balance_class=False,
                 patch=None):
        self.label_idx = plan['target_keys'][target_key]
        self.data_folder = data_folder

        grid = []
        # Here slide refers to a CT serial which have multiple CT slices
        slideIDX = []
        # Train and validation dataset should be placed in one folder.
        # Test dataset is placed in a different folder compared with train & valid.
        # In a pkl file. all cases are splitted into 5 folds.
        # Each fold has an train dataset with index 0 and an validation dataset with index 1.
        if 'fold' in plan:
            # train and validation dataset
            if data_type == 'train':
                idx_list = plan['fold'][fold][0]
                if balance_class:
                    # copy case to balance the class
                    targets = np.array([
                        plan['case'][i][0][-1][self.label_idx]
                        for i in idx_list
                    ])
                    res = self.blance_class(targets, idx_list)
                    print(f'balance class samples: {len(res)}')
                    idx_list.extend(res)
            elif data_type == 'valid':
                idx_list = plan['fold'][fold][1]
            else:
                idx_list = plan['fold'][fold][0] + plan['fold'][fold][1]
        else:
            # test dataset
            idx_list = range(len(plan['case']))
        use_grids = [plan['case'][i] for i in idx_list]

        tg = [g[0][-1][self.label_idx] for g in use_grids]
        if len(np.unique(tg)) > 2 and target_key == 'Stay':
            print('Transfer stay to binary label: ', np.unique(tg))
            for g in use_grids:
                for i in range(len(g)):
                    g[i][-1][self.label_idx] = g[i][-1][self.label_idx] >= 10

        for i, g in enumerate(use_grids):
            grid.extend(g)
            slideIDX.extend([i] * len(g))

        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = 1
        self.size = np.array(plan['tile'])
        if patch is not None:
            self.size = np.array(patch)
        self.topk_data = []
        self.targets = np.array([g[0][-1][self.label_idx] for g in use_grids])
        # self.targets[self.targets != 0] = 1
        self.slide_names = np.array([g[0][-2] for g in use_grids])

        print(f'Target key: {target_key}')
        print(f'Number of slides: {len(use_grids)}')
        print(f'Number of tiles: {len(grid)}')
        print(f'Tile size: {self.size}')
        print(f'Target size: {len(self.targets)}')
        cls_lbl = np.unique(self.targets)
        cls_vol = [(self.targets == i).sum() for i in cls_lbl]
        print(f'Target label: {cls_lbl}')
        print(f'Target number: {cls_vol}')
        print(f'Slide names size: {len(self.slide_names)}')

    def blance_class(self, targets, indexs):
        """
        blance class with target and its index
        return case index
        """
        indexs = np.array(indexs)
        targets = np.array(targets)
        types = np.unique(targets)
        type_pos = np.array([targets == i for i in types])
        type_vol = np.array([np.sum(p) for p in type_pos])
        type_vol = type_vol * (type_vol.max() / type_vol - 1).astype(np.int)
        extra = [
            np.random.choice(indexs[p], v) for p, v in zip(type_pos, type_vol)
        ]
        res = []
        for e in extra:
            res.extend(e)
        return res

    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs):
        """
        make up train patchs
        """
        self.topk_data = [self.grid[x] for x in idxs]

    def shuffletraindata(self):
        self.topk_data = random.sample(self.topk_data, len(self.topk_data))

    def _read_image(self, z_idx, fname):
        to_read = self.size[0]
        image = []
        start = z_idx - to_read // 2
        end = start + to_read
        for i in range(start, end):
            fpath = path.join(self.data_folder, f'{fname}-{i:0>5d}.npy')
            try:
                image.append(np.load(fpath))
            except:
                # print(fpath)
                ...
        if len(image) < to_read:
            if start < 0:
                image = [image[0]] * (to_read - len(image)) + image
            else:
                image += [image[-1]] * (to_read - len(image))
        return np.array(image)

    def __getitem__(self, index):
        # print('slide: ',self.grid[index])
        # model 1: inference
        # model 2: training
        if self.mode == 1:
            coord, fname, *_, target = self.grid[index]
        elif self.mode == 2:
            coord, fname, *_, target = self.topk_data[index]

        target = target[self.label_idx]
        # if target != 0:
        #     target = 1
        image = self._read_image(coord[0], fname)
        # print(f'image shape: {image.shape}')

        size = self.size
        coord = np.array(coord)
        coord[0] = size[0] // 2
        bbox, pad = misc.get_bbox_pad(image.shape, size, coord)
        # print('size:', size, 'coord: ', coord, 'bbox: ', bbox, 'pad: ',pad)
        tile = image[bbox].copy()
        tile = np.pad(tile, pad, mode='constant', constant_values=tile.min())
        # print(f'tile shape: {tile.shape}')

        # tile = torch.FloatTensor(tile)

        if self.transform is not None:
            tile = self.transform(tile)

        res = tile
        if self.mode == 2:
            res = (tile, target)
        return res

    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.topk_data)
