# %%
import numpy as np

# %%
def get_bbox_pad(data_size, patch_size, center):
    """
    Calculate the bbox and needed pad according to patch center.
    """
    # bbox_low = center - np.array(patch_size) // 2
    # bbox_high = center + patch_size
    # pad_low = np.abs(np.minimum(bbox_low - 0, 0))
    # pad_high = np.abs(np.maximum(bbox_high - data_size, 0))
    # bbox_low = np.maximum(bbox_low, 0)
    # bbox_high = np.minimum(bbox_high, data_size)
    # bbox = tuple(slice(*b) for b in np.stack((bbox_low, bbox_high), 1))
    # pad = np.stack((pad_low, pad_high), 1)
    coord = np.array(center)
    size = np.array(data_size)
    patch = np.array(patch_size)

    req_stt = coord - patch//2
    req_end = req_stt + patch
    src_stt = np.maximum(0, req_stt)
    src_end = np.minimum(size, req_end)

    # print(f'req: {[(s,e) for s,e in zip(req_stt,req_end)]}')
    # print(f'src: {[(s,e) for s,e in zip(src_stt,src_end)]}')

    # print(f'tile shape: {tile.shape}')

    pad_low = np.abs(np.minimum(req_stt, 0))
    pad_high = np.abs(np.maximum(req_end - size, 0))
    pad = np.stack((pad_low, pad_high), 1)
    bbox = tuple(slice(*b) for b in np.stack((src_stt, src_end), 1))

    return bbox, pad
