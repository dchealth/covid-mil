from copy import deepcopy
import numpy as np
from batchgenerators.transforms import Compose, RenameTransform, GammaTransform, SpatialTransform
from batchgenerators.transforms import MirrorTransform

default_3D_augmentation_params = {
    "do_elastic": True,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "do_rotation": True,
    "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "random_crop": False,
    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,
    "mirror": True,
    "mirror_axes": (0, 1, 2),
    "p_eldef": 0.2,
    "p_scale": 0.2,
    "p_rot": 0.2,
    # "border_mode_data": "constant",
    "border_mode_data": "nearest",
}

default_2D_augmentation_params = deepcopy(default_3D_augmentation_params)

default_2D_augmentation_params["elastic_deform_alpha"] = (0., 200.)
default_2D_augmentation_params["elastic_deform_sigma"] = (9., 13.)
default_2D_augmentation_params["rotation_x"] = (-180. / 360 * 2. * np.pi,
                                                180. / 360 * 2. * np.pi)


def get_augmentation(patch_size,
                     params=default_3D_augmentation_params,
                     border_val_seg=-1):
    print(f'patch size after augmentation {patch_size}')
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform(
            patch_size,
            patch_center_dist_from_border=None,
            do_elastic_deform=params.get("do_elastic"),
            alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"),
            angle_x=params.get("rotation_x"),
            angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"),
            do_scale=params.get("do_scaling"),
            scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"),
            border_cval_data=0,
            order_data=3,
            border_mode_seg="constant",
            border_cval_seg=border_val_seg,
            order_seg=1,
            random_crop=params.get("random_crop"),
            p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"),
            p_rot_per_sample=params.get("p_rot")))
    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(
                params.get("gamma_range"),
                False,
                True,
                retain_stats=params.get("gamma_retain_stats"),
                p_per_sample=params["p_gamma"]))

    tr_transforms.append(MirrorTransform(params.get("mirror_axes")))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def get_patch_augmentation(*args, **kargs):
    aug_func = get_augmentation(*args, **kargs)

    def do_patch(data, *args, **kargs):
        data = aug_func(data=data[None], *args, **kargs)
        return data['data'][0]
    return do_patch

