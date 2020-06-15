# MIL application on COVID-19
This repository provides training and testing scripts for the MIL application on COVID-19.

## Weakly-supervised patch level classifier

### MIL Input Data

In this instruction the words "tile" and "patch" have the same means.

Input data should be stored in a folder which must has two types of file:

* `"image data files"`: Which are npy files that contain CT slices:
```
image-00001-00001.npy : The 1st CT slice for the 1st case
image-00001-00002.npy : The 2nd CT slice for the 1st case
...
image-00001-00050.npy : The 50th CT slice for the 1st case
...
image-00010-00100.npy : The 100th CT slice for the 10th case
```
* `"pkl files"`: pkl file contain the tile info, label and the corresponding cases.

Each pkl file contain a dictionary which is serialized by module pickle, this dictionary should has the folloing keys.

* `"case"`: list of all cases. Each of case is a list contains all its tiles(patch). Each of tile(Patch) should be the following structure: [(patch, image_name_prefix, case_id, case_label)]. "patch" refers the tile center in a CT slices which is a triplet similar to (z, y, x). "image_name_prefix" is the file name prefix for the image data, image-00001 for example. "case_id" is used to identify the case when in inference. "case_label" is the case label with the following structure: numpy.array([Class A value, Class B value, Class C value, ...])

* `"fold"`: A list with N element. It is "n-fold" which split the data into training and validation group. Each element has two list with the first one contains the index of training data and the second one contains the index of validation data. Here the index refers to the ordering in the key "case".

* `"tile"`: The tile/patch size.
* `"stride"`: The stride of the tile/patch, not used yet.
* `"target_spacing"`: All the cases are resampled to a same pixel spacing, this key contain contain its value.
* `"transpose_xyz"`: We assume that the 3D CT image is the axes oder: (z,y,x). This key contain the info to transpose the axes.
* `"new_shape"`: The new shape of 3D CT images after resample.
* `"target_keys"`: A dictionary map from Class Name to the index of the above "case_label". With this multiple task is possible by specifing Class Name.

### MIL Training
To train a model, use script `covid_train.py`. Run `python covid_train.py -h` to get help regarding input parameters.
Script outputs:
* **convergence.csv**: *.csv* file containing training loss and validation error metrics.
* **checkpoint_best.pth**: file containing the weights of the best model on the validation set. This file can be used with the `covid_test.py` script to run the model on a test set.

### MIL Testing
To run a model on a test set, use script `covid_test.py`. Run `python covid_test.py -h` to get help regarding input parameters.
Script outputs:
* **predictions_{dataset}.csv**: *.csv* file with case name, case target, model prediction and tumor probability entries for each case in the test data.