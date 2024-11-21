import os
import pickle

from random import shuffle

# scan the target dataset
def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out

# return the data list
def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


# return data
def get_data(opt):
    if opt.data_source == "folder":
        image, label = get_data_by_folder(opt)
    elif opt.data_source == "list":
        image, label = get_data_by_namelist(opt)
    else:
        raise ValueError("data should be loaded from folder or list")
    
    return image, label

# load data by folder origanization
def get_data_by_folder(opt):
    
    temp = 'train' if opt.data_label == 'train' else 'val'
    
    real_list = get_list(os.path.join(opt.dataset_path,temp), must_contain='0_real')
    fake_list = get_list(os.path.join(opt.dataset_path,temp), must_contain='1_fake')

    # setting the labels for the dataset
    labels_dict = dict()
    for i in real_list:
        labels_dict[i] = 0
    for i in fake_list:
        labels_dict[i] = 1

    total_list = real_list + fake_list
    shuffle(total_list)

    return total_list, labels_dict

# load data by list file
def get_data_by_namelist(opt):

    labels_dict = dict()
    total_list = list()
    with open(opt.datalist, 'r') as f:
        contents = f.readlines()
        for content in contents:
            content_split = content.split(",")
            image, label = content_split[0].strip(), int(content_split[1].strip())
            total_list.append(image)
            labels_dict[image] = label

    return total_list, labels_dict