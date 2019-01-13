import json
from collections import OrderedDict
import numpy as np
import torch

from PIL import Image
from torchvision import datasets, transforms

def get_dataloaders(means, stds, data_dir=None, batch_size=32):
    # Setup directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(means, stds)]),
                       'default': transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(means, stds)])}
    # Load the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['default']
                      ),
                      'test': datasets.ImageFolder(test_dir, transform=data_transforms['default'])}
    
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {k: torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
                   for k, dataset in image_datasets.items()}
    return dataloaders, image_datasets


def get_image_labels(filename):
    with open(filename, 'r') as f:
        image_to_name = json.load(f)
    return image_to_name

def aspect_resize_dimensions(image, size):
    image_size = np.array(image.size)
    if (image_size[1] < image_size[0]):
        shortest_side = 1
    else:
        shortest_side = 0
    resize_ratio = size / image_size[shortest_side]
    return tuple((image_size * resize_ratio).astype(int))

def center_crop_dimensions(image, size):
    width, height = image.size
    target_width = target_height = size
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2
    return (left, top, right, bottom)

def process_image(image_path, means, stds):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    image = image.resize(aspect_resize_dimensions(image, 256))
    image = image.crop(center_crop_dimensions(image, 224))
    np_image = np.array(image) / 255
    np_image = (np_image - means ) / stds
    np_image = np.transpose(np_image, (2, 0, 1))
    return np_image