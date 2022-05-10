#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.utils import draw_bounding_boxes


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.convert('L').save(filename)
    
def detection(file):
    #show(grid)
    pic1_int = read_image(str(Path(file)))
    grid = make_grid([pic1_int, pic1_int])
    
    #model 1
    batch_int = torch.stack([pic1_int])
    batch = convert_image_dtype(batch_int, dtype=torch.float)

    model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
    model = model.eval()

    outputs = model(batch)
 #   print(outputs)

   
    score_threshold =0.8
    for dog_int, output in zip(batch_int, outputs):
        boxes=output['boxes'][output['scores'] > score_threshold]
       # print(boxes)
    objs_with_boxes = [
        draw_bounding_boxes(dog_int, boxes=output['boxes'][output['scores'] > score_threshold], width=4)
        for dog_int, output in zip(batch_int, outputs)
    ]
#    show(objs_with_boxes)
 #   print(type(objs_with_boxes[0]))
    img= objs_with_boxes[0]
    img = img.detach()
    img = F.to_pil_image(img)
    plt.imshow(img)
    plt.savefig(file)
    return boxes
                          
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file", type=str, help="Put input file")
    parser.add_argument("-box_info", action='store_true',help="Print bounding box coordinate")
    args = parser.parse_args()
    mandatory_args = {'input_file'}

    if not mandatory_args.issubset(set(dir(args))):
        raise Exception(("You're missing essential arguments!"
                         "We need these to run your code."))
        
    file = args.input_file
  #  out = args.output_file
    box = detection(file)
    box = box.tolist()
 #   print(args.box_info)
    if args.box_info:
        a=0
        print("The bounding box is at:", box)
if __name__ == "__main__":
    main()
