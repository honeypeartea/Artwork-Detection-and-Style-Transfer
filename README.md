# Artwork-Detection-and-Style-Transfer

**[PyTorch Colab](https://colab.research.google.com/drive/137a1Q-XYZZRvNoTa9nQ6oDJGc8p7IdBf#scrollTo=-9VDfC-ie19P)**

## Intrdouction
Our Artwork Detection and Style Transfer project focus on museum artwork image detection and style transfer. It mainly contains three parts, the image classification, object detection, and style transfer.

We want to achieve the functions that find out what kind of artwork is on the image, what detail we can found on the image, and transfer the artwork image into other artwork styles. We believe this is very applicable for museum visitors to get to know the artworks more efficiently.

## Image Classification
Under Images_Classification folders
ML_label_create.ipynb is a file that is selected for label when running on google drive.
ML-create-image.ipynb is run locally How to filter image files for label-create files
ML_fp_v3.ipynb is the main file of train

You can visualize your result from our colab notebook. The data for image classification can be downloaded from ./datas

## Object Detection
The object detection will give the image with the highlighted box as output which will save as the same path as the input images. The following is an example of the input and the output. 

### Usage
Python object_detection.py -input_file your_image_file -box_info

The -input_file ask user to provide the images path.
The -box_info command can provide you with the coordinates of bounding box in images.

Also user can use evaluation.ipynb to evaluate their result. 

## Style Transfer
we train the model from photo to drawing, drawing to photo, painting to drawing, drawing to painting. We apply adversarial losses to mapping functions. We developing our own script to generate translated images. This parts refer to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. You can also download prertrained model by download.sh in each folder.

### Usage
pip install -r requirements.txt
Python transform.py --img your_image_file --name model_name

--img is your input image path
--name is your model path which under checkpoints
