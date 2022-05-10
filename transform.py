from torchvision import transforms
from options.test_options import TestOptions
from models import create_model
import options.util as util
import numpy as np
import sys, os
import cv2


def main():
    sys.stdout = open(os.devnull, 'w')
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.model = 'test'
    opt.no_dropout = True
    opt.preprocess = None
    model = create_model(opt)
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    path = 'not important'

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])



    images = cv2.imread(opt.img)
    images = np.array(images)

    A = preprocess(images).unsqueeze_(0)

    data = {'A': A, 'A_paths': path}

    model.set_input(data)  # unpack data from data loader
    model.test()  # run inference
    visuals = model.get_current_visuals()  # get image results
    for label, im_data in visuals.items():
        fake = util.tensor2im(im_data)

        util.save_image(fake, "result/" + opt.name + ".jpg")



if __name__ == '__main__':
    main()
