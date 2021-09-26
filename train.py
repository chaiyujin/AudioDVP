import os
import time
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

from options.options import Options
from models import resnet_model
from datasets import create_dataset
from utils.visualizer import Visualizer


if __name__ == '__main__':
    opt = Options().parse_args()   # get training options
    dataset = create_dataset(opt)
    model = resnet_model.ResnetModel(opt)
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                   Training                                                   * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    total_iters = 0
    for epoch in range(opt.num_epoch):

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                visualizer.display_current_results(model.get_current_visuals(), total_iters)

            if total_iters % opt.print_freq == 0:    # print training losses
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            iter_data_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.num_epoch, time.time() - epoch_start_time))
    model.save_network()

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                   Evaluate and save reconstruction results                                   * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    model.net.to(model.device)
    model.load_network()

    # evaluate and save result tqdm
    dataset = dataset.dataset
    bsz = model.opt.batch_size
    for i in tqdm(range(0, len(dataset), bsz), desc="Save reconstruction results"):
        batch = []
        for j in range(i, i+bsz):
            if j >= len(dataset):
                j = len(dataset) - 1
            data = dataset[j]
            batch.append(data)
        batch = default_collate(batch)
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].cuda()
        model.set_input(batch)
        model.forward()
        model.save_result()
