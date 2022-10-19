from cv2.ximgproc import guidedFilter
from net.losses import StdLoss, TLoss, L_TV, L_color
from utils.imresize import np_imresize
from utils.image_io import *
from skimage.color import rgb2hsv
import torch
from net.vae import VAE
import numpy as np
from net.Net import Net
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class uie(object):

    def __init__(self, image_name, image, opt):
        self.image_name = image_name
        self.image = image
        self.num_iter = opt.num_iter
        self.ambient_net = None
        self.image_net = None
        self.mask_net1 = None
        self.mask_net2 = None
        self.mse_loss = None
        self.learning_rate = opt.learning_rate
        self.parameters = None
        self.output_path = "output/" + opt.datasets + '/'

        self.data_type = torch.cuda.FloatTensor
        self.clip = opt.clip
        self.blur_loss = None
        self.best_result = None
        self.image_net_inputs = None
        self.mask_net_inputs1 = None
        self.mask_net_inputs2 = None
        self.image_out = None
        self.mask_out1 = None
        self.mask_out2 = None
        self.ambient_out = None
        self.total_loss = None
        self._init_all()

    def _init_images(self):
        self.original_image = self.image.copy()
        factor = 1
        image = self.image
        image_size = 600

        while image.shape[1] >= image_size or image.shape[2] >= image_size:
            new_shape_x, new_shape_y = self.image.shape[1] / factor, self.image.shape[2] / factor
            new_shape_x -= (new_shape_x % 32)
            new_shape_y -= (new_shape_y % 32)
            image = np_imresize(self.image, output_shape=(new_shape_x, new_shape_y))
            factor += 1

        self.image = image
        self.image_torch = np_to_torch(self.image).type(torch.cuda.FloatTensor)

    def _init_nets(self):
        image_net = Net(out_channel=3)

        self.image_net = image_net.type(self.data_type)

        mask_net1 = Net(out_channel=3)

        self.mask_net1 = mask_net1.type(self.data_type)

        mask_net2 = Net(out_channel=3)

        self.mask_net2 = mask_net2.type(self.data_type)

    def _init_ambient(self):
        ambient_net = VAE(self.image.shape)
        self.ambient_net = ambient_net.type(torch.cuda.FloatTensor)
        # ambient_net = Net(out_channel=3)
        # self.ambient_net = ambient_net.type(self.data_type)

    def _init_parameters(self):
        parameters = [p for p in self.image_net.parameters()] + \
                     [p for p in self.mask_net1.parameters()] + \
                     [p for p in self.mask_net2.parameters()] + \
                     [p for p in self.ambient_net.parameters()]

        self.parameters = parameters

    def _init_loss(self):
        self.mse_loss = torch.nn.MSELoss().type(self.data_type)
        self.blur_loss = StdLoss().type(self.data_type)
        self.t_loss = TLoss().type(self.data_type)
        self.color_loss = L_color().type(self.data_type)
        self.tv_loss = L_TV().type(self.data_type)

    def _init_inputs(self):
        self.image_net_inputs = np_to_torch(self.image).cuda().type(self.data_type)
        self.mask_net_inputs1 = np_to_torch(self.image).cuda().type(self.data_type)
        self.mask_net_inputs2 = np_to_torch(self.image).cuda().type(self.data_type)
        self.ambient_net_input = np_to_torch(self.image).cuda().type(self.data_type)

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_ambient()
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            optimizer.step()
            if j % 100 == 0:
                self.finalize(steps=j)

    def _optimization_closure(self, step):
        """
        :param step: the number of the iteration
        :return:
        """

        self.image_out = self.image_net(self.image_net_inputs)
        self.ambient_out = self.ambient_net(self.ambient_net_input)
        self.mask_out1 = self.mask_net1(self.mask_net_inputs1)
        self.mask_out2 = self.mask_net2(self.mask_net_inputs2)

        self.mseloss = self.mse_loss(self.mask_out1 * self.image_out + (1 - self.mask_out2) * self.ambient_out,
                                     self.image_torch)

        hsv = np_to_torch(rgb2hsv(torch_to_np(self.image_out).transpose(1, 2, 0)))
        cap_prior = hsv[:, :, :, 2] - hsv[:, :, :, 1]
        self.cap_loss = self.mse_loss(cap_prior, torch.zeros_like(cap_prior))
        vae_loss = self.ambient_net.getLoss()

        self.total_loss = self.mseloss
        self.total_loss += vae_loss
        self.total_loss += 1 * self.cap_loss
        self.total_loss += 0.1 * self.blur_loss(self.ambient_out)
        self.total_loss += 1 * torch.mean(self.color_loss(self.image_out))
        self.total_loss += 1 * self.tv_loss(self.ambient_out)
        self.total_loss += 0.1 * torch.mean(self.t_loss(self.mask_out2))
        self.total_loss.backward(retain_graph=True)

        print('Iteration %05d    Loss %f %f %0.4f%% \n' % (
            step, self.total_loss.item(),
            self.cap_loss,
            self.cap_loss / self.total_loss.item()), '\r', end='')

    def finalize(self, steps=800):

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
            os.mkdir(self.output_path + 'I/')
            os.mkdir(self.output_path + 'Normal/')
            os.mkdir(self.output_path + 'Matting/')
            os.mkdir(self.output_path + 'T1/')
            os.mkdir(self.output_path + 'T2/')
            os.mkdir(self.output_path + 'A/')

        image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
        mask_out_np1 = np.clip(torch_to_np(self.mask_out1), 0, 1)
        mask_out_np2 = np.clip(torch_to_np(self.mask_out2), 0, 1)
        ambient_out_np = np.clip(torch_to_np(self.ambient_out), 0, 1)

        final_a = ambient_out_np
        final_t1 = mask_out_np1
        final_t2 = mask_out_np2

        save_image(self.image_name, image_out_np, self.output_path + 'I/' + str(steps) + '/')

        post = np.clip((self.image - ((1 - final_t2) * final_a)) / final_t1, 0, 1)
        save_image(self.image_name, post, self.output_path + 'Normal/' + str(steps) + '/')

        final_t1 = self.t_matting(final_t1)
        final_t2 = self.t_matting(final_t2)
        post = np.clip((self.image - ((1 - final_t2) * final_a)) / final_t1, 0, 1)
        save_image(self.image_name, post, self.output_path + 'Matting/' + str(steps) + '/')

        save_image(self.image_name, mask_out_np1, self.output_path + 'T1/' + str(steps) + '/')
        save_image(self.image_name, mask_out_np2, self.output_path + 'T2/' + str(steps) + '/')

        save_image(self.image_name, final_a, self.output_path + 'A/' + str(steps) + '/')

    def t_matting(self, mask_out_np):

        refine_t = guidedFilter(self.image.transpose(1, 2, 0).astype(np.float32), mask_out_np.transpose(1, 2, 0).astype(np.float32), 50, 1e-4)
        refine_t = refine_t.transpose(2, 0, 1)
        if self.clip:
            return np.clip(refine_t, 0.1, 1)
        else:
            return np.clip(refine_t, 0, 1)


def uie_opt(opt):
    torch.cuda.set_device(opt.cuda)
    hazy_add = './data/challenging-60/*.png'
    print(hazy_add)
    for item in sorted(glob.glob(hazy_add)):
        print(item)
        name = item.split('/')[-1]
        hazy_img = prepare_image(item)
        dh = uie(name, hazy_img, opt)
        dh.optimize()
        dh.finalize(steps=opt.num_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--datasets', type=str, default="challenging-60")
    parser.add_argument('--clip', type=bool, default=True)
    parser.add_argument('--num_iter', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    options = parser.parse_args()
    uie_opt(options)

