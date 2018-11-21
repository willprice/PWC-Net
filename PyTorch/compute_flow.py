import sys
import cv2
import torch
import numpy as np
from math import ceil
from scipy.ndimage import imread
import models
import argparse

parser = argparse.ArgumentParser(description="Compute flow using PWC net")
parser.add_argument('input_0', type=str, help="Path to first image")
parser.add_argument('input_1', type=str, help="Path to second image")
parser.add_argument('output_u', type=str, help="Path to flow u image")
parser.add_argument('output_v', type=str, help="Path to flow v image")
parser.add_argument('--bound', type=float, default=25, help="Bound to clip flow at")
args = parser.parse_args()


def flow_to_image(flow_frame, min_, max_):
    clipped = np.clip(flow_frame, min_, max_)
    range_ = max_ - min_
    clipped -= min_
    clipped /= range_
    clipped *= 255.0
    assert clipped.min() >= -0.01
    assert clipped.max() <= 255.01
    img = clipped.astype(np.uint8)
    return img


def save_flow(filename, flow_image_float):
    img = flow_to_image(flow_image_float, -args.bound, args.bound)

    cv2.imwrite(filename, img)


im1_fn = args.input_0
im2_fn = args.input_1
flow_u_fn = args.output_u
flow_v_fn = args.output_v

pwc_model_fn = './pwc_net.pth.tar';

im_all = [imread(img) for img in [im1_fn, im2_fn]]
im_all = [im[:, :, :3] for im in im_all]

# rescale the image size to be multiples of 64
divisor = 64.
H = im_all[0].shape[0]
W = im_all[0].shape[1]

H_ = int(ceil(H/divisor) * divisor)
W_ = int(ceil(W/divisor) * divisor)
for i in range(len(im_all)):
    im_all[i] = cv2.resize(im_all[i], (W_, H_))

for _i, _inputs in enumerate(im_all):
    im_all[_i] = im_all[_i][:, :, ::-1]
    im_all[_i] = 1.0 * im_all[_i]/255.0

    im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
    im_all[_i] = torch.from_numpy(im_all[_i])
    im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])
    im_all[_i] = im_all[_i].float()

im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)

net = models.pwc_dc_net(pwc_model_fn)
net = net.cuda()
net.eval()

flo = net(im_all)
flo = flo[0] * 20.0
flo = flo.cpu().data.numpy()

# scale the flow back to the input size
flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) #
u_ = cv2.resize(flo[:,:,0],(W,H))
v_ = cv2.resize(flo[:,:,1],(W,H))
u_ *= W/ float(W_)
v_ *= H/ float(H_)

save_flow(flow_u_fn, u_)
save_flow(flow_v_fn, v_)
