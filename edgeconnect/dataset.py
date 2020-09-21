import os
import cv2
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask
import src.region_fill as rf

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()

        self.augment = augment
        self.training = training
        self.flo = config.FLO
        self.norm = config.NORM
        self.data = self.load_flist(flist, self.flo)
        self.edge_data = self.load_flist(edge_flist, 0)
        self.mask_data = self.load_flist(mask_flist, 0)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS



        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size
        factor = 1.
        if self.flo == 0:

            # load image
            img = imread(self.data[index])

            # gray to rgb
            if len(img.shape) < 3:
                img = gray2rgb(img)

            # resize/crop if needed
            if size != 0:
                img = self.resize(img, size[0], size[1])

            # create grayscale image
            img_gray = rgb2gray(img)

            # load mask
            mask = self.load_mask(img, index)

            edge = self.load_edge(img_gray, index, mask)

            img_filled = img

        else:

            img = self.readFlow(self.data[index])

            # resize/crop if needed
            if size != 0:
                img = self.flow_tf(img, [size[0], size[1]])

            img_gray = (img[:, :, 0] ** 2 + img[:, :, 1] ** 2) ** 0.5

            if self.norm == 1:
                # normalization
                # factor = (np.abs(img[:, :, 0]).max() ** 2 + np.abs(img[:, :, 1]).max() ** 2) ** 0.5
                factor = img_gray.max()
                img /= factor

            # load mask
            mask = self.load_mask(img, index)

            edge = self.load_edge(img_gray, index, mask)
            img_gray = img_gray / img_gray.max()

            img_filled = np.zeros(img.shape)
            img_filled[:, :, 0] = rf.regionfill(img[:, :, 0], mask)
            img_filled[:, :, 1] = rf.regionfill(img[:, :, 1], mask)


        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...].copy()
            img_filled = img_filled[:, ::-1, ...].copy()
            img_gray = img_gray[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        return self.to_tensor(img), self.to_tensor(img_filled), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask), factor

    def load_edge(self, img, index, mask):
        sigma = self.sigma

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)
            return canny(img, sigma=sigma, mask=mask).astype(np.float)

        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = imread(self.edge_data[index])
            edge = self.resize(edge, imgh, imgw)

            # non-max suppression
            if self.nms == 1:
                edge = edge * canny(img, sigma=sigma, mask=mask)

            return edge

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random

        if mask_type == 6:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        if (len(img.shape) == 3 and img.shape[2] == 2):
            return F.to_tensor(img).float()
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist, flo=0):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if flo == 0:
            if isinstance(flist, str):
                if os.path.isdir(flist):
                    flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                    flist.sort()
                    return flist

                if os.path.isfile(flist):
                    try:
                        return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                    except:
                        return [flist]
        else:
            if isinstance(flist, str):
                if os.path.isdir(flist):
                    flist = list(glob.glob(flist + '/*.flo'))
                    flist.sort()
                    return flist

                if os.path.isfile(flist):
                    try:
                        return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                    except:
                        return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def readFlow(self, fn):
        with open(fn, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
                # Reshape data into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                return np.resize(data, (int(h), int(w), 2))

    def flow_to_image(self, flow):

        UNKNOWN_FLOW_THRESH = 1e7

        u = flow[:, :, 0]
        v = flow[:, :, 1]

        maxu = -999.
        maxv = -999.
        minu = 999.
        minv = 999.

        idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
        u[idxUnknow] = 0
        v[idxUnknow] = 0

        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))

        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))

        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(-1, np.max(rad))

        # print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)

        img = self.compute_color(u, v)

        idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
        img[idx] = 0

        return np.uint8(img)


    def compute_color(self, u, v):
        """
        compute optical flow color map
        :param u: optical flow horizontal map
        :param v: optical flow vertical map
        :return: optical flow in color code
        """
        [h, w] = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0

        colorwheel = self.make_color_wheel()
        ncols = np.size(colorwheel, 0)

        rad = np.sqrt(u**2+v**2)

        a = np.arctan2(-v, -u) / np.pi

        fk = (a+1) / 2 * (ncols - 1) + 1

        k0 = np.floor(fk).astype(int)

        k1 = k0 + 1
        k1[k1 == ncols+1] = 1
        f = fk - k0

        for i in range(0, np.size(colorwheel,1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0-1] / 255
            col1 = tmp[k1-1] / 255
            col = (1-f) * col0 + f * col1

            idx = rad <= 1
            col[idx] = 1-rad[idx]*(1-col[idx])
            notidx = np.logical_not(idx)

            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

        return img


    def make_color_wheel(self):
        """
        Generate color wheel according Middlebury color code
        :return: Color wheel
        """
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR

        colorwheel = np.zeros([ncols, 3])

        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
        col += RY

        # YG
        colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
        colorwheel[col:col+YG, 1] = 255
        col += YG

        # GC
        colorwheel[col:col+GC, 1] = 255
        colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
        col += GC

        # CB
        colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
        colorwheel[col:col+CB, 2] = 255
        col += CB

        # BM
        colorwheel[col:col+BM, 2] = 255
        colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
        col += + BM

        # MR
        colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
        colorwheel[col:col+MR, 0] = 255

        return colorwheel

    def flow_tf(self, flow, size):
        flow_shape = flow.shape
        flow_resized = cv2.resize(flow, (size[1], size[0]))
        flow_resized[:, :, 0] *= (float(size[1]) / float(flow_shape[1]))
        flow_resized[:, :, 1] *= (float(size[0]) / float(flow_shape[0]))

        return flow_resized
