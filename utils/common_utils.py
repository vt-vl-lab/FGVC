from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import cv2
import copy
import numpy as np
import sys
import os
import time
from PIL import Image
import scipy.ndimage


def combine(img1, img2, slope=0.55, band_width=0.015, offset=0):

    imgH, imgW, _ = img1.shape
    band_width = int(band_width * imgH)

    if img1.shape != img2.shape:
        # img1 = cv2.resize(img1, (imgW, imgH))
        raise NameError('Shape does not match')

    center_point = (int(imgH / 2), int(imgW / 2 + offset))

    b = (center_point[1] - 1) - slope * (center_point[0] - 1)
    comp_img = np.zeros(img2.shape, dtype=np.float32)

    for x in range(imgH):
        for y in range(imgW):
            if y < (slope * x + b):
                comp_img[x, y, :] = img1[x, y, :]
            elif y > (slope * x + b):
                comp_img[x, y, :] = img2[x, y, :]

    start_point = (int(b - 0.5 * band_width), 0)
    end_point = (int(slope * (imgW - 1) + b - 0.5 * band_width), imgW - 1)

    color = (1, 1, 1)
    comp_img = cv2.line(comp_img, start_point, end_point, color, band_width, lineType=cv2.LINE_AA)

    return comp_img


def save_video(in_dir, out_dir, optimize=False):

    _, ext = os.path.splitext(sorted(os.listdir(in_dir))[0])
    dir = '"' + os.path.join(in_dir, '*' + ext) + '"'

    if optimize:
        os.system('ffmpeg -y -pattern_type glob -f image2 -i {} -pix_fmt yuv420p -preset veryslow -crf 27 {}'.format(dir, out_dir))
    else:
        os.system('ffmpeg -y -pattern_type glob -f image2 -i {} -pix_fmt yuv420p {}'.format(dir, out_dir))

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def bboxes_mask(imgH, imgW, type='ori'):
    mask = np.zeros((imgH, imgW), dtype=np.float32)
    factor = 1920 * 2 // imgW

    for indFrameH in range(int(imgH / (256 * 2 // factor))):
        for indFrameW in range(int(imgW / (384 * 2 // factor))):
            mask[indFrameH * (256 * 2 // factor) + (128 * 2 // factor) - (64 * 2 // factor) :
                 indFrameH * (256 * 2 // factor) + (128 * 2 // factor) + (64 * 2 // factor),
                 indFrameW * (384 * 2 // factor) + (192 * 2 // factor) - (64 * 2 // factor) :
                 indFrameW * (384 * 2 // factor) + (192 * 2 // factor) + (64 * 2 // factor)] = 1

    if type == 'ori':
        return mask
    elif type == 'flow':
        # Dilate 25 pixel so that all known pixel is trustworthy
        return scipy.ndimage.binary_dilation(mask, iterations=15)

def bboxes_mask_large(imgH, imgW, type='ori'):
    mask = np.zeros((imgH, imgW), dtype=np.float32)
    # mask[50 : 450, 280: 680] = 1
    mask[150 : 350, 350: 650] = 1

    if type == 'ori':
        return mask
    elif type == 'flow':
        # Dilate 35 pixel so that all known pixel is trustworthy
        return scipy.ndimage.binary_dilation(mask, iterations=35)

def gradient_mask(mask):

    gradient_mask = np.logical_or.reduce((mask,
        np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool)), axis=0),
        np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool)), axis=1)))

    return gradient_mask


def flow_edge(flow, mask=None):
    # mask: 1 indicates the missing region
    if not isinstance(mask, np.ndarray):
        mask = None
    else:
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = (1 - mask).astype(np.bool)

    flow_mag = (flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2) ** 0.5
    flow_mag = flow_mag / flow_mag.max()

    edge_canny_flow = canny_flow(flow_mag, flow, mask=mask)
    edge_canny = canny(flow_mag, sigma=2, mask=mask)

    if edge_canny_flow.sum() > edge_canny.sum():
        return edge_canny_flow
    else:
        return edge_canny


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def sigmoid_(x, thres):
    return 1. / (1 + np.exp(-x + thres))


# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()


def softmax(x, axis=None, mask_=None):

    if mask_ is None:
        mask_ = np.ones(x.shape)
    x = (x - x.max(axis=axis, keepdims=True))
    y = np.multiply(np.exp(x), mask_)
    return y / y.sum(axis=axis, keepdims=True)


# Bypass cv2's SHRT_MAX limitation
def interp(img, x, y):

    x = x.astype(np.float32).reshape(1, -1)
    y = y.astype(np.float32).reshape(1, -1)

    assert(x.shape == y.shape)

    numPix = x.shape[1]
    len_padding = (numPix // 1024 + 1) * 1024 - numPix
    padding = np.zeros((1, len_padding)).astype(np.float32)

    map_x = np.concatenate((x, padding), axis=1).reshape(1024, numPix // 1024 + 1)
    map_y = np.concatenate((y, padding), axis=1).reshape(1024, numPix // 1024 + 1)

    # Note that cv2 takes the input in opposite order, i.e. cv2.remap(img, x, y)
    mapped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    if len(img.shape) == 2:
        mapped_img = mapped_img.reshape(-1)[:numPix]
    else:
        mapped_img = mapped_img.reshape(-1, img.shape[2])[:numPix, :]

    return mapped_img


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)


def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()


# Backward flow propagating and forward flow propagating consistency check
def BFconsistCheck(flowB_neighbor, flowF_vertical, flowF_horizont,
                   holepixPos, consistencyThres):

    flowBF_neighbor = copy.deepcopy(flowB_neighbor)

    # After the backward and forward propagation, the pixel should go back
    #  to the original location.
    flowBF_neighbor[:, 0] += interp(flowF_vertical,
                                    flowB_neighbor[:, 1],
                                    flowB_neighbor[:, 0])
    flowBF_neighbor[:, 1] += interp(flowF_horizont,
                                    flowB_neighbor[:, 1],
                                    flowB_neighbor[:, 0])
    flowBF_neighbor[:, 2] += 1

    # Check photometric consistency
    BFdiff = ((flowBF_neighbor - holepixPos)[:, 0] ** 2
            + (flowBF_neighbor - holepixPos)[:, 1] ** 2) ** 0.5
    IsConsist = BFdiff < consistencyThres

    return IsConsist, BFdiff


# Forward flow propagating and backward flow propagating consistency check
def FBconsistCheck(flowF_neighbor, flowB_vertical, flowB_horizont,
                   holepixPos, consistencyThres):

    flowFB_neighbor = copy.deepcopy(flowF_neighbor)

    # After the forward and backward propagation, the pixel should go back
    #  to the original location.
    flowFB_neighbor[:, 0] += interp(flowB_vertical,
                                    flowF_neighbor[:, 1],
                                    flowF_neighbor[:, 0])
    flowFB_neighbor[:, 1] += interp(flowB_horizont,
                                    flowF_neighbor[:, 1],
                                    flowF_neighbor[:, 0])
    flowFB_neighbor[:, 2] -= 1

    # Check photometric consistency
    FBdiff = ((flowFB_neighbor - holepixPos)[:, 0] ** 2
            + (flowFB_neighbor - holepixPos)[:, 1] ** 2) ** 0.5
    IsConsist = FBdiff < consistencyThres

    return IsConsist, FBdiff


def consistCheck(flowF, flowB):

    # |--------------------|  |--------------------|
    # |       y            |  |       v            |
    # |   x   *            |  |   u   *            |
    # |                    |  |                    |
    # |--------------------|  |--------------------|

    # sub: numPix * [y x t]

    imgH, imgW, _ = flowF.shape

    (fy, fx) = np.mgrid[0 : imgH, 0 : imgW].astype(np.float32)
    fxx = fx + flowB[:, :, 0]  # horizontal
    fyy = fy + flowB[:, :, 1]  # vertical

    u = (fxx + cv2.remap(flowF[:, :, 0], fxx, fyy, cv2.INTER_LINEAR) - fx)
    v = (fyy + cv2.remap(flowF[:, :, 1], fxx, fyy, cv2.INTER_LINEAR) - fy)
    BFdiff = (u ** 2 + v ** 2) ** 0.5

    return BFdiff, np.stack((u, v), axis=2)


def get_KeySourceFrame_flowNN(sub,
                              indFrame,
                              mask,
                              videoNonLocalFlowB,
                              videoNonLocalFlowF,
                              video,
                              consistencyThres):

    imgH, imgW, _, _, nFrame = videoNonLocalFlowF.shape
    KeySourceFrame = [0, nFrame // 2, nFrame - 1]

    # Bool indicator of missing pixels at frame t
    holepixPosInd = (sub[:, 2] == indFrame)

    # Hole pixel location at frame t, i.e. [x, y, t]
    holepixPos = sub[holepixPosInd, :]

    HaveKeySourceFrameFlowNN = np.zeros((imgH, imgW, 3))
    imgKeySourceFrameFlowNN = np.zeros((imgH, imgW, 3, 3))

    for KeySourceFrameIdx in range(3):

        # flowF_neighbor
        flowF_neighbor = copy.deepcopy(holepixPos)
        flowF_neighbor = flowF_neighbor.astype(np.float32)
        flowF_vertical = videoNonLocalFlowF[:, :, 1, KeySourceFrameIdx, indFrame]
        flowF_horizont = videoNonLocalFlowF[:, :, 0, KeySourceFrameIdx, indFrame]
        flowB_vertical = videoNonLocalFlowB[:, :, 1, KeySourceFrameIdx, indFrame]
        flowB_horizont = videoNonLocalFlowB[:, :, 0, KeySourceFrameIdx, indFrame]

        flowF_neighbor[:, 0] += flowF_vertical[holepixPos[:, 0], holepixPos[:, 1]]
        flowF_neighbor[:, 1] += flowF_horizont[holepixPos[:, 0], holepixPos[:, 1]]
        flowF_neighbor[:, 2] = KeySourceFrame[KeySourceFrameIdx]

        # Round the forward flow neighbor location
        flow_neighbor_int = np.round(copy.deepcopy(flowF_neighbor)).astype(np.int32)

        # Check the forawrd/backward consistency
        IsConsist, _ = FBconsistCheck(flowF_neighbor, flowB_vertical,
                                    flowB_horizont, holepixPos, consistencyThres)

        # Check out-of-boundary
        ValidPos = np.logical_and(
            np.logical_and(flow_neighbor_int[:, 0] >= 0,
                           flow_neighbor_int[:, 0] < imgH),
            np.logical_and(flow_neighbor_int[:, 1] >= 0,
                           flow_neighbor_int[:, 1] < imgW))

        holepixPos_ = copy.deepcopy(holepixPos)[ValidPos, :]
        flow_neighbor_int = flow_neighbor_int[ValidPos, :]
        flowF_neighbor = flowF_neighbor[ValidPos, :]
        IsConsist = IsConsist[ValidPos]

        KnownInd = mask[flow_neighbor_int[:, 0],
                        flow_neighbor_int[:, 1],
                        KeySourceFrame[KeySourceFrameIdx]] == 0

        KnownInd = np.logical_and(KnownInd, IsConsist)

        imgKeySourceFrameFlowNN[:, :, :, KeySourceFrameIdx] = \
            copy.deepcopy(video[:, :, :, indFrame])

        imgKeySourceFrameFlowNN[holepixPos_[KnownInd, 0],
                                holepixPos_[KnownInd, 1],
                             :, KeySourceFrameIdx] = \
                         interp(video[:, :, :, KeySourceFrame[KeySourceFrameIdx]],
                                flowF_neighbor[KnownInd, 1].reshape(-1),
                                flowF_neighbor[KnownInd, 0].reshape(-1))

        HaveKeySourceFrameFlowNN[holepixPos_[KnownInd, 0],
                                 holepixPos_[KnownInd, 1],
                                 KeySourceFrameIdx] = 1

    return HaveKeySourceFrameFlowNN, imgKeySourceFrameFlowNN
#
def get_KeySourceFrame_flowNN_gradient(sub,
                                      indFrame,
                                      mask,
                                      videoNonLocalFlowB,
                                      videoNonLocalFlowF,
                                      gradient_x,
                                      gradient_y,
                                      consistencyThres):

    imgH, imgW, _, _, nFrame = videoNonLocalFlowF.shape
    KeySourceFrame = [0, nFrame // 2, nFrame - 1]

    # Bool indicator of missing pixels at frame t
    holepixPosInd = (sub[:, 2] == indFrame)

    # Hole pixel location at frame t, i.e. [x, y, t]
    holepixPos = sub[holepixPosInd, :]

    HaveKeySourceFrameFlowNN = np.zeros((imgH, imgW, 3))
    gradient_x_KeySourceFrameFlowNN = np.zeros((imgH, imgW, 3, 3))
    gradient_y_KeySourceFrameFlowNN = np.zeros((imgH, imgW, 3, 3))

    for KeySourceFrameIdx in range(3):

        # flowF_neighbor
        flowF_neighbor = copy.deepcopy(holepixPos)
        flowF_neighbor = flowF_neighbor.astype(np.float32)

        flowF_vertical = videoNonLocalFlowF[:, :, 1, KeySourceFrameIdx, indFrame]
        flowF_horizont = videoNonLocalFlowF[:, :, 0, KeySourceFrameIdx, indFrame]
        flowB_vertical = videoNonLocalFlowB[:, :, 1, KeySourceFrameIdx, indFrame]
        flowB_horizont = videoNonLocalFlowB[:, :, 0, KeySourceFrameIdx, indFrame]

        flowF_neighbor[:, 0] += flowF_vertical[holepixPos[:, 0], holepixPos[:, 1]]
        flowF_neighbor[:, 1] += flowF_horizont[holepixPos[:, 0], holepixPos[:, 1]]
        flowF_neighbor[:, 2] = KeySourceFrame[KeySourceFrameIdx]

        # Round the forward flow neighbor location
        flow_neighbor_int = np.round(copy.deepcopy(flowF_neighbor)).astype(np.int32)

        # Check the forawrd/backward consistency
        IsConsist, _ = FBconsistCheck(flowF_neighbor, flowB_vertical,
                                    flowB_horizont, holepixPos, consistencyThres)

        # Check out-of-boundary
        ValidPos = np.logical_and(
            np.logical_and(flow_neighbor_int[:, 0] >= 0,
                           flow_neighbor_int[:, 0] < imgH - 1),
            np.logical_and(flow_neighbor_int[:, 1] >= 0,
                           flow_neighbor_int[:, 1] < imgW - 1))

        holepixPos_ = copy.deepcopy(holepixPos)[ValidPos, :]
        flow_neighbor_int = flow_neighbor_int[ValidPos, :]
        flowF_neighbor = flowF_neighbor[ValidPos, :]
        IsConsist = IsConsist[ValidPos]

        KnownInd = mask[flow_neighbor_int[:, 0],
                        flow_neighbor_int[:, 1],
                        KeySourceFrame[KeySourceFrameIdx]] == 0

        KnownInd = np.logical_and(KnownInd, IsConsist)

        gradient_x_KeySourceFrameFlowNN[:, :, :, KeySourceFrameIdx] = \
            copy.deepcopy(gradient_x[:, :, :, indFrame])
        gradient_y_KeySourceFrameFlowNN[:, :, :, KeySourceFrameIdx] = \
            copy.deepcopy(gradient_y[:, :, :, indFrame])

        gradient_x_KeySourceFrameFlowNN[holepixPos_[KnownInd, 0],
                                        holepixPos_[KnownInd, 1],
                                     :, KeySourceFrameIdx] = \
                                 interp(gradient_x[:, :, :, KeySourceFrame[KeySourceFrameIdx]],
                                        flowF_neighbor[KnownInd, 1].reshape(-1),
                                        flowF_neighbor[KnownInd, 0].reshape(-1))

        gradient_y_KeySourceFrameFlowNN[holepixPos_[KnownInd, 0],
                                        holepixPos_[KnownInd, 1],
                                     :, KeySourceFrameIdx] = \
                                 interp(gradient_y[:, :, :, KeySourceFrame[KeySourceFrameIdx]],
                                        flowF_neighbor[KnownInd, 1].reshape(-1),
                                        flowF_neighbor[KnownInd, 0].reshape(-1))

        HaveKeySourceFrameFlowNN[holepixPos_[KnownInd, 0],
                                 holepixPos_[KnownInd, 1],
                                 KeySourceFrameIdx] = 1

    return HaveKeySourceFrameFlowNN, gradient_x_KeySourceFrameFlowNN, gradient_y_KeySourceFrameFlowNN

class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return self.max_val - 10 * torch.log(mse) / self.base10
# Get surrounding integer postiion
def IntPos(CurPos):

    x_floor = np.expand_dims(np.floor(CurPos[:, 0]).astype(np.int32), 1)
    x_ceil = np.expand_dims(np.ceil(CurPos[:, 0]).astype(np.int32), 1)
    y_floor = np.expand_dims(np.floor(CurPos[:, 1]).astype(np.int32), 1)
    y_ceil = np.expand_dims(np.ceil(CurPos[:, 1]).astype(np.int32), 1)
    Fm = np.expand_dims(np.floor(CurPos[:, 2]).astype(np.int32), 1)

    Pos_tl = np.concatenate((x_floor, y_floor, Fm), 1)
    Pos_tr = np.concatenate((x_ceil, y_floor, Fm), 1)
    Pos_bl = np.concatenate((x_floor, y_ceil, Fm), 1)
    Pos_br = np.concatenate((x_ceil, y_ceil, Fm), 1)

    return Pos_tl, Pos_tr, Pos_bl, Pos_br
