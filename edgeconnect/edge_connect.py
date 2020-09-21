import os
import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import EdgeModel, InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy
from tensorboardX import SummaryWriter


class EdgeConnect():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'

        self.debug = False
        self.model_name = model_name
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

        self.writer = SummaryWriter('/home/gaochen/Project/edge-connect/checkpoints/flow_NORM_FILL_MASK_RES/logs')

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.edge_model.load()
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 1:
            self.edge_model.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()

        else:
            self.edge_model.save()
            self.inpaint_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )
        # train_loader = DataLoader(
        #     dataset=self.train_dataset,
        #     batch_size=1,
        # )
        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.edge_model.train()
                self.inpaint_model.train()

                images, images_filled, images_gray, edges, masks, factor = self.cuda(*items)

                # edge model
                if model == 1:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                    # metrics
                    precision, recall = self.edgeacc(edges * masks, outputs * masks)
                    logs.append(('precision', precision.item()))
                    logs.append(('recall', recall.item()))

                    # backward
                    self.edge_model.backward(gen_loss, dis_loss)
                    iteration = self.edge_model.iteration

                # inpaint model
                elif model == 2:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, images_filled, edges, masks)
                    # import ipdb; ipdb.set_trace()
                    # cv2.imwrite('/home/gaochen/test.png', self.flow2img(images * factor, 10)[0])
                    # cv2.imwrite('/home/gaochen/images_filled.png', self.flow2img(images_filled * factor, 10)[0])
                    # cv2.imwrite('/home/gaochen/outputs.png', self.flow2img(outputs.detach() * factor, 10)[0])
                    # cv2.imwrite('/home/gaochen/edges.png', edges.cpu().numpy()[0,0,:,:] * 255)

                    if self.config.NORM == 1:
                        outputs = outputs * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                        images = images * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                        images_filled = images_filled * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)

                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    if self.config.FLO == 0:
                        psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    elif self.config.FLO == 1:
                        psnr = self.psnr(torch.from_numpy(self.flow2img(images, 10)), torch.from_numpy(self.flow2img(outputs_merged.detach(), 10)))
                    else:
                        assert(0)
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(torch.abs(images))).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration

                # inpaint with edge model
                elif model == 3:
                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs = self.edge_model(images_gray, edges, masks)
                        outputs = outputs * masks + edges * (1 - masks)
                    else:
                        outputs = edges

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, images_filled, outputs.detach(), masks)

                    if self.config.NORM == 1:
                        outputs = outputs * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                        images = images * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                        images_filled = images_filled * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)

                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    if self.config.FLO == 0:
                        psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    elif self.config.FLO == 1:
                        psnr = self.psnr(torch.from_numpy(self.flow2img(images)), torch.from_numpy(self.flow2img(outputs_merged)))
                    else:
                        assert(0)

                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(torch.abs(images))).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                # joint model
                else:
                    # train
                    e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                    e_outputs = e_outputs * masks + edges * (1 - masks)
                    i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, images_filled, e_outputs, masks)

                    if self.config.NORM == 1:
                        i_outputs = i_outputs * factor.reshape(-1, 1, 1, 1).type(i_outputs.dtype)
                        images = images * factor.reshape(-1, 1, 1, 1).type(i_outputs.dtype)
                        images_filled = images_filled * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)

                    outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                    # metrics
                    if self.config.FLO == 0:
                        psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    elif self.config.FLO == 1:
                        psnr = self.psnr(torch.from_numpy(self.flow2img(images)), torch.from_numpy(self.flow2img(outputs_merged)))
                    else:
                        assert(0)

                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(torch.abs(images))).float()
                    precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                    e_logs.append(('pre', precision.item()))
                    e_logs.append(('rec', recall.item()))
                    i_logs.append(('psnr', psnr.item()))
                    i_logs.append(('mae', mae.item()))
                    logs = e_logs + i_logs

                    # backward
                    self.inpaint_model.backward(i_gen_loss, i_dis_loss)
                    self.edge_model.backward(e_gen_loss, e_dis_loss)
                    iteration = self.inpaint_model.iteration


                if iteration >= max_iteration:
                    keep_training = False
                    break

                for idx in range(len(logs)):
                    self.writer.add_scalar(logs[idx][0], logs[idx][1], iteration)


                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.edge_model.eval()
        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            # images, images_gray, edges, masks = self.cuda(*items)
            images, images_filled, images_gray, edges, masks, factor = self.cuda(*items)

            # edge model
            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))


            # inpaint model
            elif model == 2:
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, images_filled, edges, masks)

                if self.config.NORM == 1:
                    outputs = outputs * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                    images = images * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                    images_filled = images_filled * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)

                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                if self.config.FLO == 0:
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                elif self.config.FLO == 1:
                    psnr = self.psnr(torch.from_numpy(self.flow2img(images)), torch.from_numpy(self.flow2img(outputs_merged)))
                else:
                    assert(0)

                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # inpaint with edge model
            elif model == 3:
                # eval
                outputs = self.edge_model(images_gray, edges, masks)
                outputs = outputs * masks + edges * (1 - masks)

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, images_filled, outputs.detach(), masks)

                if self.config.NORM == 1:
                    outputs = outputs * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                    images = images * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                    images_filled = images_filled * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)

                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                if self.config.FLO == 0:
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                elif self.config.FLO == 1:
                    psnr = self.psnr(torch.from_numpy(self.flow2img(images)), torch.from_numpy(self.flow2img(outputs_merged)))
                else:
                    assert(0)

                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # joint model
            else:
                # eval
                e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                e_outputs = e_outputs * masks + edges * (1 - masks)
                i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, images_filled, e_outputs, masks)

                if self.config.NORM == 1:
                    i_outputs = i_outputs * factor.reshape(-1, 1, 1, 1).type(i_outputs.dtype)
                    images = images * factor.reshape(-1, 1, 1, 1).type(i_outputs.dtype)
                    images_filled = images_filled * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)

                outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                # metrics
                if self.config.FLO == 0:
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                elif self.config.FLO == 1:
                    psnr = self.psnr(torch.from_numpy(self.flow2img(images)), torch.from_numpy(self.flow2img(outputs_merged)))
                else:
                    assert(0)

                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                e_logs.append(('pre', precision.item()))
                e_logs.append(('rec', recall.item()))
                i_logs.append(('psnr', psnr.item()))
                i_logs.append(('mae', mae.item()))
                logs = e_logs + i_logs


            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

    def test(self):
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_filled, images_gray, edges, masks, factor = self.cuda(*items)
            # images, images_gray, edges, masks = self.cuda(*items)
            index += 1

            # edge model
            if model == 1:
                outputs = self.edge_model(images_gray, edges, masks)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images, images_filled, edges, masks)

                if self.config.NORM == 1:
                    outputs = outputs * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                    images = images * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                    images_filled = images_filled * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # inpaint with edge model / joint model
            else:
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, images_filled, edges, masks)

                if self.config.NORM == 1:
                    outputs = outputs * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                    images = images * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                    images_filled = images_filled * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)

                outputs_merged = (outputs * masks) + (images * (1 - masks))

            if os.path.splitext(name)[1] == '.flo':
                name = os.path.splitext(name)[0] + '.png'

            if self.config.FLO == 0:
                output = self.postprocess(outputs_merged)[0]
            elif self.config.FLO == 1 and model == 1:
                output = self.postprocess(outputs_merged)[0]
            elif self.config.FLO == 1:
                output = torch.from_numpy(self.flow2img(outputs_merged.detach()))
            else:
                assert(0)
            path = os.path.join(self.results_path, name)
            print(index, name)

            imsave(output, path)

            if self.debug:
                edges = self.postprocess(1 - edges)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')

                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        # images, images_gray, edges, masks = self.cuda(*items)
        images, images_filled, images_gray, edges, masks, factor = self.cuda(*items)

        # cv2.imwrite('/home/gaochen/test.png', images_gray.detach().cpu().numpy()[0].transpose(1,2,0)*255)
        # cv2.imwrite('/home/gaochen/test.png', edges.detach().cpu().numpy()[0].transpose(1,2,0)*255)
        # edge model

        if model == 1:
            iteration = self.edge_model.iteration
            inputs = (images_gray * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks)
            outputs_merged = (outputs * masks) + (edges * (1 - masks))

        # inpaint model
        elif model == 2:
            iteration = self.inpaint_model.iteration
            outputs = self.inpaint_model(images, images_filled, edges, masks)

            if self.config.NORM == 1:
                outputs = outputs * factor.reshape(-1, 1, 1, 1).float()
                images = images * factor.reshape(-1, 1, 1, 1).float()
                images_filled = images_filled * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)

            outputs_merged = (outputs * masks) + (images * (1 - masks))
            inputs = (images * (1 - masks)) # + masks
        # inpaint with edge model / joint model
        else:
            iteration = self.inpaint_model.iteration
            outputs = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs * masks + edges * (1 - masks)).detach()
            outputs = self.inpaint_model(images, images_filled, edges, masks)

            if self.config.NORM == 1:
                outputs = outputs * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                images = images * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)
                images_filled = images_filled * factor.reshape(-1, 1, 1, 1).type(outputs.dtype)

            outputs_merged = (outputs * masks) + (images * (1 - masks))
            inputs = (images * (1 - masks)) # + masks
        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        if self.config.FLO == 0:
            images_ = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(edges),
                self.postprocess(outputs),
                self.postprocess(outputs_merged),
                img_per_row = image_per_row
            )
        elif self.config.FLO == 1:
            if self.config.FILL == 1:
                images_ = stitch_images(
                    torch.from_numpy(self.flow2img(images, 10)),
                    torch.from_numpy(self.flow2img(inputs, 10)),
                    torch.from_numpy(self.flow2img(images_filled, 10)),
                    self.postprocess(edges),
                    torch.from_numpy(self.flow2img(outputs.detach(), 10)),
                    torch.from_numpy(self.flow2img(outputs_merged.detach(), 10)),
                    img_per_row = image_per_row
                )
            else:
                images_ = stitch_images(
                    torch.from_numpy(self.flow2img(images, 10)),
                    torch.from_numpy(self.flow2img(inputs, 10)),
                    self.postprocess(edges),
                    torch.from_numpy(self.flow2img(outputs.detach(), 10)),
                    torch.from_numpy(self.flow2img(outputs_merged.detach(), 10)),
                    img_per_row = image_per_row
                )
        else:
            assert(0)
        #
        # self.writer.add_image('images', vutils.make_grid(torch.from_numpy(self.flow2img(images, 10)).permute(0, 3, 1, 2), scale_each=True), iteration)
        # self.writer.add_image('inputs', vutils.make_grid(torch.from_numpy(self.flow2img(inputs, 10)).permute(0, 3, 1, 2), scale_each=True), iteration)
        # self.writer.add_image('images_filled', vutils.make_grid(torch.from_numpy(self.flow2img(images_filled, 10)).permute(0, 3, 1, 2), scale_each=True), iteration)
        # self.writer.add_image('outputs', vutils.make_grid(torch.from_numpy(self.flow2img(outputs.detach(), 10)).permute(0, 3, 1, 2), scale_each=True), iteration)
        # self.writer.add_image('outputs_merged', vutils.make_grid(torch.from_numpy(self.flow2img(outputs_merged.detach(), 10)).permute(0, 3, 1, 2), scale_each=True), iteration)

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images_.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()


    def flow2img(self, flows, global_max=None):
        flows = flows.permute(0, 2, 3, 1)
        imgs = np.empty((0, flows.shape[1], flows.shape[2], 3), np.uint8)
        for idx in range(len(flows)):
            imgs = np.concatenate((imgs, np.expand_dims((self.flow_to_image(flows[idx, :, :, :].cpu().numpy(), global_max)), 0)), axis=0)
        return imgs


    def flow_to_image(self, flow, global_max):

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

        if global_max != None:
            maxrad = global_max
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
