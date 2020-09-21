from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import copy
import numpy as np
import scipy.io as sio
from utils.common_utils import interp, BFconsistCheck, \
    FBconsistCheck, consistCheck, get_KeySourceFrame_flowNN_gradient


def get_flowNN_gradient(args,
                        gradient_x,
                        gradient_y,
                        mask_RGB,
                        mask,
                        videoFlowF,
                        videoFlowB,
                        videoNonLocalFlowF,
                        videoNonLocalFlowB):

    # gradient_x:         imgH x (imgW - 1 + 1) x 3 x nFrame
    # gradient_y:         (imgH - 1 + 1) x imgW x 3 x nFrame
    # mask_RGB:           imgH x imgW x nFrame
    # mask:               imgH x imgW x nFrame
    # videoFlowF:         imgH x imgW x 2 x (nFrame - 1) | [u, v]
    # videoFlowB:         imgH x imgW x 2 x (nFrame - 1) | [u, v]
    # videoNonLocalFlowF: imgH x imgW x 2 x 3 x nFrame
    # videoNonLocalFlowB: imgH x imgW x 2 x 3 x nFrame

    if args.Nonlocal:
        num_candidate = 5
    else:
        num_candidate = 2
    imgH, imgW, nFrame = mask.shape
    numPix = np.sum(mask)

    # |--------------------|  |--------------------|
    # |       y            |  |       v            |
    # |   x   *            |  |   u   *            |
    # |                    |  |                    |
    # |--------------------|  |--------------------|

    # sub:            numPix * 3 | [y, x, t]
    # flowNN:         numPix * 3 * 2 | [y, x, t], [BN, FN]
    # HaveFlowNN:     imgH * imgW * nFrame * 2
    # numPixInd:      imgH * imgW * nFrame
    # consistencyMap: imgH * imgW * 5 * nFrame | [BN, FN, NL2, NL3, NL4]
    # consistency_uv: imgH * imgW * [BN, FN] * [u, v] * nFrame

    # sub: numPix * [y, x, t] | position of mising pixels
    sub = np.concatenate((np.where(mask == 1)[0].reshape(-1, 1),
                          np.where(mask == 1)[1].reshape(-1, 1),
                          np.where(mask == 1)[2].reshape(-1, 1)), axis=1)

    # flowNN: numPix * [y, x, t] * [BN, FN] | flow neighbors
    flowNN = np.ones((numPix, 3, 2)) * 99999   # * -1
    HaveFlowNN = np.ones((imgH, imgW, nFrame, 2)) * 99999
    HaveFlowNN[mask, :] = 0
    numPixInd = np.ones((imgH, imgW, nFrame)) * -1
    consistencyMap = np.zeros((imgH, imgW, num_candidate, nFrame))
    consistency_uv = np.zeros((imgH, imgW, 2, 2, nFrame))

    # numPixInd[y, x, t] gives the index of the missing pixel@[y, x, t] in sub,
    # i.e. which row. numPixInd[y, x, t] = idx; sub[idx, :] = [y, x, t]
    for idx in range(len(sub)):
        numPixInd[sub[idx, 0], sub[idx, 1], sub[idx, 2]] = idx

    # Initialization
    frameIndSetF = range(1, nFrame)
    frameIndSetB = range(nFrame - 2, -1, -1)

    # 1. Forward Pass (backward flow propagation)
    print('Forward Pass......')

    NN_idx = 0 # BN:0
    for indFrame in frameIndSetF:

        # Bool indicator of missing pixels at frame t
        holepixPosInd = (sub[:, 2] == indFrame)

        # Hole pixel location at frame t, i.e. [y, x, t]
        holepixPos = sub[holepixPosInd, :]

        # Calculate the backward flow neighbor. Should be located at frame t-1
        flowB_neighbor = copy.deepcopy(holepixPos)
        flowB_neighbor = flowB_neighbor.astype(np.float32)

        flowB_vertical = videoFlowB[:, :, 1, indFrame - 1]  # t --> t-1
        flowB_horizont = videoFlowB[:, :, 0, indFrame - 1]
        flowF_vertical = videoFlowF[:, :, 1, indFrame - 1]  # t-1 --> t
        flowF_horizont = videoFlowF[:, :, 0, indFrame - 1]

        flowB_neighbor[:, 0] += flowB_vertical[holepixPos[:, 0], holepixPos[:, 1]]
        flowB_neighbor[:, 1] += flowB_horizont[holepixPos[:, 0], holepixPos[:, 1]]
        flowB_neighbor[:, 2] -= 1

        # Round the backward flow neighbor location
        flow_neighbor_int = np.round(copy.deepcopy(flowB_neighbor)).astype(np.int32)

        # Chen: I should combine the following two operations together
        # Check the backward/forward consistency
        IsConsist, _ = BFconsistCheck(flowB_neighbor,
                                      flowF_vertical,
                                      flowF_horizont,
                                      holepixPos,
                                      args.consistencyThres)

        BFdiff, BF_uv = consistCheck(videoFlowF[:, :, :, indFrame - 1],
                                     videoFlowB[:, :, :, indFrame - 1])

        # Check out-of-boundary
        # Last column and last row does not have valid gradient
        ValidPos = np.logical_and(
            np.logical_and(flow_neighbor_int[:, 0] >= 0,
                           flow_neighbor_int[:, 0] < imgH - 1),
            np.logical_and(flow_neighbor_int[:, 1] >= 0,
                           flow_neighbor_int[:, 1] < imgW - 1))

        # Only work with pixels that are not out-of-boundary
        holepixPos = holepixPos[ValidPos, :]
        flowB_neighbor = flowB_neighbor[ValidPos, :]
        flow_neighbor_int = flow_neighbor_int[ValidPos, :]
        IsConsist = IsConsist[ValidPos]

        # For each missing pixel in holepixPos|[y, x, t],
        # we check its backward flow neighbor flowB_neighbor|[y', x', t-1].

        # Case 1: If mask[round(y'), round(x'), t-1] == 0,
        #         the backward flow neighbor of [y, x, t] is known.
        #         [y', x', t-1] is the backward flow neighbor.

        # KnownInd: Among all backward flow neighbors, which pixel is known.
        KnownInd = mask[flow_neighbor_int[:, 0],
                        flow_neighbor_int[:, 1],
                        indFrame - 1] == 0

        KnownIsConsist = np.logical_and(KnownInd, IsConsist)

        # We save backward flow neighbor flowB_neighbor in flowNN
        flowNN[numPixInd[holepixPos[KnownIsConsist, 0],
                         holepixPos[KnownIsConsist, 1],
                         indFrame].astype(np.int32), :, NN_idx] = \
                                                flowB_neighbor[KnownIsConsist, :]
        # flowNN[np.where(holepixPosInd == 1)[0][ValidPos][KnownIsConsist], :, 0] = \
        #                                         flowB_neighbor[KnownIsConsist, :]

        # We mark [y, x, t] in HaveFlowNN as 1
        HaveFlowNN[holepixPos[KnownIsConsist, 0],
                   holepixPos[KnownIsConsist, 1],
                   indFrame,
                   NN_idx] = 1

        # HaveFlowNN[:, :, :, 0]
        # 0: Backward flow neighbor can not be reached
        # 1: Backward flow neighbor can be reached
        # -1: Pixels that do not need to be completed

        consistency_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], NN_idx, 0, indFrame] = np.abs(BF_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], 0])
        consistency_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], NN_idx, 1, indFrame] = np.abs(BF_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], 1])

        # Case 2: If mask[round(y'), round(x'), t-1] == 1,
        #  the pixel@[round(y'), round(x'), t-1] is also occluded.
        #  We further check if we already assign a backward flow neighbor for the backward flow neighbor
        #  If HaveFlowNN[round(y'), round(x'), t-1] == 0,
        #   this is isolated pixel. Do nothing.
        #  If HaveFlowNN[round(y'), round(x'), t-1] == 1,
        #   we can borrow the value and refine it.

        UnknownInd = np.invert(KnownInd)

        # If we already assign a backward flow neighbor@[round(y'), round(x'), t-1]
        HaveNNInd = HaveFlowNN[flow_neighbor_int[:, 0],
                               flow_neighbor_int[:, 1],
                               indFrame - 1,
                               NN_idx] == 1

        # Unknown & IsConsist & HaveNNInd
        Valid_ = np.logical_and.reduce((UnknownInd, HaveNNInd, IsConsist))

        refineVec = np.concatenate((
            (flowB_neighbor[:, 0] - flow_neighbor_int[:, 0]).reshape(-1, 1),
            (flowB_neighbor[:, 1] - flow_neighbor_int[:, 1]).reshape(-1, 1),
            np.zeros((flowB_neighbor[:, 0].shape[0])).reshape(-1, 1)), 1)

        # Check if the transitive backward flow neighbor of [y, x, t] is known.
        # Sometimes after refinement, it is no longer known.
        flowNN_tmp = copy.deepcopy(flowNN[numPixInd[flow_neighbor_int[:, 0],
                                                    flow_neighbor_int[:, 1],
                                                    indFrame - 1].astype(np.int32), :, NN_idx] + refineVec[:, :])
        flowNN_tmp = np.round(flowNN_tmp).astype(np.int32)

        # Check out-of-boundary. flowNN_tmp may be out-of-boundary
        ValidPos_ = np.logical_and(
            np.logical_and(flowNN_tmp[:, 0] >= 0,
                           flowNN_tmp[:, 0] < imgH - 1),
            np.logical_and(flowNN_tmp[:, 1] >= 0,
                           flowNN_tmp[:, 1] < imgW - 1))

        # Change the out-of-boundary value to 0, in order to run mask[y,x,t]
        # in the next line. It won't affect anything as ValidPos_ is saved already
        flowNN_tmp[np.invert(ValidPos_), :] = 0
        ValidNN = mask[flowNN_tmp[:, 0],
                       flowNN_tmp[:, 1],
                       flowNN_tmp[:, 2]] == 0

        # Valid = np.logical_and.reduce((Valid_, ValidNN, ValidPos_))
        Valid = np.logical_and.reduce((Valid_, ValidPos_))

        # We save the transitive backward flow neighbor flowB_neighbor in flowNN
        flowNN[numPixInd[holepixPos[Valid, 0],
                         holepixPos[Valid, 1],
                         indFrame].astype(np.int32), :, NN_idx] = \
        flowNN[numPixInd[flow_neighbor_int[Valid, 0],
                         flow_neighbor_int[Valid, 1],
                         indFrame - 1].astype(np.int32), :, NN_idx] + refineVec[Valid, :]

        # We mark [y, x, t] in HaveFlowNN as 1
        HaveFlowNN[holepixPos[Valid, 0],
                   holepixPos[Valid, 1],
                   indFrame,
                   NN_idx] = 1

        consistency_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], NN_idx, 0, indFrame] = np.maximum(np.abs(BF_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], 0]), np.abs(consistency_uv[flow_neighbor_int[Valid, 0], flow_neighbor_int[Valid, 1], NN_idx, 0, indFrame - 1]))
        consistency_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], NN_idx, 1, indFrame] = np.maximum(np.abs(BF_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], 1]), np.abs(consistency_uv[flow_neighbor_int[Valid, 0], flow_neighbor_int[Valid, 1], NN_idx, 1, indFrame - 1]))

        consistencyMap[:, :, NN_idx, indFrame] = (consistency_uv[:, :, NN_idx, 0, indFrame] ** 2 + consistency_uv[:, :, NN_idx, 1, indFrame] ** 2) ** 0.5

        print("Frame {0:3d}: {1:8d} + {2:8d} = {3:8d}"
        .format(indFrame,
                np.sum(HaveFlowNN[:, :, indFrame, NN_idx] == 1),
                np.sum(HaveFlowNN[:, :, indFrame, NN_idx] == 0),
                np.sum(HaveFlowNN[:, :, indFrame, NN_idx] != 99999)))

    # 2. Backward Pass (forward flow propagation)
    print('Backward Pass......')

    NN_idx = 1 # FN:1
    for indFrame in frameIndSetB:

        # Bool indicator of missing pixels at frame t
        holepixPosInd = (sub[:, 2] == indFrame)

        # Hole pixel location at frame t, i.e. [y, x, t]
        holepixPos = sub[holepixPosInd, :]

        # Calculate the forward flow neighbor. Should be located at frame t+1
        flowF_neighbor = copy.deepcopy(holepixPos)
        flowF_neighbor = flowF_neighbor.astype(np.float32)

        flowF_vertical = videoFlowF[:, :, 1, indFrame]  # t --> t+1
        flowF_horizont = videoFlowF[:, :, 0, indFrame]
        flowB_vertical = videoFlowB[:, :, 1, indFrame]  # t+1 --> t
        flowB_horizont = videoFlowB[:, :, 0, indFrame]

        flowF_neighbor[:, 0] += flowF_vertical[holepixPos[:, 0], holepixPos[:, 1]]
        flowF_neighbor[:, 1] += flowF_horizont[holepixPos[:, 0], holepixPos[:, 1]]
        flowF_neighbor[:, 2] += 1

        # Round the forward flow neighbor location
        flow_neighbor_int = np.round(copy.deepcopy(flowF_neighbor)).astype(np.int32)

        # Check the forawrd/backward consistency
        IsConsist, _ = FBconsistCheck(flowF_neighbor,
                                      flowB_vertical,
                                      flowB_horizont,
                                      holepixPos,
                                      args.consistencyThres)

        FBdiff, FB_uv = consistCheck(videoFlowB[:, :, :, indFrame],
                                     videoFlowF[:, :, :, indFrame])

        # Check out-of-boundary
        # Last column and last row does not have valid gradient
        ValidPos = np.logical_and(
            np.logical_and(flow_neighbor_int[:, 0] >= 0,
                           flow_neighbor_int[:, 0] < imgH - 1),
            np.logical_and(flow_neighbor_int[:, 1] >= 0,
                           flow_neighbor_int[:, 1] < imgW - 1))

        # Only work with pixels that are not out-of-boundary
        holepixPos = holepixPos[ValidPos, :]
        flowF_neighbor = flowF_neighbor[ValidPos, :]
        flow_neighbor_int = flow_neighbor_int[ValidPos, :]
        IsConsist = IsConsist[ValidPos]

        # Case 1:
        KnownInd = mask[flow_neighbor_int[:, 0],
                        flow_neighbor_int[:, 1],
                        indFrame + 1] == 0

        KnownIsConsist = np.logical_and(KnownInd, IsConsist)
        flowNN[numPixInd[holepixPos[KnownIsConsist, 0],
                         holepixPos[KnownIsConsist, 1],
                         indFrame].astype(np.int32), :, NN_idx] = \
                                                flowF_neighbor[KnownIsConsist, :]

        HaveFlowNN[holepixPos[KnownIsConsist, 0],
                   holepixPos[KnownIsConsist, 1],
                   indFrame,
                   NN_idx] = 1

        consistency_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], NN_idx, 0, indFrame] = np.abs(FB_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], 0])
        consistency_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], NN_idx, 1, indFrame] = np.abs(FB_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], 1])

        # Case 2:
        UnknownInd = np.invert(KnownInd)
        HaveNNInd = HaveFlowNN[flow_neighbor_int[:, 0],
                               flow_neighbor_int[:, 1],
                               indFrame + 1,
                               NN_idx] == 1

        # Unknown & IsConsist & HaveNNInd
        Valid_ = np.logical_and.reduce((UnknownInd, HaveNNInd, IsConsist))

        refineVec = np.concatenate((
            (flowF_neighbor[:, 0] - flow_neighbor_int[:, 0]).reshape(-1, 1),
            (flowF_neighbor[:, 1] - flow_neighbor_int[:, 1]).reshape(-1, 1),
            np.zeros((flowF_neighbor[:, 0].shape[0])).reshape(-1, 1)), 1)

        # Check if the transitive backward flow neighbor of [y, x, t] is known.
        # Sometimes after refinement, it is no longer known.
        flowNN_tmp = copy.deepcopy(flowNN[numPixInd[flow_neighbor_int[:, 0],
                                                    flow_neighbor_int[:, 1],
                                                    indFrame + 1].astype(np.int32), :, NN_idx] + refineVec[:, :])
        flowNN_tmp = np.round(flowNN_tmp).astype(np.int32)

        # Check out-of-boundary. flowNN_tmp may be out-of-boundary
        ValidPos_ = np.logical_and(
            np.logical_and(flowNN_tmp[:, 0] >= 0,
                           flowNN_tmp[:, 0] < imgH - 1),
            np.logical_and(flowNN_tmp[:, 1] >= 0,
                           flowNN_tmp[:, 1] < imgW - 1))

        # Change the out-of-boundary value to 0, in order to run mask[y,x,t]
        # in the next line. It won't affect anything as ValidPos_ is saved already
        flowNN_tmp[np.invert(ValidPos_), :] = 0
        ValidNN = mask[flowNN_tmp[:, 0],
                       flowNN_tmp[:, 1],
                       flowNN_tmp[:, 2]] == 0

        # Valid = np.logical_and.reduce((Valid_, ValidNN, ValidPos_))
        Valid = np.logical_and.reduce((Valid_, ValidPos_))

        # We save the transitive backward flow neighbor flowB_neighbor in flowNN
        flowNN[numPixInd[holepixPos[Valid, 0],
                         holepixPos[Valid, 1],
                         indFrame].astype(np.int32), :, NN_idx] = \
        flowNN[numPixInd[flow_neighbor_int[Valid, 0],
                         flow_neighbor_int[Valid, 1],
                         indFrame + 1].astype(np.int32), :, NN_idx] + refineVec[Valid, :]

        # We mark [y, x, t] in HaveFlowNN as 1
        HaveFlowNN[holepixPos[Valid, 0],
                   holepixPos[Valid, 1],
                   indFrame,
                   NN_idx] = 1

        consistency_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], NN_idx, 0, indFrame] = np.maximum(np.abs(FB_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], 0]), np.abs(consistency_uv[flow_neighbor_int[Valid, 0], flow_neighbor_int[Valid, 1], NN_idx, 0, indFrame + 1]))
        consistency_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], NN_idx, 1, indFrame] = np.maximum(np.abs(FB_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], 1]), np.abs(consistency_uv[flow_neighbor_int[Valid, 0], flow_neighbor_int[Valid, 1], NN_idx, 1, indFrame + 1]))

        consistencyMap[:, :, NN_idx, indFrame] = (consistency_uv[:, :, NN_idx, 0, indFrame] ** 2 + consistency_uv[:, :, NN_idx, 1, indFrame] ** 2) ** 0.5

        print("Frame {0:3d}: {1:8d} + {2:8d} = {3:8d}"
        .format(indFrame,
                np.sum(HaveFlowNN[:, :, indFrame, NN_idx] == 1),
                np.sum(HaveFlowNN[:, :, indFrame, NN_idx] == 0),
                np.sum(HaveFlowNN[:, :, indFrame, NN_idx] != 99999)))

    # Interpolation
    gradient_x_BN = copy.deepcopy(gradient_x)
    gradient_y_BN = copy.deepcopy(gradient_y)
    gradient_x_FN = copy.deepcopy(gradient_x)
    gradient_y_FN = copy.deepcopy(gradient_y)

    for indFrame in range(nFrame):
        # Index of missing pixel whose backward flow neighbor is from frame indFrame
        SourceFmInd = np.where(flowNN[:, 2, 0] == indFrame)

        print("{0:8d} pixels are from source Frame {1:3d}"
                        .format(len(SourceFmInd[0]), indFrame))
        # The location of the missing pixel whose backward flow neighbor is
        # from frame indFrame flowNN[SourceFmInd, 0, 0], flowNN[SourceFmInd, 1, 0]

        if len(SourceFmInd[0]) != 0:

            # |--------------------|
            # |       y            |
            # |   x   *            |
            # |                    |
            # |--------------------|
            # sub: numPix x 3 [y, x, t]
            # img: [y, x]
            # interp(img, x, y)

            gradient_x_BN[sub[SourceFmInd[0], :][:, 0],
                    sub[SourceFmInd[0], :][:, 1],
                 :, sub[SourceFmInd[0], :][:, 2]] = \
                interp(gradient_x_BN[:, :, :, indFrame],
                        flowNN[SourceFmInd, 1, 0].reshape(-1),
                        flowNN[SourceFmInd, 0, 0].reshape(-1))

            gradient_y_BN[sub[SourceFmInd[0], :][:, 0],
                    sub[SourceFmInd[0], :][:, 1],
                 :, sub[SourceFmInd[0], :][:, 2]] = \
                interp(gradient_y_BN[:, :, :, indFrame],
                        flowNN[SourceFmInd, 1, 0].reshape(-1),
                        flowNN[SourceFmInd, 0, 0].reshape(-1))

            assert(((sub[SourceFmInd[0], :][:, 2] - indFrame) <= 0).sum() == 0)

    for indFrame in range(nFrame - 1, -1, -1):
        # Index of missing pixel whose forward flow neighbor is from frame indFrame
        SourceFmInd = np.where(flowNN[:, 2, 1] == indFrame)
        print("{0:8d} pixels are from source Frame {1:3d}"
                        .format(len(SourceFmInd[0]), indFrame))
        if len(SourceFmInd[0]) != 0:

            gradient_x_FN[sub[SourceFmInd[0], :][:, 0],
                          sub[SourceFmInd[0], :][:, 1],
                       :, sub[SourceFmInd[0], :][:, 2]] = \
                interp(gradient_x_FN[:, :, :, indFrame],
                         flowNN[SourceFmInd, 1, 1].reshape(-1),
                         flowNN[SourceFmInd, 0, 1].reshape(-1))

            gradient_y_FN[sub[SourceFmInd[0], :][:, 0],
                    sub[SourceFmInd[0], :][:, 1],
                 :, sub[SourceFmInd[0], :][:, 2]] = \
                interp(gradient_y_FN[:, :, :, indFrame],
                         flowNN[SourceFmInd, 1, 1].reshape(-1),
                         flowNN[SourceFmInd, 0, 1].reshape(-1))

            assert(((indFrame - sub[SourceFmInd[0], :][:, 2]) <= 0).sum() == 0)

    # New mask
    mask_tofill = np.zeros((imgH, imgW, nFrame)).astype(np.bool)

    for indFrame in range(nFrame):
        if args.Nonlocal:
            consistencyMap[:, :, 2, indFrame], _ = consistCheck(
                videoNonLocalFlowB[:, :, :, 0, indFrame],
                videoNonLocalFlowF[:, :, :, 0, indFrame])
            consistencyMap[:, :, 3, indFrame], _ = consistCheck(
                videoNonLocalFlowB[:, :, :, 1, indFrame],
                videoNonLocalFlowF[:, :, :, 1, indFrame])
            consistencyMap[:, :, 4, indFrame], _ = consistCheck(
                videoNonLocalFlowB[:, :, :, 2, indFrame],
                videoNonLocalFlowF[:, :, :, 2, indFrame])

        HaveNN = np.zeros((imgH, imgW, num_candidate))

        if args.Nonlocal:
            HaveKeySourceFrameFlowNN, gradient_x_KeySourceFrameFlowNN, gradient_y_KeySourceFrameFlowNN = \
                get_KeySourceFrame_flowNN_gradient(sub,
                                                  indFrame,
                                                  mask,
                                                  videoNonLocalFlowB,
                                                  videoNonLocalFlowF,
                                                  gradient_x,
                                                  gradient_y,
                                                  args.consistencyThres)

            HaveNN[:, :, 2] = HaveKeySourceFrameFlowNN[:, :, 0] == 1
            HaveNN[:, :, 3] = HaveKeySourceFrameFlowNN[:, :, 1] == 1
            HaveNN[:, :, 4] = HaveKeySourceFrameFlowNN[:, :, 2] == 1

        HaveNN[:, :, 0] = HaveFlowNN[:, :, indFrame, 0] == 1
        HaveNN[:, :, 1] = HaveFlowNN[:, :, indFrame, 1] == 1

        NotHaveNN = np.logical_and(np.invert(HaveNN.astype(np.bool)),
                np.repeat(np.expand_dims((mask[:, :, indFrame]), 2), num_candidate, axis=2))

        if args.Nonlocal:
            HaveNN_sum = np.logical_or.reduce((HaveNN[:, :, 0],
                                               HaveNN[:, :, 1],
                                               HaveNN[:, :, 2],
                                               HaveNN[:, :, 3],
                                               HaveNN[:, :, 4]))
        else:
            HaveNN_sum = np.logical_or.reduce((HaveNN[:, :, 0],
                                               HaveNN[:, :, 1]))

        gradient_x_Candidate = np.zeros((imgH, imgW, 3, num_candidate))
        gradient_y_Candidate = np.zeros((imgH, imgW, 3, num_candidate))

        gradient_x_Candidate[:, :, :, 0] = gradient_x_BN[:, :, :, indFrame]
        gradient_y_Candidate[:, :, :, 0] = gradient_y_BN[:, :, :, indFrame]
        gradient_x_Candidate[:, :, :, 1] = gradient_x_FN[:, :, :, indFrame]
        gradient_y_Candidate[:, :, :, 1] = gradient_y_FN[:, :, :, indFrame]

        if args.Nonlocal:
            gradient_x_Candidate[:, :, :, 2] = gradient_x_KeySourceFrameFlowNN[:, :, :, 0]
            gradient_y_Candidate[:, :, :, 2] = gradient_y_KeySourceFrameFlowNN[:, :, :, 0]
            gradient_x_Candidate[:, :, :, 3] = gradient_x_KeySourceFrameFlowNN[:, :, :, 1]
            gradient_y_Candidate[:, :, :, 3] = gradient_y_KeySourceFrameFlowNN[:, :, :, 1]
            gradient_x_Candidate[:, :, :, 4] = gradient_x_KeySourceFrameFlowNN[:, :, :, 2]
            gradient_y_Candidate[:, :, :, 4] = gradient_y_KeySourceFrameFlowNN[:, :, :, 2]

        consistencyMap[:, :, :, indFrame] = np.exp( - consistencyMap[:, :, :, indFrame] / args.alpha)

        consistencyMap[NotHaveNN[:, :, 0], 0, indFrame] = 0
        consistencyMap[NotHaveNN[:, :, 1], 1, indFrame] = 0

        if args.Nonlocal:
            consistencyMap[NotHaveNN[:, :, 2], 2, indFrame] = 0
            consistencyMap[NotHaveNN[:, :, 3], 3, indFrame] = 0
            consistencyMap[NotHaveNN[:, :, 4], 4, indFrame] = 0

        weights = (consistencyMap[HaveNN_sum, :, indFrame] * HaveNN[HaveNN_sum, :]) / ((consistencyMap[HaveNN_sum, :, indFrame] * HaveNN[HaveNN_sum, :]).sum(axis=1, keepdims=True))

        # Fix the numerical issue. 0 / 0
        fix = np.where((consistencyMap[HaveNN_sum, :, indFrame] * HaveNN[HaveNN_sum, :]).sum(axis=1, keepdims=True) == 0)[0]
        weights[fix, :] = HaveNN[HaveNN_sum, :][fix, :] / HaveNN[HaveNN_sum, :][fix, :].sum(axis=1, keepdims=True)

        # Fuse RGB channel independently
        gradient_x[HaveNN_sum, 0, indFrame] = \
            np.sum(np.multiply(gradient_x_Candidate[HaveNN_sum, 0, :], weights), axis=1)
        gradient_x[HaveNN_sum, 1, indFrame] = \
            np.sum(np.multiply(gradient_x_Candidate[HaveNN_sum, 1, :], weights), axis=1)
        gradient_x[HaveNN_sum, 2, indFrame] = \
            np.sum(np.multiply(gradient_x_Candidate[HaveNN_sum, 2, :], weights), axis=1)

        gradient_y[HaveNN_sum, 0, indFrame] = \
            np.sum(np.multiply(gradient_y_Candidate[HaveNN_sum, 0, :], weights), axis=1)
        gradient_y[HaveNN_sum, 1, indFrame] = \
            np.sum(np.multiply(gradient_y_Candidate[HaveNN_sum, 1, :], weights), axis=1)
        gradient_y[HaveNN_sum, 2, indFrame] = \
            np.sum(np.multiply(gradient_y_Candidate[HaveNN_sum, 2, :], weights), axis=1)

        mask_tofill[np.logical_and(np.invert(HaveNN_sum), mask[:, :, indFrame]), indFrame] = True

    return gradient_x, gradient_y, mask_tofill
