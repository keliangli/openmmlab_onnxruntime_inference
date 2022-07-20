import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import torch.nn.functional as F
from itertools import product
from math import sqrt
import numpy as np
from torch.nn.modules.utils import _pair

max_num_detections = 40
nms_pre = 1000

strides=[550.0 / x for x in [69, 35, 18, 9, 5]]
strides = [_pair(stride) for stride in strides]

centers=[(550 * 0.5 / x, 550 * 0.5 / x) for x in [69, 35, 18, 9, 5]]
ratios=torch.from_numpy(np.array([0.5, 1.0, 2.0]))

base_sizes=torch.from_numpy(np.array([8, 16, 32, 64, 128]))

octave_base_scale=3
scales_per_octave=1

octave_scales = np.array([2**(i / scales_per_octave) for i in range(scales_per_octave)])
scales = torch.from_numpy(octave_scales * octave_base_scale)


def gen_base_anchors():
    """Generate base anchors.

    Returns:
        list(torch.Tensor): Base anchors of a feature grid in multiple \
            feature levels.
    """
    multi_level_base_anchors = []
    for i, base_size in enumerate(base_sizes):
        center = centers[i]
        multi_level_base_anchors.append(
                gen_single_level_base_anchors(
                base_size,
                scales=scales,
                ratios=ratios,
                center=center))

    return multi_level_base_anchors

def gen_single_level_base_anchors(base_size,
                                  scales,
                                  ratios,
                                  center=None):
    """Generate base anchors of a single level.

    Args:
        base_size (int | float): Basic size of an anchor.
        scales (torch.Tensor): Scales of the anchor.
        ratios (torch.Tensor): The ratio between between the height
            and width of anchors in a single level.
        center (tuple[float], optional): The center of the base anchor
            related to a single feature grid. Defaults to None.

    Returns:
        torch.Tensor: Anchors in a single-level feature maps.
    """
    w = base_size
    h = base_size
    x_center, y_center = center

    h_ratios = torch.sqrt(ratios)
    w_ratios = 1 / h_ratios

    ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)

    # use float anchor and the anchor's center is aligned with the
    # pixel center
    base_anchors = [
        x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
        y_center + 0.5 * hs
    ]
    base_anchors = torch.stack(base_anchors, dim=-1)

    return base_anchors



def _meshgrid(x, y, row_major=True):
    """Generate mesh grid of x and y.

    Args:
        x (torch.Tensor): Grids of x dimension.
        y (torch.Tensor): Grids of y dimension.
        row_major (bool, optional): Whether to return y grids first.
            Defaults to True.

    Returns:
        tuple[torch.Tensor]: The mesh grids of x and y.
    """
    # use shape instead of len to keep tracing while exporting to onnx
    xx = x.repeat(y.shape[0])
    yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx


def single_level_grid_priors(featmap_size,
                             level_idx,
                             dtype=torch.float32):
    """Generate grid anchors of a single level.

    Note:
        This function is usually called by method ``self.grid_priors``.

    Args:
        featmap_size (tuple[int]): Size of the feature maps.
        level_idx (int): The index of corresponding feature map level.
        dtype (obj:`torch.dtype`): Date type of points.Defaults to
            ``torch.float32``.
        device (str, optional): The device the tensor will be put on.
            Defaults to 'cuda'.

    Returns:
        torch.Tensor: Anchors in the overall feature maps.
    """

    base_anchors = gen_base_anchors()[level_idx].to(dtype)

    feat_h, feat_w = featmap_size
    stride_w, stride_h = strides[level_idx]
    # First create Range with the default dtype, than convert to
    # target `dtype` for onnx exporting.
    shift_x = torch.arange(0, feat_w).to(dtype) * stride_w
    shift_y = torch.arange(0, feat_h).to(dtype) * stride_h

    shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
    shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
    # first feat_w elements correspond to the first row of shifts
    # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
    # shifted anchors (K, A, 4), reshape to (K*A, 4)

    all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
    all_anchors = all_anchors.view(-1, 4)
    # first A rows correspond to A anchors of (0, 0) in feature map,
    # then (0, 1), (0, 2), ...
    return all_anchors


def grid_priors(featmap_sizes, dtype=torch.float32):
    """Generate grid anchors in multiple feature levels.

    Args:
        featmap_sizes (list[tuple]): List of feature map sizes in
            multiple feature levels.
        dtype (:obj:`torch.dtype`): Dtype of priors.
            Default: torch.float32.
        device (str): The device where the anchors will be put on.

    Return:
        list[torch.Tensor]: Anchors in multiple feature levels. \
            The sizes of each tensor should be [N, 4], where \
            N = width * height * num_base_anchors, width and height \
            are the sizes of the corresponding feature level, \
            num_base_anchors is the number of anchors for that level.
    """

    num_levels = len(featmap_sizes)

    multi_level_anchors = []
    for i in range(num_levels):
        anchors = single_level_grid_priors(featmap_sizes[i], level_idx=i, dtype=dtype)
        multi_level_anchors.append(anchors)
    return multi_level_anchors


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    return torch.clamp(max_xy - min_xy, min=0).prod(3)  # inter

def jaccard(box_a, box_b, iscrowd:bool=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs






def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(0.1, 0.1, 0.2, 0.2),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               clip_border=True,
               add_ctr_clamp=False,
               ctr_clamp=32):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 4) or (N, 4). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1.).
        max_shape (tuple[int, int]): Maximum bounds for boxes, specifies
           (H, W). Default None.
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Default True.
        add_ctr_clamp (bool): Whether to add center clamp. When set to True,
            the center of the prediction bounding box will be clamped to
            avoid being too far away from the center of the anchor.
            Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 4) or (N, 4), where 4
           represent tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32, 3))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """

    num_bboxes, num_classes = deltas.size(0), deltas.size(1) // 4
    if num_bboxes == 0:
        return deltas

    deltas = deltas.reshape(-1, 4)

    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    denorm_deltas = deltas * stds + means

    dxy = denorm_deltas[:, :2]
    dwh = denorm_deltas[:, 2:]

    # Compute width/height of each roi
    rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
    pxy = ((rois_[:, :2] + rois_[:, 2:]) * 0.5)
    pwh = (rois_[:, 2:] - rois_[:, :2])

    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes = torch.cat([x1y1, x2y2], dim=-1)
    if clip_border and max_shape is not None:
        bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
    bboxes = bboxes.reshape(num_bboxes, -1)
    return bboxes


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]

#generator anchor
featmap_sizes = [(69, 69),(35, 35),(18, 18),(9, 9),(5, 5)]
anchors =  grid_priors(featmap_sizes, dtype=torch.float32)

def detect():

    # Run inference
    img = cv2.imread("demo.jpg")
    img = cv2.resize(img,(550,550))
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img).to('cpu')
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    import onnxruntime
    session = onnxruntime.InferenceSession("yolact_r50_1x8_coco_new.onnx")

    img = img.cpu().numpy()  # torch to numpy

    outputs = []
    for item in session.get_outputs():
        outputs.append(item.name)

    pred = session.run(outputs, {session.get_inputs()[0].name: img})

    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_coeffs = []

    cls_52 = torch.from_numpy(pred[0]).permute(0, 2, 3, 1).reshape(-1, 81).softmax(-1)
    loc_52 = torch.from_numpy(pred[5]).permute(0, 2, 3, 1).contiguous().reshape(-1, 4)
    mask_52 = torch.from_numpy(pred[10]).permute(0, 2, 3, 1).contiguous().reshape(-1, 32)

    priors_52 = anchors[0]

    if 0 < nms_pre < cls_52.shape[0]:
        max_scores, _ = cls_52[:, :-1].max(dim=1)
        _, topk_inds = max_scores.topk(nms_pre)
        priors_52 = priors_52[topk_inds, :]
        loc_52 = loc_52[topk_inds, :]
        cls_52 = cls_52[topk_inds, :]
        mask_52 = mask_52[topk_inds, :]

    bboxes = delta2bbox(priors_52,loc_52,max_shape=(550, 550, 3))

    mlvl_bboxes.append(bboxes)
    mlvl_scores.append(cls_52)
    mlvl_coeffs.append(mask_52)


    cls_26 = torch.from_numpy(pred[1]).permute(0, 2, 3, 1).contiguous().reshape(-1, 81).softmax(-1)
    loc_26 = torch.from_numpy(pred[6]).permute(0, 2, 3, 1).contiguous().reshape(-1, 4)
    mask_26 = torch.from_numpy(pred[11]).permute(0, 2, 3, 1).contiguous().reshape(-1, 32)

    priors_26 = anchors[1]

    if 0 < nms_pre < cls_26.shape[0]:
        max_scores, _ = priors_26[:, :-1].max(dim=1)
        _, topk_inds = max_scores.topk(nms_pre)
        priors_26 = priors_26[topk_inds, :]
        loc_26 = loc_26[topk_inds, :]
        cls_26 = cls_26[topk_inds, :]
        mask_26 = mask_26[topk_inds, :]

    bboxes = delta2bbox(priors_26,loc_26,max_shape=(550, 550, 3))

    mlvl_bboxes.append(bboxes)
    mlvl_scores.append(cls_26)
    mlvl_coeffs.append(mask_26)


    cls_13 = torch.from_numpy(pred[2]).permute(0, 2, 3, 1).contiguous().reshape(-1, 81).softmax(-1)
    loc_13 = torch.from_numpy(pred[7]).permute(0, 2, 3, 1).contiguous().reshape(-1, 4)
    mask_13 = torch.from_numpy(pred[12]).permute(0, 2, 3, 1).contiguous().reshape(-1, 32)

    priors_13 = anchors[2]

    if 0 < nms_pre < cls_13.shape[0]:
        max_scores, _ = priors_13[:, :-1].max(dim=1)
        _, topk_inds = max_scores.topk(nms_pre)
        priors_13 = priors_13[topk_inds, :]
        loc_13 = loc_13[topk_inds, :]
        cls_13 = cls_13[topk_inds, :]
        mask_13 = mask_13[topk_inds, :]

    bboxes = delta2bbox(priors_13,loc_13,max_shape=(550, 550, 3))

    mlvl_bboxes.append(bboxes)
    mlvl_scores.append(cls_13)
    mlvl_coeffs.append(mask_13)

    cls_7 = torch.from_numpy(pred[3]).permute(0, 2, 3, 1).contiguous().reshape(-1, 81).softmax(-1)
    loc_7 = torch.from_numpy(pred[8]).permute(0, 2, 3, 1).contiguous().reshape(-1, 4)
    mask_7 = torch.from_numpy(pred[13]).permute(0, 2, 3, 1).contiguous().reshape(-1, 32)

    priors_7 = anchors[3]

    if 0 < nms_pre < cls_7.shape[0]:
        max_scores, _ = priors_7[:, :-1].max(dim=1)
        _, topk_inds = max_scores.topk(nms_pre)
        priors_7 = priors_7[topk_inds, :]
        loc_7 = loc_7[topk_inds, :]
        cls_7 = cls_7[topk_inds, :]
        mask_7 = mask_7[topk_inds, :]

    bboxes = delta2bbox(priors_7,loc_7,max_shape=(550, 550, 3))

    mlvl_bboxes.append(bboxes)
    mlvl_scores.append(cls_7)
    mlvl_coeffs.append(mask_7)

    cls_4 = torch.from_numpy(pred[4]).permute(0, 2, 3, 1).contiguous().reshape(-1, 81).softmax(-1)
    loc_4 = torch.from_numpy(pred[9]).permute(0, 2, 3, 1).contiguous().reshape(-1, 4)
    mask_4 = torch.from_numpy(pred[14]).permute(0, 2, 3, 1).contiguous().reshape(-1, 32)

    priors_4 = anchors[4]

    if 0 < nms_pre < cls_4.shape[0]:
        max_scores, _ = priors_4[:, :-1].max(dim=1)
        _, topk_inds = max_scores.topk(nms_pre)
        priors_4 = priors_4[topk_inds, :]
        loc_4 = loc_4[topk_inds, :]
        cls_4 = cls_4[topk_inds, :]
        mask_4 = mask_4[topk_inds, :]

    bboxes = delta2bbox(priors_4,loc_4,max_shape=(550, 550, 3))

    mlvl_bboxes.append(bboxes)
    mlvl_scores.append(cls_4)
    mlvl_coeffs.append(mask_4)

    proto = torch.from_numpy(pred[15]).permute(0, 2, 3, 1).contiguous()

    mlvl_bboxes = torch.cat(mlvl_bboxes)
    mlvl_bboxes /= mlvl_bboxes.new_tensor([1.,1.,1.,1.])
    mlvl_scores = torch.cat(mlvl_scores)
    mlvl_coeffs = torch.cat(mlvl_coeffs)

    det_bboxes, det_labels, det_coeffs = fast_nms(mlvl_bboxes, mlvl_scores,mlvl_coeffs,0.5,0.5,10,100)

    bbox_results = [bbox2result(det_bbox, det_label,81) for det_bbox, det_label in zip(det_bboxes, det_labels)]
 
    print(det_bboxes)
    print(det_labels)
    print(det_coeffs)

if __name__ == '__main__':
    detect()
