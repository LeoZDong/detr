# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

import detr.util.box_ops as box_ops
# from util import box_ops
from detr.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

# from .backbone import build_backbone
from .matcher import build_matcher
# from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
#                            dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from .position_encoding import PositionalEncoding


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, transformer_outer, transformer_inner, num_classes, num_queries_out, num_queries_in, \
                 latent_dim, num_channels=256, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer_outer: outer transformer to handle block-level sequences. See transformer.py
            transformer_inner: inner transformer to handle step-level sequences. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            latent_dim: size of latent dimension for the latent vector of each predicted part.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        # Model parameters
        self.num_queries_out = num_queries_out
        self.num_queries_in = num_queries_in

        self.transformer_outer = transformer_outer
        self.transformer_inner = transformer_inner

        self.hidden_dim_outer = transformer_outer.d_model
        self.hidden_dim_inner = transformer_inner.d_model

        # Embedding / projection layers for hidden states of transformer
        # Embedding for outer transformer
        self.bbox_dim_embed = MLP(self.hidden_dim_outer, self.hidden_dim_outer, 6, 3)
        self.latent_embed = MLP(self.hidden_dim_outer, self.hidden_dim_outer, latent_dim, 3)
        # Embedding for inner transformer
        self.class_embed = nn.Linear(self.hidden_dim_inner, num_classes + 1)
        # (currently only supports translation, so 3 dimensions)
        self.bbox_trans_embed = MLP(self.hidden_dim_inner, self.hidden_dim_inner, 3, 3)

        # Query embedding for transformer
        self.query_embed_outer = nn.Embedding(num_queries_out, self.hidden_dim_outer)
        self.query_embed_inner = nn.Embedding(num_queries_in, self.hidden_dim_inner)

        # Input projection layers for transformer
        self.input_proj_outer = nn.Conv1d(num_channels, self.hidden_dim_outer, kernel_size=1)
        self.input_proj_inner = nn.Conv1d(num_channels, self.hidden_dim_inner, kernel_size=1)

        # Projects a 1 x d feature to num_queries_in x d sequence
        # Used to generate the source sequence of the inner transformer
        self.feat2seq_block = []
        for i in range(self.num_queries_out):
            feat2seq = []
            for j in range(self.num_queries_in):
                net = nn.Linear(self.hidden_dim_outer, self.hidden_dim_inner)
                feat2seq.append(net)
            feat2seq = nn.ModuleList(feat2seq)
            self.feat2seq_block.append(feat2seq)

        self.feat2seq_block = nn.ModuleList(self.feat2seq_block)

        self.aux_loss = aux_loss

    def forward(self, src_outer):
        """ The forward directly takes the source sequence that represents the
        input visual.

        Args:
            src (tensor): (bs, backbone.num_channels, num_queries) Source sequence.

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, center_z, height, width, depth). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "pred_latent": Predicted latent vectors for each predicted part.
                                Shape = [batch_size x num_queries x self.latent_dim]
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # Outer transformer
        mask_outer = torch.zeros((src_outer.shape[0], src_outer.shape[2]), dtype=torch.bool).to(src_outer.device)  # Do not mask anything
        pos_enc_outer = PositionalEncoding(self.transformer_outer.d_model)
        pos_outer = pos_enc_outer(src_outer)

        # Disable auxilary loss; directly use last hidden layer
        h_outer = self.transformer_outer(self.input_proj_outer(src_outer), mask_outer, self.query_embed_outer.weight, pos_outer)[0][-1]

        # Embedding for outer transformer
        outputs_bbox_dim = self.bbox_dim_embed(h_outer).sigmoid()
        outputs_latent = self.latent_embed(h_outer)

        # Inner transformer
        mask_inner = torch.zeros((h_outer.shape[0], self.num_queries_in), dtype=torch.bool).to(h_outer.device)
        pos_enc_inner = PositionalEncoding(self.transformer_inner.d_model)

        h_inner = []
        for i in range(self.num_queries_out):
            # Extract source sequence for inner transformer from outer transformer's hidden layer
            h_outer_i = h_outer[:, i, :]
            src_inner = []
            for j in range(self.num_queries_in):
                src_inner.append(self.feat2seq[i][j](h_outer_i))

            src_inner = torch.stack(src_inner, 2) # (bs, hidden_dim_inner, num_queries_in)
            pos_inner = pos_enc_inner(src_inner)

            # Pass through inner transformer
            h_inner_i = self.transformer_inner(self.input_proj_inner(src_inner), mask_inner, self.query_embed_inner.weight, pos_inner)[0][-1]
            h_inner.append(h_inner_i)

        h_inner = torch.stack(h_inner, 1) # (bs, num_queries_out, num_queries_in, dim)

        # Embedding for inner transformer
        outputs_bbox_trans = self.bbox_trans_embed(h_inner).sigmoid() - 0.5 # (bs, num_queries_out, num_queries_in, 6)
        outputs_class = self.class_embed(h_inner) # (bs, num_queries_out, num_queries_in, num_classes + 1)

        # Postprocessing
        out = self.postprocess(outputs_bbox_dim, outputs_latent, outputs_bbox_trans, outputs_class)
        return out

        # out = {'tokens_logits': outputs_class[-1], 'bbox': outputs_coord[-1], 'latent': outputs_latent}
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        # return out

    def postprocess(self, outputs_bbox_dim, outputs_latent, outputs_bbox_trans, outputs_class):
        """ Postprocess the outputs of the double transformer.

        Args:
            outputs_bbox_dim (tensor): (bs, num_queries_out, 6) Bounding box dimension for each block.
            outputs_latent (tensor): (bs, num_queries_out, dim) Latent code of each block.
            outputs_bbox_trans (tensor): (bs, num_queries_out, num_queries_in, 3) Bouding box translation for each step within a block.
            outputs_class (tensor): (bs, num_queries_out, num_queries_in, num_classes + 1) Class prediction for each step within a block.
                Currently, this serves as a present/absent binary classification.
        """
        # Apply step-wise transformation to predicted bounding box dimensions (TODO: make sure this stacking is alright without copy)
        outputs_bbox = torch.stack([outputs_bbox_dim] * self.num_queries_in, 2)
        # Translation along x, y, z axes
        outputs_bbox[:, :, :, :3] += outputs_bbox_trans

        outputs_latent = torch.stack([outputs_latent] * self.num_queries_in, 2)

        out = {'tokens_logits': outputs_class, 'bbox': outputs_bbox, 'latent': outputs_latent}
        return out



    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args, pdif_args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    transformer_outer = build_transformer(args)
    transformer_inner = build_transformer(args)

    model = DETR(
        transformer_outer,
        transformer_inner,
        num_classes=1,
        # num_queries=args.num_queries,
        num_queries_out=pdif_args.max_seq_len,
        num_queries_in=pdif_args.max_block_len,
        aux_loss=args.aux_loss,
        latent_dim=pdif_args.latent_dim
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
