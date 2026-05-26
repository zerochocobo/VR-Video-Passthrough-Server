"""Export MatAnyone2 inference components to ONNX.

Run this script in a separate PyTorch/MatAnyone2 export environment. The main
PTMediaServer runtime should stay ONNX-only and does not import this file.

MatAnyone2 does not implement a single forward() suitable for ONNX export. Its
official inference path calls submodules through InferenceCore.step(), so this
script exports the practical pieces needed by an ONNX Runtime pipeline:

1. image_key.onnx
   image -> multi-scale image features, pixel feature, key, shrinkage, selection
2. mask_memory.onnx
   image + first-frame mask + image features + sensory -> mask value, sensory,
   object summaries
3. first_frame_refine.onnx
   first-frame memory read + decoder refinement
4. propagate.onnx
   memory read + decoder for following frames
5. propagate_update.onnx
   optional fused propagate + mask-memory update for following frames
6. step_update.onnx
   optional full following-frame step: image -> image features -> propagate ->
   mask-memory update

The exported graphs are fixed to the requested export height/width. Use a
divisible-by-16 size that matches the offline preprocessing size, e.g. 512 or
1024. Keeping fixed spatial shapes is intentional for the first integration
because MatAnyone2's memory tensors are shape-sensitive and easier to validate
without dynamic HW axes.


uv run python export_matanyone2_onnx.py --out-dir models/matanyone2_onnx_1024_bs1 --height 1024 --width 1024 --batch-size 1


"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


def _import_torch():
    try:
        import torch
    except Exception as e:  # pragma: no cover - export environment only
        raise RuntimeError(
            "PyTorch is required only in the separate MatAnyone2 export environment. "
            "Install MatAnyone2 there, then run this script from that environment."
        ) from e
    return torch


def _load_model(args, torch):
    try:
        from matanyone2 import MatAnyone2
    except Exception:
        try:
            from matanyone2.model.matanyone2 import MatAnyone2
        except Exception as e:
            raise RuntimeError(
                "Cannot import MatAnyone2. Install the upstream project first, e.g. "
                "`pip install git+https://github.com/pq-yang/MatAnyone2.git`."
            ) from e

    if args.repo_id:
        model = MatAnyone2.from_pretrained(args.repo_id)
    else:
        raise RuntimeError("--repo-id is currently required because upstream local checkpoint loading needs cfg.")

    model.to(args.device)
    model.eval()
    return model


def _export(torch, module, inputs: tuple, out_path: Path, input_names: list[str], output_names: list[str], opset: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        kwargs = {
            "input_names": input_names,
            "output_names": output_names,
            "opset_version": opset,
            "do_constant_folding": True,
            "export_params": True,
        }
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            torch.onnx.export(module, inputs, str(out_path), dynamo=False, **kwargs)
        else:
            torch.onnx.export(module, inputs, str(out_path), **kwargs)


def _safe_check_onnx(path: Path) -> str:
    try:
        import onnx

        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        return "ok"
    except Exception as e:
        return f"check skipped/failed: {type(e).__name__}: {e}"


def _patch_matanyone2_export_ops(torch) -> None:
    """Patch MatAnyone2 ops that legacy ONNX export cannot lower.

    Upstream `downsample_groups(..., mode="area")` lowers to adaptive average
    pooling with an output size computed from tensor shapes. The legacy exporter
    requires that output size to be a Python constant. For fixed-shape export,
    the ratios used by MatAnyone2 are powers of two, so avg_pool2d with equal
    kernel/stride is equivalent and exports cleanly.
    """
    import torch.nn.functional as F
    import matanyone2.model.big_modules as big_modules
    import matanyone2.model.group_modules as group_modules
    import matanyone2.model.modules as modules
    import matanyone2.model.transformer.object_summarizer as object_summarizer
    import matanyone2.model.transformer.positional_encoding as positional_encoding

    def _interpolate_groups(g, ratio: float, mode: str, align_corners):
        batch_size, num_objects = g.shape[:2]
        flat = g.flatten(start_dim=0, end_dim=1)
        if mode == "area" and ratio < 1:
            inv = round(1.0 / float(ratio))
            if abs((1.0 / float(ratio)) - inv) > 1e-6 or inv <= 0:
                raise RuntimeError(f"unsupported MatAnyone2 export downsample ratio: {ratio}")
            flat = F.avg_pool2d(flat, kernel_size=inv, stride=inv)
        else:
            flat = F.interpolate(flat, scale_factor=ratio, mode=mode, align_corners=align_corners)
        return flat.view(batch_size, num_objects, *flat.shape[1:])

    def _upsample_groups(g, ratio: float = 2, mode: str = "bilinear", align_corners: bool = False):
        return _interpolate_groups(g, ratio, mode, align_corners)

    def _downsample_groups(g, ratio: float = 1 / 2, mode: str = "area", align_corners=None):
        return _interpolate_groups(g, ratio, mode, align_corners)

    group_modules.interpolate_groups = _interpolate_groups
    group_modules.upsample_groups = _upsample_groups
    group_modules.downsample_groups = _downsample_groups
    modules.upsample_groups = _upsample_groups
    modules.downsample_groups = _downsample_groups

    def _gconv2d_forward(self, g):
        batch_size, num_objects = g.shape[:2]
        flat = g.flatten(start_dim=0, end_dim=1).to(self.weight.dtype)
        out = torch.nn.functional.conv2d(
            flat,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return out.view(batch_size, num_objects, *out.shape[1:])

    if hasattr(group_modules, "GConv2d"):
        group_modules.GConv2d.forward = _gconv2d_forward

    def _module_dtype(module, fallback):
        for param in module.parameters(recurse=True):
            return param.dtype
        return fallback

    def _sensory_fullscale_forward(self, g, h):
        g = self.g16_conv(g[0]) + self.g8_conv(_downsample_groups(g[1], ratio=1 / 2)) + \
            self.g4_conv(_downsample_groups(g[2], ratio=1 / 4)) + \
            self.g2_conv(_downsample_groups(g[3], ratio=1 / 8)) + \
            self.g1_conv(_downsample_groups(g[4], ratio=1 / 16))
        dtype = _module_dtype(self.transform, g.dtype)
        values = self.transform(torch.cat([g.to(dtype), h.to(dtype)], dim=2))
        return modules._recurrent_update(h.to(dtype), values)

    def _sensory_forward(self, g, h):
        g = self.g16_conv(g[0]) + self.g8_conv(_downsample_groups(g[1], ratio=1 / 2)) + \
            self.g4_conv(_downsample_groups(g[2], ratio=1 / 4))
        dtype = _module_dtype(self.transform, g.dtype)
        values = self.transform(torch.cat([g.to(dtype), h.to(dtype)], dim=2))
        return modules._recurrent_update(h.to(dtype), values)

    def _sensory_deep_forward(self, g, h):
        dtype = _module_dtype(self.transform, g.dtype)
        values = self.transform(torch.cat([g.to(dtype), h.to(dtype)], dim=2))
        return modules._recurrent_update(h.to(dtype), values)

    if hasattr(modules, "SensoryUpdater_fullscale"):
        modules.SensoryUpdater_fullscale.forward = _sensory_fullscale_forward
    if hasattr(modules, "SensoryUpdater"):
        modules.SensoryUpdater.forward = _sensory_forward
    if hasattr(modules, "SensoryDeepUpdater"):
        modules.SensoryDeepUpdater.forward = _sensory_deep_forward

    def _object_summarizer_forward(self, masks, value, need_weights: bool = False):
        h, w = value.shape[-2:]
        masks = F.interpolate(masks, size=(h, w), mode="area")
        masks = masks.unsqueeze(-1)
        inv_masks = 1 - masks
        repeated_masks = torch.cat([
            masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
            inv_masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
        ], dim=-1)
        value = value.permute(0, 1, 3, 4, 2)
        value = self.input_proj(value)
        if self.add_pe:
            value = value + self.pos_enc(value)
        dtype = _module_dtype(self.feature_pred, value.dtype)
        value = value.to(dtype)
        repeated_masks = repeated_masks.to(dtype)
        feature = self.feature_pred(value)
        logits = self.weights_pred(value)
        sums, area = object_summarizer._weighted_pooling(repeated_masks, feature, logits)
        summaries = torch.cat([sums, area], dim=-1)
        if need_weights:
            return summaries, logits
        return summaries, None

    if hasattr(object_summarizer, "ObjectSummarizer"):
        object_summarizer.ObjectSummarizer.forward = _object_summarizer_forward

    def _mask_decoder_forward(
        self,
        ms_image_feat,
        memory_readout,
        sensory,
        *,
        chunk_size: int = -1,
        update_sensory: bool = True,
        seg_pass: bool = False,
        last_mask=None,
        sigmoid_residual=False,
    ):
        batch_size, num_objects = memory_readout.shape[:2]
        f8, f4, f2, f1 = self.decoder_feat_proc(ms_image_feat[1:])
        if chunk_size < 1 or chunk_size >= num_objects:
            chunk_size = num_objects
            fast_path = True
            new_sensory = sensory
        else:
            new_sensory = torch.empty_like(sensory) if update_sensory else sensory
            fast_path = False
        all_logits = []
        for i in range(0, num_objects, chunk_size):
            p16 = memory_readout if fast_path else memory_readout[:, i:i + chunk_size]
            actual_chunk_size = p16.shape[1]
            p8 = self.up_16_8(p16, f8)
            p4 = self.up_8_4(p8, f4)
            p2 = self.up_4_2(p4, f2)
            p1 = self.up_2_1(p2, f1)
            pred = self.pred_seg if seg_pass else self.pred_mat
            dtype = _module_dtype(pred, p1.dtype)
            pred_in = F.relu(p1.flatten(start_dim=0, end_dim=1).to(dtype))
            res = pred(pred_in)
            if last_mask is not None:
                if sigmoid_residual:
                    res = (torch.sigmoid(res) - 0.5) * 2
                logits = last_mask.to(res.dtype) + res
            else:
                logits = res
            if update_sensory:
                p1 = torch.cat([p1, logits.view(batch_size, actual_chunk_size, 1, *logits.shape[-2:]).to(p1.dtype)], 2)
                if fast_path:
                    new_sensory = self.sensory_update([p16, p8, p4, p2, p1], sensory)
                else:
                    new_sensory[:, i:i + chunk_size] = self.sensory_update(
                        [p16, p8, p4, p2, p1],
                        sensory[:, i:i + chunk_size],
                    )
            all_logits.append(logits)
        logits = torch.cat(all_logits, dim=0)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])
        return new_sensory, logits

    if hasattr(big_modules, "MaskDecoder"):
        big_modules.MaskDecoder.forward = _mask_decoder_forward

    def _uncert_pred_forward(self, last_frame_feat, cur_frame_feat, last_mask, mem_val_diff):
        dtype = _module_dtype(self.conv1x1_v2, last_frame_feat.dtype)
        last_mask = F.interpolate(last_mask.to(dtype), size=last_frame_feat.shape[-2:], mode="area")
        x = torch.cat([
            last_frame_feat.to(dtype),
            cur_frame_feat.to(dtype),
            last_mask,
            mem_val_diff.to(dtype),
        ], dim=1)
        x = self.conv1x1_v2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv3x3(x)
        x = self.bn2(x)
        x = self.relu(x)
        return self.conv3x3_out(x)

    if hasattr(big_modules, "UncertPred"):
        big_modules.UncertPred.forward = _uncert_pred_forward

    def _positional_encoding_forward(self, tensor):
        if len(tensor.shape) != 4 and len(tensor.shape) != 5:
            raise RuntimeError(f"The input tensor has to be 4/5d, got {tensor.shape}!")
        if len(tensor.shape) == 5:
            num_objects = tensor.shape[1]
            sample = tensor[:, 0]
        else:
            num_objects = None
            sample = tensor
        if self.channel_last:
            batch_size, h, w, _c = sample.shape
        else:
            batch_size, _c, h, w = sample.shape
        batch_size = int(batch_size)
        h = int(h)
        w = int(w)

        inv_freq = self.inv_freq.detach().float().cpu().numpy()
        pos_y = np.arange(h, dtype=np.float32)
        pos_x = np.arange(w, dtype=np.float32)
        if self.normalize:
            pos_y = pos_y / (pos_y[-1] + self.eps) * self.scale
            pos_x = pos_x / (pos_x[-1] + self.eps) * self.scale
        sin_inp_y = np.einsum("i,j->ij", pos_y, inv_freq)
        sin_inp_x = np.einsum("i,j->ij", pos_x, inv_freq)

        def emb(inp):
            stacked = np.stack([np.sin(inp), np.cos(inp)], axis=-1)
            return stacked.reshape(*inp.shape[:-1], -1)

        emb_y = emb(sin_inp_y)[:, None, :]
        emb_x = emb(sin_inp_x)
        out = np.zeros((h, w, self.dim * 2), dtype=np.float32)
        out[:, :, :self.dim] = emb_x[None, :, :]
        out[:, :, self.dim:] = emb_y
        if not self.channel_last and self.transpose_output:
            pass
        elif (not self.channel_last) or self.transpose_output:
            out = np.transpose(out, (2, 0, 1))
        penc = torch.tensor(out, device=tensor.device, dtype=tensor.dtype).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        if num_objects is None:
            return penc
        return penc.unsqueeze(1)

    if hasattr(positional_encoding, "PositionalEncoding"):
        positional_encoding.PositionalEncoding.forward = _positional_encoding_forward


def _shape_list(tensors: Iterable) -> list[list[int]]:
    return [list(t.shape) for t in tensors]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MatAnyone2 subgraphs to ONNX.")
    parser.add_argument("--repo-id", default="PeiqingYang/MatAnyone2", help="Hugging Face repo id for MatAnyone2")
    parser.add_argument("--out-dir", default="models/matanyone2_onnx", help="directory for ONNX outputs")
    parser.add_argument("--height", type=int, default=1024, help="fixed export input height; must be divisible by 16")
    parser.add_argument("--width", type=int, default=1024, help="fixed export input width; must be divisible by 16")
    parser.add_argument("--batch-size", type=int, default=1, help="fixed export batch size; use 2 for SBS left/right batching")
    parser.add_argument("--objects", type=int, default=1, help="number of target objects; use 1 for current passthrough")
    parser.add_argument("--memory-frames", type=int, default=1, help="dummy memory frames for propagate export")
    parser.add_argument("--device", default="cuda", help="torch device in export environment")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="export using half precision dummy inputs/model")
    parser.add_argument("--no-check", action="store_true", help="skip onnx.checker validation")
    args = parser.parse_args()

    if args.height % 16 or args.width % 16:
        raise SystemExit("--height and --width must be divisible by 16")
    if args.objects != 1:
        raise SystemExit("first integration supports --objects 1 only")

    torch = _import_torch()
    _patch_matanyone2_export_ops(torch)
    model = _load_model(args, torch)
    if args.fp16:
        model.half()

    class ImageKeyExport(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, image):
            ms_features, pix_feat = self.net.encode_image(image)
            key, shrinkage, selection = self.net.transform_key(ms_features[0])
            return (*ms_features, pix_feat, key, shrinkage, selection)

    class MaskMemoryExport(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, image, mask, sensory, pix_feat):
            return self.net.encode_mask(
                image,
                pix_feat,
                sensory,
                mask,
                deep_update=True,
                chunk_size=-1,
                need_weights=False,
            )[:3]

    class FirstFrameRefineExport(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, f16, f8, f4, f2, f1, pix_feat, last_msk_value, obj_memory, sensory, last_mask):
            pixel_readout = self.net.pixel_fusion(pix_feat, last_msk_value, sensory, last_mask)
            memory_readout, _ = self.net.readout_query(pixel_readout, obj_memory, selector=None, seg_pass=False)
            new_sensory, logits, prob = self.net.segment(
                [f16, f8, f4, f2, f1],
                memory_readout,
                sensory,
                update_sensory=True,
                last_mask=last_mask,
            )
            return prob, new_sensory, logits

    class PropagateExport(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(
            self,
            f16,
            f8,
            f4,
            f2,
            f1,
            pix_feat,
            key,
            selection,
            memory_key,
            memory_shrinkage,
            msk_value,
            obj_memory,
            sensory,
            last_mask,
            last_pix_feat,
            last_pred_mask,
            last_msk_value,
        ):
            from matanyone2.model.utils.memory_utils import get_affinity, readout
            from matanyone2.utils.device import safe_autocast

            batch_size, num_objects = msk_value.shape[:2]
            with safe_autocast(enabled=False):
                affinity = get_affinity(
                    memory_key.float(),
                    memory_shrinkage.float(),
                    key.float(),
                    selection.float(),
                    uncert_mask=None,
                )
            flat_msk_value = msk_value.flatten(start_dim=1, end_dim=2).float()
            pixel_readout = readout(affinity, flat_msk_value, None)
            pixel_readout = pixel_readout.view(batch_size, num_objects, self.net.value_dim, *pixel_readout.shape[-2:])
            last_flat_value = flat_msk_value[:, :, -1]
            last_value = last_flat_value.view(batch_size, num_objects, self.net.value_dim, *pixel_readout.shape[-2:])
            uncert = self.net.pred_uncertainty(last_pix_feat, pix_feat, last_pred_mask, pixel_readout[:, 0] - last_value[:, 0])
            uncert_prob = uncert["prob"].unsqueeze(1)
            pixel_readout = pixel_readout * uncert_prob + last_flat_value.unsqueeze(1) * (1 - uncert_prob)
            pixel_readout = self.net.pixel_fusion(pix_feat, pixel_readout, sensory, last_mask)
            memory_readout, _ = self.net.readout_query(pixel_readout, obj_memory, selector=None, seg_pass=False)
            new_sensory, logits, prob = self.net.segment(
                [f16, f8, f4, f2, f1],
                memory_readout,
                sensory,
                update_sensory=True,
                last_mask=last_mask,
            )
            uncert_prob = uncert["prob"] if isinstance(uncert, dict) and "prob" in uncert else logits[:, :1]
            return prob, new_sensory, logits, uncert_prob

    class PropagateUpdateExport(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.propagate = PropagateExport(net)
            self.net = net

        def forward(
            self,
            image,
            f16,
            f8,
            f4,
            f2,
            f1,
            pix_feat,
            key,
            selection,
            memory_key,
            memory_shrinkage,
            msk_value,
            obj_memory,
            sensory,
            last_mask,
            last_pix_feat,
            last_pred_mask,
            last_msk_value,
        ):
            prob, new_sensory, logits, uncert_prob = self.propagate(
                f16,
                f8,
                f4,
                f2,
                f1,
                pix_feat,
                key,
                selection,
                memory_key,
                memory_shrinkage,
                msk_value,
                obj_memory,
                sensory,
                last_mask,
                last_pix_feat,
                last_pred_mask,
                last_msk_value,
            )
            alpha = prob[:, 1:2].clamp(0, 1)
            new_msk_value, updated_sensory, updated_obj_memory = self.net.encode_mask(
                image,
                pix_feat,
                new_sensory,
                alpha,
                deep_update=True,
                chunk_size=-1,
                need_weights=False,
            )[:3]
            return prob, updated_sensory, new_msk_value, updated_obj_memory, logits, uncert_prob

    class StepUpdateExport(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.image_key = ImageKeyExport(net)
            self.propagate_update = PropagateUpdateExport(net)

        def forward(
            self,
            image,
            memory_key,
            memory_shrinkage,
            msk_value,
            obj_memory,
            sensory,
            last_mask,
            last_pix_feat,
            last_pred_mask,
            last_msk_value,
        ):
            f16, f8, f4, f2, f1, pix_feat, key, _shrinkage, selection = self.image_key(image)
            prob, new_sensory, new_msk_value, new_obj_memory, logits, uncert_prob = self.propagate_update(
                image,
                f16,
                f8,
                f4,
                f2,
                f1,
                pix_feat,
                key,
                selection,
                memory_key,
                memory_shrinkage,
                msk_value,
                obj_memory,
                sensory,
                last_mask,
                last_pix_feat,
                last_pred_mask,
                last_msk_value,
            )
            return prob, new_sensory, new_msk_value, new_obj_memory, pix_feat, logits, uncert_prob

    dtype = torch.float16 if args.fp16 else torch.float32
    batch_size = max(1, int(args.batch_size))
    image = torch.rand(batch_size, 3, args.height, args.width, device=args.device, dtype=dtype)

    out_dir = Path(args.out_dir).resolve()
    image_key_path = out_dir / "matanyone2_image_key.onnx"
    mask_memory_path = out_dir / "matanyone2_mask_memory.onnx"
    first_refine_path = out_dir / "matanyone2_first_frame_refine.onnx"
    propagate_path = out_dir / "matanyone2_propagate.onnx"
    propagate_update_path = out_dir / "matanyone2_propagate_update.onnx"
    step_update_path = out_dir / "matanyone2_step_update.onnx"

    image_key = ImageKeyExport(model).eval()
    with torch.inference_mode():
        f16, f8, f4, f2, f1, pix_feat, key, shrinkage, selection = image_key(image)
    _export(
        torch,
        image_key,
        (image,),
        image_key_path,
        ["image"],
        ["f16", "f8", "f4", "f2", "f1", "pix_feat", "key", "shrinkage", "selection"],
        args.opset,
    )

    mh, mw = key.shape[-2:]
    sensory_dim = int(model.cfg.model.sensory_dim)
    mask = torch.rand(batch_size, args.objects, args.height, args.width, device=args.device, dtype=dtype)
    sensory = torch.zeros(batch_size, args.objects, sensory_dim, mh, mw, device=args.device, dtype=dtype)
    mask_memory = MaskMemoryExport(model).eval()
    with torch.inference_mode():
        msk_value, new_sensory, obj_memory = mask_memory(image, mask, sensory, pix_feat)
    _export(
        torch,
        mask_memory,
        (image, mask, sensory, pix_feat),
        mask_memory_path,
        ["image", "mask", "sensory", "pix_feat"],
        ["msk_value", "new_sensory", "obj_memory"],
        args.opset,
    )

    first_refine = FirstFrameRefineExport(model).eval()
    obj_memory_t = obj_memory.unsqueeze(2)
    _export(
        torch,
        first_refine,
        (f16, f8, f4, f2, f1, pix_feat, msk_value, obj_memory_t, new_sensory, mask),
        first_refine_path,
        ["f16", "f8", "f4", "f2", "f1", "pix_feat", "last_msk_value", "obj_memory", "sensory", "last_mask"],
        ["prob", "new_sensory", "logits"],
        args.opset,
    )

    memory_t = max(1, int(args.memory_frames))
    memory_key = key.unsqueeze(2).repeat(1, 1, memory_t, 1, 1)
    memory_shrinkage = shrinkage.unsqueeze(2).repeat(1, 1, memory_t, 1, 1)
    msk_value_t = msk_value.unsqueeze(3).repeat(1, 1, 1, memory_t, 1, 1)
    obj_memory_t = obj_memory.unsqueeze(2).repeat(1, 1, memory_t, 1, 1)
    propagate = PropagateExport(model).eval()
    _export(
        torch,
        propagate,
        (
            f16,
            f8,
            f4,
            f2,
            f1,
            pix_feat,
            key,
            selection,
            memory_key,
            memory_shrinkage,
            msk_value_t,
            obj_memory_t,
            new_sensory,
            mask,
            pix_feat,
            mask,
            msk_value,
        ),
        propagate_path,
        [
            "f16",
            "f8",
            "f4",
            "f2",
            "f1",
            "pix_feat",
            "key",
            "selection",
            "memory_key",
            "memory_shrinkage",
            "msk_value",
            "obj_memory",
            "sensory",
            "last_mask",
            "last_pix_feat",
            "last_pred_mask",
            "last_msk_value",
        ],
        ["prob", "new_sensory", "logits", "uncert_prob"],
        args.opset,
    )

    propagate_update = PropagateUpdateExport(model).eval()
    _export(
        torch,
        propagate_update,
        (
            image,
            f16,
            f8,
            f4,
            f2,
            f1,
            pix_feat,
            key,
            selection,
            memory_key,
            memory_shrinkage,
            msk_value_t,
            obj_memory_t,
            new_sensory,
            mask,
            pix_feat,
            mask,
            msk_value,
        ),
        propagate_update_path,
        [
            "image",
            "f16",
            "f8",
            "f4",
            "f2",
            "f1",
            "pix_feat",
            "key",
            "selection",
            "memory_key",
            "memory_shrinkage",
            "msk_value",
            "obj_memory",
            "sensory",
            "last_mask",
            "last_pix_feat",
            "last_pred_mask",
            "last_msk_value",
        ],
        ["prob", "new_sensory", "new_msk_value", "new_obj_memory", "logits", "uncert_prob"],
        args.opset,
    )

    step_update = StepUpdateExport(model).eval()
    _export(
        torch,
        step_update,
        (
            image,
            memory_key,
            memory_shrinkage,
            msk_value_t,
            obj_memory_t,
            new_sensory,
            mask,
            pix_feat,
            mask,
            msk_value,
        ),
        step_update_path,
        [
            "image",
            "memory_key",
            "memory_shrinkage",
            "msk_value",
            "obj_memory",
            "sensory",
            "last_mask",
            "last_pix_feat",
            "last_pred_mask",
            "last_msk_value",
        ],
        ["prob", "new_sensory", "new_msk_value", "new_obj_memory", "pix_feat", "logits", "uncert_prob"],
        args.opset,
    )

    manifest = {
        "repo_id": args.repo_id,
        "height": args.height,
        "width": args.width,
        "batch_size": batch_size,
        "objects": args.objects,
        "memory_frames": memory_t,
        "fp16": bool(args.fp16),
        "opset": args.opset,
        "image_key_outputs": {
            "f16_f8_f4_f2_f1_pix_key_shrinkage_selection": _shape_list(
                [f16, f8, f4, f2, f1, pix_feat, key, shrinkage, selection]
            )
        },
        "mask_memory_outputs": {
            "msk_value_new_sensory_obj_memory": _shape_list([msk_value, new_sensory, obj_memory])
        },
        "files": {
            "image_key": str(image_key_path),
            "mask_memory": str(mask_memory_path),
            "first_frame_refine": str(first_refine_path),
            "propagate": str(propagate_path),
            "propagate_update": str(propagate_update_path),
            "step_update": str(step_update_path),
        },
    }
    if not args.no_check:
        manifest["onnx_check"] = {
            "image_key": _safe_check_onnx(image_key_path),
            "mask_memory": _safe_check_onnx(mask_memory_path),
            "first_frame_refine": _safe_check_onnx(first_refine_path),
            "propagate": _safe_check_onnx(propagate_path),
            "propagate_update": _safe_check_onnx(propagate_update_path),
            "step_update": _safe_check_onnx(step_update_path),
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
