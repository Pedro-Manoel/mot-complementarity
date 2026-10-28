"""The three detection backends (ultralytics, D-FINE, torchvision Faster R-CNN), each with its native preprocessing.

Only the common parameters are set; everything else stays at the library default and weights are read from local
files only. `detect` returns the person boxes `[x1, y1, x2, y2, score, 0]` and the full model output in `info["raw"]`.

Usage: `model = load_model(detector, paths, device, checkpoint=variant['checkpoint'])`, then `bind(model, variant)`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..provenance import CheckpointMismatch, sha256_bytes, verify_checkpoint

# person label in torchvision's COCO91; D-FINE and ultralytics use COCO80 with person 0
_PERSON_COCO91 = 1


class _Base:
    def __init__(self, name: str, backend: str, device: str, precision: str, versions: dict):
        self.name, self.backend, self.device, self.precision, self.versions = name, backend, device, precision, versions
        self._input_hw = None
        self.factory_settings: dict = {}

    def describe_model(self) -> dict:
        return {"backend": self.backend, "detector": self.name, "device": str(self.device),
                "precision": self.precision, "versions": dict(self.versions),
                "factory_settings": dict(self.factory_settings),
                "checkpoint_sha256_loaded": getattr(self, "checkpoint_sha256", None),
                "label_space": dict(getattr(self, "label_space", {}) or {})}


def _raw_output(bboxes, scores, labels) -> np.ndarray:
    """`[x1, y1, x2, y2, score, native label]` for every box, in the order the model returned them."""
    if not len(bboxes):
        return np.empty((0, 6), np.float32)
    return np.column_stack([np.asarray(bboxes, np.float32), np.asarray(scores, np.float32),
                            np.asarray(labels, np.float32)]).astype(np.float32)


def _versions(*packages) -> dict:
    import importlib.metadata as md
    out = {}
    for p in packages:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


_BACKEND_PACKAGES = {"ultralytics": ("torch", "ultralytics"), "dfine": ("torch", "transformers", "torchvision"),
                     "torchvision": ("torch", "torchvision")}
# Bumped when an adapter changes its output with the same library versions.
_ADAPTER_CONTRACT = {"ultralytics": "3", "dfine": "4", "torchvision": "4"}


def backend_versions(backend: str) -> dict:
    """Versions that determine a backend's output, from package metadata; they enter the detection cell identity."""
    return {**_versions(*_BACKEND_PACKAGES[backend]), "adapter_contract": _ADAPTER_CONTRACT[backend]}


def _require_fp32(precision: str) -> None:
    if precision != "fp32":
        raise ValueError(f"precision {precision!r} is not supported by this version of the module; "
                         "the protocol uses fp32")


class _Ultralytics(_Base):
    def __init__(self, detector: dict, paths: dict, device: str, precision: str, checkpoint: dict):
        _require_fp32(precision)
        target = verify_checkpoint(detector, checkpoint, paths)
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        super().__init__(detector["name"], "ultralytics", device, precision, _versions("torch", "ultralytics"))
        self.checkpoint_sha256 = target["sha256"]
        self.model = YOLO(str(target["path"]))
        class_names = dict(self.model.model.names)
        if class_names.get(0) != "person":
            raise ValueError(f"{detector['name']}: the checkpoint does not have person at id 0 "
                             f"({class_names.get(0)!r})")
        self.label_space = {"name": "coco80 (ultralytics names)", "person_label": 0,
                            "labels": {str(k): str(v) for k, v in sorted(class_names.items())}}
        self.head = self.model.model.model[-1]
        # read before the first predict, so these are the checkpoint's own values
        self.factory_settings = {"end2end": bool(getattr(self.head, "end2end", False)), "iou": float(get_cfg().iou),
                                 "stride": int(self.model.model.stride.max()), "n_classes": len(class_names)}
        self.model.to(device)
        self.model.model.register_forward_pre_hook(self._capture)

    def _capture(self, _module, inputs):
        x = inputs[0]
        self._input_hw = (int(x.shape[-2]), int(x.shape[-1]))

    def detect(self, frame, v: dict):
        det = v["detection"]
        cap = int(det["max_det"])
        self._input_hw = None
        # no iou (end2end has no NMS) and no classes: persons are selected afterwards, so max_det counts all classes
        res = self.model.predict(frame, conf=float(det["conf_floor"]), imgsz=int(v["resolution"]["requested"]),
                                 max_det=cap, verbose=False, half=False, device=self.device)
        bboxes = res[0].boxes
        total = 0 if bboxes is None else len(bboxes)
        if not total:
            dets, raw = np.empty((0, 6), np.float32), np.empty((0, 6), np.float32)
        else:
            xyxy, conf, cls = bboxes.xyxy.cpu().numpy(), bboxes.conf.cpu().numpy(), bboxes.cls.cpu().numpy()
            raw = _raw_output(xyxy, conf, cls)
            person = cls == 0
            dets = np.hstack([xyxy[person], conf[person][:, None], np.zeros((int(person.sum()), 1))]).astype(np.float32)
        return dets, {"input_hw": self._input_hw, "resized_hw": None, "n_before_nms": None, "n_after_nms": len(dets),
                      "saturated": total >= cap, "raw": raw}

    def settings(self) -> dict:
        a = self.model.predictor.args
        return {"conf": float(a.conf), "max_det": int(a.max_det), "classes": a.classes, "iou_passed": False,
                "end2end": bool(getattr(self.head, "end2end", False)), "input_hw": self._input_hw}


class _DFine(_Base):
    def __init__(self, detector: dict, paths: dict, device: str, precision: str, checkpoint: dict):
        _require_fp32(precision)
        # the revision recorded in the plan, not the current refs/main
        target = verify_checkpoint(detector, checkpoint, paths)
        import inspect

        import torch
        from transformers import AutoImageProcessor, DFineForObjectDetection
        super().__init__(detector["name"], "dfine", device, precision,
                         _versions("torch", "transformers", "torchvision"))
        repo = detector["checkpoint"]["hf_repo"]
        self.checkpoint_sha256, self.revision = target["sha256"], target["revision"]
        kw = {"local_files_only": True, "cache_dir": str(paths["hf_cache_dir"]), "revision": target["revision"]}
        self.processor = AutoImageProcessor.from_pretrained(repo, **kw)
        self.model = DFineForObjectDetection.from_pretrained(repo, **kw).to(torch.device(device)).eval()
        id2label = {int(k): str(v) for k, v in dict(self.model.config.id2label).items()}
        person_ids = [i for i, r in id2label.items() if r.lower() == "person"]
        if len(id2label) != 80 or person_ids != [0]:
            raise ValueError(f"{detector['name']}: the checkpoint does not use COCO80 with person at id 0 "
                             f"({len(id2label)} labels, person at {person_ids})")
        self.label_space = {"name": "coco80 (config.id2label)", "person_label": 0,
                            "labels": {str(k): v for k, v in sorted(id2label.items())}}
        self.n_queries = int(getattr(self.model.config, "num_queries", 0))
        proc_size = self.processor.size
        self.factory_settings = {
            "num_queries": self.n_queries, "feat_strides": list(getattr(self.model.config, "feat_strides", []) or []),
            "processor_size": {"height": getattr(proc_size, "height", None),
                               "width": getattr(proc_size, "width", None)},
            "post_process_threshold":
                inspect.signature(self.processor.post_process_object_detection).parameters["threshold"].default}
        self._last: dict = {}

    def detect(self, frame, v: dict):
        import torch
        det, side = v["detection"], int(v["resolution"]["requested"])
        if int(det["max_det"]) != self.n_queries:
            raise ValueError(f"{self.name}: the common cap {det['max_det']} differs from the {self.n_queries} fixed "
                             "D-FINE queries; the plan rejects this catalog")
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        inputs = self.processor(images=rgb, size={"height": side, "width": side},
                                return_tensors="pt").to(self.model.device)
        in_shape = inputs["pixel_values"].shape
        with torch.inference_mode():
            outputs = self.model(**inputs)
        # threshold 0 keeps the top 300 (query x class) candidates over all classes, unclipped and without NMS; if the
        # lowest of them is above the floor, admissible boxes were cut and the frame counts as saturated
        pos = self.processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=[(h, w)])[0]
        all_scores = pos["scores"].detach().cpu().numpy().astype(np.float32)
        bboxes = pos["boxes"].detach().cpu().numpy().astype(np.float32)
        labels = pos["labels"].detach().cpu().numpy()
        cut = float(all_scores.min()) if len(all_scores) else None
        raw = _raw_output(bboxes, all_scores, labels)
        keep = (all_scores >= float(det["conf_floor"])) & (labels == 0)
        bboxes, scores = bboxes[keep], all_scores[keep]
        dets = np.column_stack([bboxes, scores, np.zeros(len(bboxes), np.float32)]).astype(np.float32) \
            if len(bboxes) else np.empty((0, 6), np.float32)
        full = len(all_scores) >= (self.n_queries or 10 ** 9)
        self._last = {"size": [side, side], "threshold": 0.0, "conf_floor": float(det["conf_floor"]),
                      "n_candidates": int(len(all_scores))}
        return dets, {"input_hw": (int(in_shape[-2]), int(in_shape[-1])), "resized_hw": None, "n_before_nms": None,
                      "n_after_nms": len(dets), "candidate_cut_score": cut,
                      "saturated": bool(full and cut is not None and cut > float(det["conf_floor"])), "raw": raw}

    def settings(self) -> dict:
        return {**self._last, "num_queries": self.n_queries}


class _Torchvision(_Base):
    def __init__(self, detector: dict, paths: dict, device: str, precision: str, checkpoint: dict):
        _require_fp32(precision)
        target = verify_checkpoint(detector, checkpoint, paths)
        import io

        import torch
        from torchvision.models import detection as tvd
        super().__init__(detector["name"], "torchvision", device, precision, _versions("torch", "torchvision"))
        enum = detector["checkpoint"]["torchvision_weights"]
        if not enum.startswith("FasterRCNN_ResNet50_FPN_V2_Weights"):
            raise ValueError(f"{detector['name']}: only Faster R-CNN ResNet50-FPN v2 is supported here, got {enum}")
        self.model = tvd.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=91)
        categories = list(getattr(tvd.FasterRCNN_ResNet50_FPN_V2_Weights, enum.split(".")[-1]).meta["categories"])
        if len(categories) != 91 or categories[_PERSON_COCO91] != "person":
            raise ValueError(f"{detector['name']}: the weights do not use COCO91 with person at id {_PERSON_COCO91}")
        self.label_space = {"name": "coco91 (torchvision weights.meta categories)", "person_label": _PERSON_COCO91,
                            "labels": {str(i): c for i, c in enumerate(categories)}}
        rh, tr, rpn = self.model.roi_heads, self.model.transform, self.model.rpn
        self.factory_settings = {"nms_thresh": float(rh.nms_thresh), "score_thresh": float(rh.score_thresh),
                                 "detections_per_img": int(rh.detections_per_img), "min_size": list(tr.min_size),
                                 "max_size": int(tr.max_size), "size_divisible": int(tr.size_divisible),
                                 "rpn_pre_nms_top_n_test": int(rpn._pre_nms_top_n["testing"]),
                                 "rpn_post_nms_top_n_test": int(rpn._post_nms_top_n["testing"])}
        # hash and load the same bytes
        data = target["path"].read_bytes()
        if sha256_bytes(data) != checkpoint["sha256"]:
            raise CheckpointMismatch(f"checkpoint {detector['name']}: the file changed while loading")
        self.checkpoint_sha256 = checkpoint["sha256"]
        state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(torch.device(device)).eval()
        self.model.transform.register_forward_hook(self._capture)
        self._resized = None

    def _capture(self, _module, _inputs, output):
        image_list = output[0]
        self._input_hw = (int(image_list.tensors.shape[-2]), int(image_list.tensors.shape[-1]))
        self._resized = tuple(int(x) for x in image_list.image_sizes[0])

    def detect(self, frame, v: dict):
        import cv2
        import torch
        from torchvision.transforms import functional as F
        det, side = v["detection"], int(v["resolution"]["requested"])
        # min_size = max_size = r makes r the longer side; NMS stays at the default
        tr, rh = self.model.transform, self.model.roi_heads
        tr.min_size, tr.max_size = (side,), side
        rh.score_thresh, rh.detections_per_img = float(det["conf_floor"]), int(det["max_det"])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb).to(next(self.model.parameters()).device)
        with torch.inference_mode():
            pred = self.model([tensor])[0]
        bboxes = pred["boxes"].cpu().numpy().astype(np.float32)
        scores = pred["scores"].cpu().numpy().astype(np.float32)
        labels = pred["labels"].cpu().numpy()
        output = _raw_output(bboxes, scores, labels)             # all classes, after NMS
        raw = len(bboxes)
        keep = (scores >= float(det["conf_floor"])) & (labels == _PERSON_COCO91)
        bboxes, scores = bboxes[keep], scores[keep]
        dets = np.column_stack([bboxes, scores, np.zeros(len(bboxes), np.float32)]).astype(np.float32) \
            if len(bboxes) else np.empty((0, 6), np.float32)
        # n_before_nms is unknown: NMS runs inside the model
        return dets, {"input_hw": self._input_hw, "resized_hw": self._resized, "n_before_nms": None,
                      "n_after_nms": len(dets), "saturated": raw >= int(det["max_det"]), "raw": output}

    def settings(self) -> dict:
        tr, rh = self.model.transform, self.model.roi_heads
        return {"min_size": list(tr.min_size), "max_size": int(tr.max_size), "score_thresh": float(rh.score_thresh),
                "detections_per_img": int(rh.detections_per_img), "nms_thresh": float(rh.nms_thresh),
                "input_hw": self._input_hw, "resized_hw": self._resized}


_CLASSES = {"ultralytics": _Ultralytics, "dfine": _DFine, "torchvision": _Torchvision}


def load_model(detector: dict, paths: dict, device: str, precision: str = "fp32", checkpoint: dict | None = None):
    """`checkpoint` is the one resolved by the plan (`variant["checkpoint"]`); the weights are verified against it."""
    try:
        cls = _CLASSES[detector["backend"]]
    except KeyError:
        raise ValueError(f"unknown backend: {detector['backend']!r}") from None
    return cls(detector, paths, device, precision, checkpoint)


class BoundBackend:
    """A loaded model bound to a variant (resolution and detection block)."""

    def __init__(self, loaded_model, variant: dict):
        if variant["detector"]["name"] != loaded_model.name:
            raise ValueError(f"variant {variant['detector_variant_id']} does not belong to detector "
                             f"{loaded_model.name}")
        self.loaded_model, self.variant = loaded_model, variant

    def describe(self) -> dict:
        return self.loaded_model.describe_model()

    def detect(self, frame_bgr):
        return self.loaded_model.detect(frame_bgr, self.variant)

    def settings(self) -> dict:
        return self.loaded_model.settings()


def bind(loaded_model, variant: dict) -> BoundBackend:
    return BoundBackend(loaded_model, variant)
