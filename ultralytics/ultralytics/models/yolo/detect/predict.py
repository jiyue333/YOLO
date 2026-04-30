# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import LOGGER, nms, ops


class DetectionPredictor(BasePredictor):
    """A class extending the BasePredictor class for prediction based on a detection model.

    This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
    with bounding boxes and class predictions.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The detection model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        postprocess: Process raw model predictions into detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.
        get_obj_feats: Extract object features from the feature maps.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo26n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __call__(self, source=None, model=None, stream: bool = False, *args, **kwargs):
        """Run standard or optional SAHI sliced prediction."""
        if getattr(self.args, "sahi", False):
            try:
                results = self.sahi_inference(source=source, model=model)
                return results if stream else list(results)
            except ImportError:
                LOGGER.warning("SAHI sliced inference requested but package 'sahi' is not installed; falling back.")
        return super().__call__(source=source, model=model, stream=stream, *args, **kwargs)

    def sahi_inference(self, source=None, model=None):
        """Run SAHI sliced inference for image files with lazy optional imports.

        References:
            https://arxiv.org/abs/2202.06934
            https://github.com/obss/sahi
        """
        from pathlib import Path

        import cv2
        import torch
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        from ultralytics.data.utils import IMG_FORMATS

        source = source if source is not None else self.args.source
        if source is None:
            raise ValueError("SAHI inference requires an image file or directory source.")

        source_path = Path(source)
        if source_path.is_dir():
            image_paths = sorted(p for p in source_path.rglob("*.*") if p.suffix[1:].lower() in IMG_FORMATS)
        else:
            image_paths = [source_path]

        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(model or self.args.model),
            confidence_threshold=float(self.args.conf or 0.25),
            device=str(self.args.device or "cpu"),
        )
        mapping = getattr(detection_model, "category_mapping", {}) or {}
        names = {int(k): v for k, v in mapping.items()} if mapping else getattr(self.model, "names", {})

        for image_path in image_paths:
            sliced = get_sliced_prediction(
                str(image_path),
                detection_model,
                slice_height=int(self.args.sahi_slice_size),
                slice_width=int(self.args.sahi_slice_size),
                overlap_height_ratio=float(self.args.sahi_overlap),
                overlap_width_ratio=float(self.args.sahi_overlap),
                postprocess_type="NMS",
                postprocess_match_threshold=float(self.args.iou),
            )
            orig_img = cv2.imread(str(image_path))
            if orig_img is None:
                LOGGER.warning(f"Skipping unreadable SAHI source: {image_path}")
                continue
            rows = []
            for obj in sliced.object_prediction_list:
                x1, y1, x2, y2 = obj.bbox.to_xyxy()
                rows.append([x1, y1, x2, y2, float(obj.score.value), float(obj.category.id)])
            boxes = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 6), dtype=torch.float32)
            result = Results(orig_img, path=str(image_path), names=names, boxes=boxes)
            result.speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
            yield result

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo26n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            adaptive_nms=getattr(self.args, "adaptive_nms", False),
            adaptive_nms_max_iou=getattr(self.args, "adaptive_nms_max_iou", 0.75),
            adaptive_nms_density_threshold=getattr(self.args, "adaptive_nms_density_threshold", 3),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        """Extract object features from the feature maps."""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def construct_results(self, preds, img, orig_imgs):
        """Construct a list of Results objects from model predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
