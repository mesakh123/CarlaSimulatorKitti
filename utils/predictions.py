import requests
import cv2
import os
import numpy as np

import onnxruntime
import cv2

from .custom_classes import COCO_CLASSES, KITTI_CLASSES

session = None
EP_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def load_model():
    global session
    session = onnxruntime.InferenceSession(
        os.path.join(os.getcwd(), "models", "yolox_s.onnx"), providers=EP_list
    )


try:
    load_model()
except:
    pass


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1,
        )
        cv2.putText(
            img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1
        )

    return img


# origin_img : cv2 image
def predict(origin_img, conf=0.1, classes=None):
    global session
    if session is None:
        try:
            load_model()
        except:
            return None, origin_img
    input_shape = (640, 640)
    img, ratio = preprocess(origin_img, input_shape)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape, p6=False)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(
            origin_img.astype(np.int32),
            final_boxes,
            final_scores,
            final_cls_inds,
            conf=conf,
            class_names=classes,
        )
    return (final_boxes, final_scores, final_cls_inds), origin_img


def predict_remote(model_host, model_port, image, conf=0.6, classes=None):

    success, encoded_image = cv2.imencode(".png", image)
    result = None
    try:
        result = requests.post(
            "http://{}:{}/predict".format(model_host, model_port),
            data=encoded_image.tobytes(),
        )
        result = result.json()["inferences"]
    except:
        pass
    if result is not None:
        result_class = result["class"]
        indexes_from_coco = [int(COCO_CLASSES.index(v)) for v in result_class]
        image = vis(
            image.astype(np.int32),
            result["bbox"],
            result["prob"],
            indexes_from_coco,
            conf=conf,
            class_names=classes,
        )
        return (result["bbox"], result["prob"], indexes_from_coco), image
    return None, image


_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)
