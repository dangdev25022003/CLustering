import os
import sys
import yaml
import copy
import base64
import numpy as np
import cv2
import shutil
import torch
import onnxruntime as ort
from .utils import logger
from zipfile import ZipFile


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __deepcopy__(self, content):
        return copy.deepcopy(dict(self))


def create_predictor(config, mode):

    if mode == "embed":
        model_dir = '/media/dangph/data-1tb1/data_donghonuoc/clustering/test_sku_embedding/models/product/ali/'
    if mode == "det":
        model_dir = '../models/det/yolov8'

    if model_dir is None:
        print("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)

    model_file_path = model_dir + '/inference.onnx'
    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))

    providers = ['CUDAExecutionProvider','CPUExecutionProvider']

    sess = ort.InferenceSession(model_file_path, providers=providers)
    return sess, sess.get_inputs()[0], None, None


def create_attr_dict(yaml_config):
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


def parse_config(cfg_file):
    """Load a config file into AttrDict"""
    with open(cfg_file, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.SafeLoader))
    create_attr_dict(yaml_config)
    return yaml_config


def print_dict(d, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    placeholder = "-" * 60
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ",
                                         logger.coloring(k, "HEADER")))
            print_dict(v, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ",
                                         logger.coloring(str(k), "HEADER")))
            for value in v:
                print_dict(value, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ",
                                           logger.coloring(k, "HEADER"),
                                           logger.coloring(v, "OKGREEN")))
        if k.isupper():
            logger.info(placeholder)


def get_config(fname, overrides=None, show=True):
    """
    Read config from file
    """
    assert os.path.exists(fname), (
        'config file({}) is not exist'.format(fname))
    config = parse_config(fname)
    # if show:
    #     print_dict(config)
    return config


def unzip(zip_file, dist_path):
    with ZipFile(zip_file, 'r') as zip:
        zip.extractall(dist_path)


def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb'}
    return any([path.lower().endswith(e) for e in img_end])


def get_image_file_list(data_dir):
    img_lists = []
    img_name_lists = []
    product_name_lists = []
    for product_name in os.listdir(data_dir):
        product_dir = os.path.join(data_dir, product_name)
        for single_file in os.listdir(product_dir):
            file_path = os.path.join(product_dir, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                img_name_lists.append(single_file)
                img_lists.append(file_path)
                product_name_lists.append(product_name)
    return img_lists, img_name_lists, product_name_lists


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def get_iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1

    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1

    area_of_union = gt_height * gt_width + \
        pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou


def crop_image(det_results, ori_img, iou_threshold):
    h, w = ori_img.shape[:2]
    s_max = -1
    if len(det_results) > 0:
        for i in range(len(det_results)):
            if det_results[i]['bbox'] is not None:
                x1, y1, x2, y2 = det_results[i]['bbox']
                x1, y1, x2, y2 = max(0, int(x1)), max(
                    0, int(y1)), min(w, int(x2)), min(h, int(y2))
                s = abs((x2-x1)*(y2-y1))
                prediction_bbox = np.array([0, 0, w, h], dtype=np.float32)
                iou = get_iou(det_results[i]['bbox'], prediction_bbox)
                if s > s_max and iou > iou_threshold:
                    s_max = s
                    x1_choose, y1_choose, x2_choose, y2_choose = x1, y1, x2, y2
                else:
                    x1_choose, y1_choose, x2_choose, y2_choose = 0, 0, w, h
            else:
                x1, y1, x2, y2 = det_results[i]['bbox']
                x1_choose, y1_choose, x2_choose, y2_choose = max(
                    0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        roi_img = ori_img[y1_choose:y2_choose, x1_choose:x2_choose, :]
        return roi_img
    else:
        return ori_img


def crop_image_1(det_results, ori_img, iou_threshold):
    h, w = ori_img.shape[:2]
    s_max = -1
    check = False
    if len(det_results) > 0:
        for i in range(len(det_results)):
            if det_results[i]['bbox'] is not None:
                x1, y1, x2, y2 = det_results[i]['bbox']
                x1, y1, x2, y2 = max(0, int(x1)), max(
                    0, int(y1)), min(w, int(x2)), min(h, int(y2))
                s = abs((x2-x1)*(y2-y1))
                prediction_bbox = np.array([0, 0, w, h], dtype=np.float32)
                iou = get_iou(det_results[i]['bbox'], prediction_bbox)
                if s > s_max and iou > iou_threshold:
                    check = True
                    s_max = s
                    x1_choose, y1_choose, x2_choose, y2_choose = x1, y1, x2, y2
                else:
                    continue
            else:
                x1, y1, x2, y2 = det_results[i]['bbox']
                x1_choose, y1_choose, x2_choose, y2_choose = max(
                    0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        if not check:
            x1_choose, y1_choose, x2_choose, y2_choose = 0, 0, w, h
        roi_img = ori_img[y1_choose:y2_choose, x1_choose:x2_choose, :]
    else:
        roi_img = ori_img
        x1_choose, y1_choose, x2_choose, y2_choose = 0, 0, w, h
    return roi_img, [x1_choose, y1_choose, x2_choose, y2_choose]


def get_locate_crop_image(det_results, width, height, iou_threshold):
    s_max = -1
    if len(det_results) > 0:
        for i in range(len(det_results)):
            if det_results[i]['bbox'] is not None:
                x1, y1, x2, y2 = det_results[i]['bbox']
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(
                    width, int(x2)), min(height, int(y2))
                s = abs((x2-x1)*(y2-y1))
                prediction_bbox = np.array(
                    [0, 0, width, height], dtype=np.float32)
                iou = get_iou(det_results[i]['bbox'], prediction_bbox)
                if s > s_max and iou > iou_threshold:
                    s_max = s
                    x1_choose, y1_choose, x2_choose, y2_choose = x1, y1, x2, y2
                else:
                    x1_choose, y1_choose, x2_choose, y2_choose = 0, 0, width, height
            else:
                x1, y1, x2, y2 = det_results[i]['bbox']
                x1_choose, y1_choose, x2_choose, y2_choose = max(0, int(x1)), max(
                    0, int(y1)), min(width, int(x2)), min(height, int(y2))
        return (x1_choose, y1_choose, x2_choose, y2_choose)
    else:
        return (0, 0, width, height)

def padding(image):
    color = (255, 255, 255)  

    # Lấy kích thước của ảnh
    height, width = image.shape[:2]
    pad_height = int(height*2)
    pad_width = int(width*2)
    if pad_height%2!=0:
        pad_height += 1
    if pad_width%2!=0:
        pad_width += 1
    half_pad_height = int(pad_height / 2)
    half_pad_width= int(pad_width / 2)
    # Tạo một ảnh mới có kích thước lớn hơn
    new_height = height + pad_height # 4 phía mỗi phía
    new_width = width + pad_width
    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Thiết lập màu cho ảnh mới
    new_image[:, :] = color

    # Chèn ảnh gốc vào ảnh mới
    new_image[half_pad_height:height+half_pad_height, half_pad_width:width+half_pad_width] = image
    
    return new_image


def create_path(path_dir):
    if not os.path.exists(path_dir):
        try:
            os.mkdir(path_dir)
        except OSError as error:
            print(error)


def zip_folder(folder_path, zip_filename):
    try:
        # Create a zip file with the specified name
        shutil.make_archive(zip_filename, 'zip', folder_path)
    except Exception as e:
        raise TypeError(f'Error: {e}')


def delete_folder(folder_path):
    try:
        # Delete the folder and its contents
        shutil.rmtree(folder_path)
    except Exception as e:
        raise TypeError(f'Error: {e}')
