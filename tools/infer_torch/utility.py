# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import math
import paddle
from paddle import inference
import time
from ppocr.utils.logging import get_logger


def str2bool(v):
    return v.lower() in ("true", "t", "1")

def init_args(det_model_dir, rec_model_dir, cls_model_dir,
                image_dir = None,
                use_gpu = False, ir_optim= True, use_tensorrt = False, min_subgraph_size = 15, precision = 'fp32', gpu_mem = 500,
                det_algorithm = 'DB', det_limit_side_len = 960, det_limit_type = 'max',
                det_db_thresh = 0.3, det_db_box_thresh = 0.6, det_db_unclip_ratio = 1.5, max_batch_size = 10, use_dilation = False,
                    det_db_score_mode = 'fast',
                det_east_score_thresh = 0.8, det_east_cover_thresh = 0.1, det_east_nms_thresh = 0.2,
                det_sast_score_thresh = 0.5, det_sast_polygon = False, det_sast_nms_thresh = 0.2, 
                rec_algorithm = 'CRNN', rec_image_shape = '3, 32, 320', rec_char_type = 'ch', rec_batch_num = 6, max_text_length = 25, 
                    rec_char_dict_path = "./ppocr/utils/ppocr_keys_v1.txt", use_space_char = True, vis_font_path = "./doc/fonts/simfang.ttf", drop_score = 0.5,
                e2e_algorithm = 'PGNet', e2e_model_dir = None, e2e_limit_side_len = 768, e2e_limit_type = 'max',
                e2e_pgnet_score_thresh = 0.5, e2e_char_dict_path = "./ppocr/utils/ic15_dict.txt", e2e_pgnet_valid_set = 'totaltext',
                    e2e_pgnet_polygon = True, e2e_pgnet_mode = 'fast',
                use_angle_cls = False, cls_image_shape = "3, 48, 192", label_list = ['0', '180'], cls_batch_num = 6, cls_thresh = 0.9,
                enable_mkldnn = False, cpu_threads = 10, use_pdserving = False, warmup = True,
                use_mp = False, total_process_num = 1, process_id = 0, benchmark = False, save_log_path = "./log_output/", show_log = False):

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=use_gpu)
    parser.add_argument("--ir_optim", type=str2bool, default=ir_optim)
    parser.add_argument("--use_tensorrt", type=str2bool, default=use_tensorrt)
    parser.add_argument("--min_subgraph_size", type=int, default=min_subgraph_size)
    parser.add_argument("--precision", type=str, default=precision)
    parser.add_argument("--gpu_mem", type=int, default=gpu_mem)

    # params for text detector
    parser.add_argument("--image_dir", type=str, default=image_dir)
    parser.add_argument("--det_algorithm", type=str, default=det_algorithm)
    parser.add_argument("--det_model_dir", type=str, default=det_model_dir)
    parser.add_argument("--det_limit_side_len", type=float, default=det_limit_side_len)
    parser.add_argument("--det_limit_type", type=str, default=det_limit_type)

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=det_db_thresh)
    parser.add_argument("--det_db_box_thresh", type=float, default=det_db_box_thresh)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=det_db_unclip_ratio)
    parser.add_argument("--max_batch_size", type=int, default=max_batch_size)
    parser.add_argument("--use_dilation", type=str2bool, default=use_dilation)
    parser.add_argument("--det_db_score_mode", type=str, default=det_db_score_mode)
    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=det_east_score_thresh)
    parser.add_argument("--det_east_cover_thresh", type=float, default=det_east_cover_thresh)
    parser.add_argument("--det_east_nms_thresh", type=float, default=det_east_nms_thresh)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=det_sast_score_thresh)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=det_sast_nms_thresh)
    parser.add_argument("--det_sast_polygon", type=str2bool, default=det_sast_polygon)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default=rec_algorithm)
    parser.add_argument("--rec_model_dir", type=str, default=rec_model_dir)
    parser.add_argument("--rec_image_shape", type=str, default=rec_image_shape)
    parser.add_argument("--rec_char_type", type=str, default=rec_char_type)
    parser.add_argument("--rec_batch_num", type=int, default=rec_batch_num)
    parser.add_argument("--max_text_length", type=int, default=max_text_length)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default=rec_char_dict_path)
    parser.add_argument("--use_space_char", type=str2bool, default=use_space_char)
    parser.add_argument(
        "--vis_font_path", type=str, default=vis_font_path)
    parser.add_argument("--drop_score", type=float, default=drop_score)

    # params for e2e
    parser.add_argument("--e2e_algorithm", type=str, default=e2e_algorithm)
    parser.add_argument("--e2e_model_dir", type=str, default=e2e_model_dir)
    parser.add_argument("--e2e_limit_side_len", type=float, default=e2e_limit_side_len)
    parser.add_argument("--e2e_limit_type", type=str, default=e2e_limit_type)

    # PGNet parmas
    parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=e2e_pgnet_score_thresh)
    parser.add_argument(
        "--e2e_char_dict_path", type=str, default=e2e_char_dict_path)
    parser.add_argument("--e2e_pgnet_valid_set", type=str, default=e2e_pgnet_valid_set)
    parser.add_argument("--e2e_pgnet_polygon", type=str2bool, default=e2e_pgnet_polygon)
    parser.add_argument("--e2e_pgnet_mode", type=str, default=e2e_pgnet_mode)

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=use_angle_cls)
    parser.add_argument("--cls_model_dir", type=str, default=cls_model_dir)
    parser.add_argument("--cls_image_shape", type=str, default=cls_image_shape)
    parser.add_argument("--label_list", type=list, default=label_list)
    parser.add_argument("--cls_batch_num", type=int, default=cls_batch_num)
    parser.add_argument("--cls_thresh", type=float, default=cls_thresh)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=enable_mkldnn)
    parser.add_argument("--cpu_threads", type=int, default=cpu_threads)
    parser.add_argument("--use_pdserving", type=str2bool, default=use_pdserving)
    parser.add_argument("--warmup", type=str2bool, default=warmup)

    # multi-process
    parser.add_argument("--use_mp", type=str2bool, default=use_mp)
    parser.add_argument("--total_process_num", type=int, default=total_process_num)
    parser.add_argument("--process_id", type=int, default=process_id)

    parser.add_argument("--benchmark", type=str2bool, default=benchmark)
    parser.add_argument("--save_log_path", type=str, default=save_log_path)

    parser.add_argument("--show_log", type=str2bool, default=show_log)
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def create_predictor(args, mode, logger):
    if mode == "det":
        model_dir = args.det_model_dir
    elif mode == 'cls':
        model_dir = args.cls_model_dir
    elif mode == 'rec':
        model_dir = args.rec_model_dir
    elif mode == 'table':
        model_dir = args.table_model_dir
    else:
        model_dir = args.e2e_model_dir

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    model_file_path = model_dir + "/inference.pdmodel"
    params_file_path = model_dir + "/inference.pdiparams"
    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(model_file_path))
    if not os.path.exists(params_file_path):
        raise ValueError("not find params file path {}".format(
            params_file_path))

    config = inference.Config(model_file_path, params_file_path)

    if hasattr(args, 'precision'):
        if args.precision == "fp16" and args.use_tensorrt:
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32
    else:
        precision = inference.PrecisionType.Float32

    if args.use_gpu:
        gpu_id = get_infer_gpuid()
        if gpu_id is None:
            raise ValueError(
                "Not found GPU in current device. Please check your device or set args.use_gpu as False"
            )
        config.enable_use_gpu(args.gpu_mem, 0)
        if args.use_tensorrt:
            config.enable_tensorrt_engine(
                precision_mode=precision,
                max_batch_size=args.max_batch_size,
                min_subgraph_size=args.min_subgraph_size)
            # skip the minmum trt subgraph
        if mode == "det":
            min_input_shape = {
                "x": [1, 3, 50, 50],
                "conv2d_92.tmp_0": [1, 120, 20, 20],
                "conv2d_91.tmp_0": [1, 24, 10, 10],
                "conv2d_59.tmp_0": [1, 96, 20, 20],
                "nearest_interp_v2_1.tmp_0": [1, 256, 10, 10],
                "nearest_interp_v2_2.tmp_0": [1, 256, 20, 20],
                "conv2d_124.tmp_0": [1, 256, 20, 20],
                "nearest_interp_v2_3.tmp_0": [1, 64, 20, 20],
                "nearest_interp_v2_4.tmp_0": [1, 64, 20, 20],
                "nearest_interp_v2_5.tmp_0": [1, 64, 20, 20],
                "elementwise_add_7": [1, 56, 2, 2],
                "nearest_interp_v2_0.tmp_0": [1, 256, 2, 2]
            }
            max_input_shape = {
                "x": [1, 3, 1280, 1280],
                "conv2d_92.tmp_0": [1, 120, 400, 400],
                "conv2d_91.tmp_0": [1, 24, 200, 200],
                "conv2d_59.tmp_0": [1, 96, 400, 400],
                "nearest_interp_v2_1.tmp_0": [1, 256, 200, 200],
                "conv2d_124.tmp_0": [1, 256, 400, 400],
                "nearest_interp_v2_2.tmp_0": [1, 256, 400, 400],
                "nearest_interp_v2_3.tmp_0": [1, 64, 400, 400],
                "nearest_interp_v2_4.tmp_0": [1, 64, 400, 400],
                "nearest_interp_v2_5.tmp_0": [1, 64, 400, 400],
                "elementwise_add_7": [1, 56, 400, 400],
                "nearest_interp_v2_0.tmp_0": [1, 256, 400, 400]
            }
            opt_input_shape = {
                "x": [1, 3, 640, 640],
                "conv2d_92.tmp_0": [1, 120, 160, 160],
                "conv2d_91.tmp_0": [1, 24, 80, 80],
                "conv2d_59.tmp_0": [1, 96, 160, 160],
                "nearest_interp_v2_1.tmp_0": [1, 256, 80, 80],
                "nearest_interp_v2_2.tmp_0": [1, 256, 160, 160],
                "conv2d_124.tmp_0": [1, 256, 160, 160],
                "nearest_interp_v2_3.tmp_0": [1, 64, 160, 160],
                "nearest_interp_v2_4.tmp_0": [1, 64, 160, 160],
                "nearest_interp_v2_5.tmp_0": [1, 64, 160, 160],
                "elementwise_add_7": [1, 56, 40, 40],
                "nearest_interp_v2_0.tmp_0": [1, 256, 40, 40]
            }
            min_pact_shape = {
                "nearest_interp_v2_26.tmp_0": [1, 256, 20, 20],
                "nearest_interp_v2_27.tmp_0": [1, 64, 20, 20],
                "nearest_interp_v2_28.tmp_0": [1, 64, 20, 20],
                "nearest_interp_v2_29.tmp_0": [1, 64, 20, 20]
            }
            max_pact_shape = {
                "nearest_interp_v2_26.tmp_0": [1, 256, 400, 400],
                "nearest_interp_v2_27.tmp_0": [1, 64, 400, 400],
                "nearest_interp_v2_28.tmp_0": [1, 64, 400, 400],
                "nearest_interp_v2_29.tmp_0": [1, 64, 400, 400]
            }
            opt_pact_shape = {
                "nearest_interp_v2_26.tmp_0": [1, 256, 160, 160],
                "nearest_interp_v2_27.tmp_0": [1, 64, 160, 160],
                "nearest_interp_v2_28.tmp_0": [1, 64, 160, 160],
                "nearest_interp_v2_29.tmp_0": [1, 64, 160, 160]
            }
            min_input_shape.update(min_pact_shape)
            max_input_shape.update(max_pact_shape)
            opt_input_shape.update(opt_pact_shape)
        elif mode == "rec":
            min_input_shape = {"x": [1, 3, 32, 10]}
            max_input_shape = {"x": [args.rec_batch_num, 3, 32, 1024]}
            opt_input_shape = {"x": [args.rec_batch_num, 3, 32, 320]}
        elif mode == "cls":
            min_input_shape = {"x": [1, 3, 48, 10]}
            max_input_shape = {"x": [args.rec_batch_num, 3, 48, 1024]}
            opt_input_shape = {"x": [args.rec_batch_num, 3, 48, 320]}
        else:
            min_input_shape = {"x": [1, 3, 10, 10]}
            max_input_shape = {"x": [1, 3, 512, 512]}
            opt_input_shape = {"x": [1, 3, 256, 256]}
        config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                          opt_input_shape)

    else:
        config.disable_gpu()
        if hasattr(args, "cpu_threads"):
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        else:
            # default cpu threads as 10
            config.set_cpu_math_library_num_threads(10)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()

    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()

    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    if mode == 'table':
        config.delete_pass("fc_fuse_pass")  # not supported for table
    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors, config


def get_infer_gpuid():
    if not paddle.fluid.core.is_compiled_with_rocm():
        cmd = "nvidia-smi"
    else:
        cmd = "rocm-smi"
    res = os.popen(cmd).readlines()
    if len(res) == 0:
        return None
    if not paddle.fluid.core.is_compiled_with_rocm():
        cmd = "env | grep CUDA_VISIBLE_DEVICES"
    else:
        cmd = "env | grep HIP_VISIBLE_DEVICES"
    env_cuda = os.popen(cmd).readlines()
    if len(env_cuda) == 0:
        return 0
    else:
        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])


def draw_e2e_res(dt_boxes, strs, img_path):
    src_im = cv2.imread(img_path)
    for box, str in zip(dt_boxes, strs):
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        cv2.putText(
            src_im,
            str,
            org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.7,
            color=(0, 255, 0),
            thickness=1)
    return src_im


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./doc/fonts/simfang.ttf"):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


def base64_to_cv2(b64str):
    import base64
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


if __name__ == '__main__':
    pass
