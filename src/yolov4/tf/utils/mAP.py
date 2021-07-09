u"""
MIT License

Copyright (c) 2020-2021 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from os import makedirs, path
import shutil

import cv2
import numpy as np
from tqdm import tqdm
from io import open


def create_mAP_input_files(
    yolo,
    dataset,
    mAP_path,
    images_optional = False,
    num_sample = None,
):
    u"""
    Ref: https://github.com/Cartucho/mAP
    gt: name left top right bottom
    dr: name confidence left top right bottom

    @param `yolo`
    @param `dataset`
    @param `mAP_path`
    @param `images_optional`: If `True`, images are copied to the
            `mAP_path`.
    @param `num_sample`: Number of images for mAP. If `None`, all images in
            `data_set` are used.
    """
    input_path = path.join(mAP_path, u"input")

    if path.exists(input_path):
        shutil.rmtree(input_path)
    makedirs(input_path)

    gt_dir_path = path.join(input_path, u"ground-truth")
    dr_dir_path = path.join(input_path, u"detection-results")
    makedirs(gt_dir_path)
    makedirs(dr_dir_path)

    img_dir_path = u""
    if images_optional:
        img_dir_path = path.join(input_path, u"images-optional")
        makedirs(img_dir_path)

    max_dataset_size = len(dataset.dataset)

    if num_sample is None:
        num_sample = max_dataset_size

    if num_sample > max_dataset_size:
        num_sample = max_dataset_size

    for i in tqdm(xrange(num_sample)):
        # image_path, [[x, y, w, h, class_id], ...]
        image_path, gt_bboxes = dataset.dataset[i % max_dataset_size].copy()

        if images_optional:
            target_path = path.join(img_dir_path, u"image_{}.jpg".format(i))
            shutil.copy(image_path, target_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        gt_bboxes = gt_bboxes * np.array([width, height, width, height, 1])

        # ground-truth
        with open(
            path.join(gt_dir_path, u"image_{}.txt".format(i)),
            u"w",
        ) as fd:
            for xywhc in gt_bboxes:
                # name left top right bottom
                class_name = yolo.config.names[int(xywhc[4])].replace(u" ", u"_")
                left = int(xywhc[0] - xywhc[2] / 2)
                top = int(xywhc[1] - xywhc[3] / 2)
                right = int(xywhc[0] + xywhc[2] / 2)
                bottom = int(xywhc[1] + xywhc[3] / 2)
                fd.write(
                    u"{} {} {} {} {}\n".format(
                        class_name, left, top, right, bottom
                    )
                )

        # predict
        pred_bboxes = yolo.predict(image, prob_thresh=0.01)
        # Dim(-1, (x, y, w, h, cls_id, prob))
        pred_bboxes = pred_bboxes * np.array(
            [width, height, width, height, 1, 1]
        )

        # detection-results
        with open(
            path.join(dr_dir_path, u"image_{}.txt".format(i)),
            u"w",
        ) as fd:
            for xywhcp in pred_bboxes:
                # name confidence left top right bottom
                class_name = yolo.config.names[int(xywhcp[4])].replace(u" ", u"_")
                probability = xywhcp[5]
                if probability < 0.01:
                    continue
                left = int(xywhcp[0] - xywhcp[2] / 2)
                top = int(xywhcp[1] - xywhcp[3] / 2)
                right = int(xywhcp[0] + xywhcp[2] / 2)
                bottom = int(xywhcp[1] + xywhcp[3] / 2)
                fd.write(
                    u"{} {} {} {} {} {}\n".format(
                        class_name, probability, left, top, right, bottom
                    )
                )
