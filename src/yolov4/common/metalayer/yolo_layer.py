u"""
MIT License

Copyright (c) 2021 Hyeonki Hong <hhk7734@gmail.com>

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
from __future__ import absolute_import
import numpy as np


from .base_layer import BaseLayer


class YoloLayer(BaseLayer):
    def __init__(self, index, type_index):
        super(YoloLayer, self).__init__(index=index, type_index=type_index, type_name=u"yolo")
        self._anchors = np.ndarray([])
        self._beta_nms = None
        self._classes = 20
        self._cls_normalizer = 1.0
        self._iou_loss = u"iou"
        self._iou_thresh = 1.0
        self._iou_normalizer = 0.75
        self._label_smooth_eps = 0.0
        self._mask = np.ndarray([])
        self._max = 200
        self._new_coords = False
        self._nms_kind = u"greedynms"
        self._num = 1
        self._obj_normalizer = 1.0
        self._scale_x_y = 1.0

    @property
    def anchors(self):
        return self._anchors

    @property
    def beta_nms(self):
        if self._nms_kind == u"greedynms":
            return 0.6
        return self._beta_nms

    @property
    def bflops(self):
        return 0

    @property
    def classes(self):
        return self._classes

    @property
    def cls_normalizer(self):
        return self._cls_normalizer

    @property
    def iou_loss(self):
        return self._iou_loss

    @property
    def iou_thresh(self):
        u"""
        Recommended to use iou_thresh=0.213
        """
        return self._iou_thresh

    @property
    def iou_normalizer(self):
        return self._iou_normalizer

    @property
    def label_smooth_eps(self):
        return self._label_smooth_eps

    @property
    def mask(self):
        return self._mask

    @property
    def max(self):
        return self._max

    @property
    def new_coords(self):
        return self._new_coords

    @property
    def nms_kind(self):
        return self._nms_kind

    @property
    def obj_normalizer(self):
        return self._obj_normalizer

    @property
    def scale_x_y(self):
        return self._scale_x_y

    @property
    def total(self):
        return self._num

    # def __repr__(self):
    #     rep = f"{self.index:4}  "
    #     rep += f"{self.type[:5]:5s}_"
    #     rep += f"{self.type_index:<3}  "
    #     rep += f"iou_loss: {self._iou_loss}, "
    #     rep += f"iou_norm: {self._iou_normalizer}, "
    #     rep += f"obj_norm: {self._obj_normalizer}, "
    #     rep += f"cls_norm: {self._cls_normalizer}, "
    #     rep += u"\n                 "
    #     rep += f"scale_x_y: {self._scale_x_y}, "
    #     rep += f"new_coords: {self._new_coords}, "
    #     rep += f"NMS: {self._nms_kind}, "
    #     rep += f"beta_nms: {self._beta_nms}, "
    #     rep += u"\n                 "
    #     rep += f"iou_thresh: {self._iou_thresh}, "
    #     rep += f"label_smooth_eps: {self._label_smooth_eps}, "
    #     return rep

    def __setitem__(self, key, value):
        if key in (
            u"iou_loss",
            u"nms_kind",
        ):
            self.__setattr__("_"+str(key), unicode(value))
        elif key in (u"classes", u"max", u"num"):
            self.__setattr__("_"+str(key), int(value))
        elif key in (
            u"beta_nms",
            u"cls_normalizer",
            u"iou_thresh",
            u"iou_normalizer",
            u"label_smooth_eps",
            u"obj_normalizer",
            u"scale_x_y",
        ):
            self.__setattr__("_"+str(key), float(value))
        elif key in (u"new_coords",):
            self.__setattr__("_"+str(key), bool(int(value)))
        elif key in (u"mask",):
            self.__setattr__(
                "_"+str(key),
                np.array([int(i.strip()) for i in value.split(u",")], np.int32),
            )
        elif key == u"anchors":
            value = [int(i.strip()) for i in value.split(u",")]
            _value = []
            for i in xrange(len(value) // 2):
                _value.append((value[2 * i], value[2 * i + 1]))
            self.__setattr__("_"+str(key), np.array(_value, np.float32))
        elif key == u"input_shape":
            self.__setattr__("_"+str(key), value)
            self._output_shape = self._input_shape
        else:
            raise KeyError("_"+str(key) + " is not supported")
