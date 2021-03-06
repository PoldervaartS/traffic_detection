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

from __future__ import division
from __future__ import absolute_import
import numpy as np

from . import parser
from .metalayer import YoloLayer


class YOLOConfig(object):
    def __init__(self):
        # self._metalayers: Dict[Union[str, int], Any] = {}
        # self._layer_count: Dict[str, int]
        # self._model_name: str
        # self._names: Dict[int, str] = {}

        self._metalayers = {}
        self._layer_count = {}
        self._model_name = u""
        self._names = {}

    # def find_metalayer(self, layer_type: str, layer_index: int) -> Any:
    def find_metalayer(self, layer_type, layer_index):
        u"""
        Usage:
            last_yolo_layer = config.find_metalayer("yolo", -1)
        """
        if layer_index < 0:
            count = self._layer_count[layer_type]
            layer_index = count + layer_index
        # return self._metalayers["{layer_type}_{layer_index}"]
        return self._metalayers[unicode(layer_type) + u"_" + unicode(layer_index)]

    def summary(self):
        print self._metalayers[-1]
        print u"index layer No. filters  size/strd(dil)   input_shape         "
        print u"output_shape    1e9 flops"
        total_bflops = 0
        for i in xrange(self._layer_count[u"total"]):
            print self._metalayers[i]
            total_bflops += self._metalayers[i].bflops
        # print(f"Total B(1e9)flops: {total_bflops:6.3f}")
        print u"Total B(1e9)flops: " + unicode(total_bflops)

    # Parse ####################################################################

    # def parse_cfg(self, cfg_path: str):
    def parse_cfg(self, cfg_path):
        (
            self._metalayers,
            self._layer_count,
            self._model_name,
        ) = parser.parse_cfg(cfg_path=cfg_path)

        if self.layer_count[u"yolo"] > 0:
            self._metayolos = [
                self.find_metalayer(u"yolo", i)
                for i in xrange(self.layer_count[u"yolo"])
            ]
        elif self.layer_count[u"yolo_tpu"] > 0:
            self._metayolos = [
                self.find_metalayer(u"yolo_tpu", i)
                for i in xrange(self.layer_count[u"yolo_tpu"])
            ]

        if len(self._metayolos) > 0:
            self._masks = [yolo.mask for yolo in self._metayolos]

            self._anchors = self._metayolos[0].anchors / np.array(
                [self.net.width, self.net.height], np.float32
            )

    # def parse_names(self, names_path: str):
    def parse_names(self, names_path):
        self._names = parser.parse_names(names_path=names_path)

    # Property #################################################################

    @property
    def anchors(self): # -> np.ndarray:
        return self._anchors

    @property
    def layer_count(self): # -> Dict[str, int]:
        u"""
        key: metalayer.type
        value: the number of layers of the same type
        """
        return self._layer_count

    @property
    def masks(self): # -> List[np.ndarray]:
        return self._masks

    @property
    def metalayers(self): # -> Dict[Union[str, int], Any]:
        return self._metalayers

    @property
    def metayolos(self): # -> List[YoloLayer]:
        return self._metayolos

    @property
    def model_name(self): # -> str:
        return self._model_name

    @property
    def names(self): # -> Dict[int, str]:
        u"""
        class names
        """
        return self._names

    # Magic ####################################################################

    # def __getattr__(self, metalayer: str) -> Any:
    def __getattr__(self, metalayer):
        return self._metalayers[metalayer]
