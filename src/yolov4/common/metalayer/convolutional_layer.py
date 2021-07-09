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
from .base_layer import BaseLayer


class ConvolutionalLayer(BaseLayer):
    def __init__(self, index, type_index):
        super(ConvolutionalLayer, self).__init__(
            index=index, type_index=type_index, type_name=u"convolutional"
        )
        self._activation = u"logistic"
        self._batch_normalize = 0
        self._filters = 1
        self._pad = False
        self._padding = 0
        self._size = 1
        self._stride = 1

    @property
    def activation(self):
        return self._activation

    @property
    def batch_normalize(self):
        return self._batch_normalize

    @property
    def bflops(self):
        u"""
        kernel: size x size x input_filter x output_filter
            >>
                bias: output_filter
                or
                batch_normalization: output_filter x 4
        """
        return (
            2
            * self._size
            * self._size
            * self._input_shape[-1]
            * self._output_shape[-1]
            * self._output_shape[0]
            * self._output_shape[1]
        ) / 1e9

    @property
    def filters(self):
        return self._filters

    @property
    def pad(self):
        return self._pad

    @property
    def padding(self):
        if self._pad:
            return self._size // 2
        return self._padding

    @property
    def size(self):
        return self._size

    @property
    def stride(self):
        return self._stride

    # def __repr__(self):
    #     rep = f"{self.index:4}  "
    #     rep += f"{self.type[:5]}_"
    #     rep += f"{self.type_index:<3}  "
    #     rep += f"{self.filters:4}     "
    #     rep += f"{self.size:2} x{self.size:2} /{self.stride:2}     "
    #     rep += f"{self.input_shape[0]:4} "
    #     rep += f"x{self.input_shape[1]:4} "
    #     rep += f"x{self.input_shape[2]:4} -> "
    #     rep += f"{self.output_shape[0]:4} "
    #     rep += f"x{self.output_shape[1]:4} "
    #     rep += f"x{self.output_shape[2]:4}  "
    #     rep += f"{self.bflops:6.3f}"
    #     return rep

    def __setitem__(self, key, value):
        if key in (u"activation",):
            self.__setattr__("_"+str(key), unicode(value))
        elif key in (u"batch_normalize", u"filters", u"padding", u"size", u"stride"):
            self.__setattr__("_"+str(key), int(value))
        elif key in (u"pad",):
            self.__setattr__("_"+str(key), bool(int(value)))
        elif key == u"input_shape":
            self.__setattr__("_"+str(key), value)
            self._output_shape = (
                self._input_shape[0] // self._stride,
                self._input_shape[1] // self._stride,
                self._filters,
            )
        else:
            raise KeyError("_" + str(key) + " is not supported")
