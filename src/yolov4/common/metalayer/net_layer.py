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

from .base_layer import BaseLayer


class NetLayer(BaseLayer):
    def __init__(self, index, type_index):
        super(NetLayer, self).__init__(index=index, type_index=type_index, type_name=u"net")
        self._batch = 1
        self._burn_in = 0
        self._channels = 0
        self._height = 0
        self._learning_rate = 0.001
        self._max_batches = 0
        self._momentum = 0.9
        self._mosaic = False
        self._policy = u"steps"
        self._power = 4
        self._scales = ()
        self._steps = ()
        self._width = 0

    @property
    def batch(self):
        return self._batch

    @property
    def burn_in(self):
        return self._burn_in

    @property
    def channels(self):
        # override
        return self._channels

    @property
    def height(self):
        # override
        return self._height

    @property
    def input_shape(self):
        # override
        return (self._height, self._width, self._channels)

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def max_batches(self):
        return self._max_batches

    @property
    def momentum(self):
        return self._momentum

    @property
    def mosaic(self):
        return self._mosaic

    @property
    def name(self):
        # override
        return self._type_

    @property
    def output_shape(self):
        # override
        return (self._height, self._width, self._channels)

    @property
    def policy(self):
        return self._policy

    @property
    def power(self):
        return self._power

    @property
    def scales(self):
        return self._scales

    @property
    def steps(self):
        return self._steps

    @property
    def width(self):
        # override
        return self._width

    # def __repr__(self):
    #     rep = f"batch: {self._batch}"
    #     return rep

    def __setitem__(self, key, value):
        if key in (u"policy",):
            self.__setattr__("_"+str(key), unicode(value))
        elif key in (
            u"batch",
            u"burn_in",
            u"channels",
            u"height",
            u"max_batches",
            u"power",
            u"width",
        ):
            self.__setattr__("_"+str(key), int(value))
        elif key in (u"mosaic",):
            self.__setattr__("_"+str(key), bool(int(value)))
        elif key in (u"learning_rate", u"momentum"):
            self.__setattr__("_"+str(key), float(value))
        elif key in (u"steps",):
            self.__setattr__(
                "_"+str(key), tuple(int(i.strip()) for i in value.split(u","))
            )
        elif key in (u"scales",):
            self.__setattr__(
                "_"+str(key), tuple(float(i.strip()) for i in value.split(u","))
            )
        else:
            raise KeyError("_"+str(key) + " is not supported")
