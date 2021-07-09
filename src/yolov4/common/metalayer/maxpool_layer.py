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


class MaxpoolLayer(BaseLayer):
    def __init__(self, index, type_index):
        super(MaxpoolLayer, self).__init__(
            index=index, type_index=type_index, type_name=u"maxpool"
        )
        self._size = 1
        self._stride = 1

    @property
    def bflops(self): # -> float:
        return (
            self.size
            * self.size
            * self.input_shape[-1]
            * self.output_shape[0]
            * self.output_shape[0]
        ) / 1e9

    @property
    def size(self): # -> int:
        return self._size

    @property
    def stride(self): # -> int:
        return self._stride

    # def __repr__(self): # -> str:
    #     rep = f"{self.index:4}  "
    #     rep += f"{self.type[:5]}_"
    #     rep += f"{self.type_index:<3}           "
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
        if key in (u"size", u"stride"):
            self.__setattr__("_"+str(key), int(value))
        elif key == u"input_shape":
            self.__setattr__("_"+str(key), value)
            self._output_shape = (
                self._input_shape[0] // self._stride,
                self._input_shape[1] // self._stride,
                self._input_shape[2],
            )
        else:
            raise KeyError("_"+str(key) + " is not supported")
