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


class ShortcutLayer(BaseLayer):
    def __init__(self, index, type_index):
        super(ShortcutLayer, self).__init__(
            index=index, type_index=type_index, type_name=u"shortcut"
        )
        self._activation = u"linear"
        self._from = ()

    @property
    def activation(self):
        return self._activation

    @property
    def bflops(self):
        return (
            self.output_shape[0]
            * self.output_shape[1]
            * self.output_shape[2]
            * len(self._from)
        ) / 1e9

    @property
    def layers(self):
        # 'from' is python keyword.
        out = list(self._from)
        out.append(self._index_ - 1)
        return tuple(out)

    # def __repr__(self):
    #     rep = f"{self.index:4}  "
    #     rep += f"{self.type[:5]}_"
    #     rep += f"{self.type_index:<3}   "
    #     for layer in self.layers:
    #         rep += f"{layer:3},"
    #     rep += u" " * 4 * (6 - len(self.layers))
    #     rep += u"                -> "
    #     rep += f"{self.output_shape[0]:4} "
    #     rep += f"x{self.output_shape[1]:4} "
    #     rep += f"x{self.output_shape[2]:4}  "
    #     rep += f"{self.bflops:6.3f}"
    #     return rep

    def __setitem__(self, key, value):
        if key in (u"activation",):
            self.__setattr__("_"+str(key), unicode(value))
        elif key in (u"from",):
            self.__setattr__(
                "_"+str(key),
                tuple(
                    int(i) if int(i) >= 0 else self._index_ + int(i)
                    for i in value.split(u",")
                ),
            )
        elif key == u"input_shape":
            self.__setattr__("_"+str(key), value)
            self._output_shape = self._input_shape[0]
        else:
            raise KeyError("_"+str(key) + " is not supported")
