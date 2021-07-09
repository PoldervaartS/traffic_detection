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
from __future__ import with_statement
from __future__ import absolute_import
from os import path
import pathlib
import random

import numpy as np

from .metalayer import (
    ConvolutionalLayer,
    MaxpoolLayer,
    NetLayer,
    RouteLayer,
    ShortcutLayer,
    UpsampleLayer,
    YoloLayer,
    YoloTpuLayer,
)
from io import open

def parse_cfg(cfg_path):
    u"""
    @return
        Dict[layer_name or layer_index, metalayer]
        Dict[layer_type, count]
        model_name
    """
    # metalayers: Dict[Union[str, int], Any] = {}
    metalayers = {}
    # count: Dict[str, int] = {
    count = {
        u"convolutional": 0,
        u"maxpool": 0,
        u"net": 0,
        u"route": 0,
        u"shortcut": 0,
        u"total": -1,
        u"upsample": 0,
        u"yolo": 0,
        u"yolo_tpu": 0,
    }
    layer_type = u"net"

    # meta_layer: Dict[str, Any] = {
    meta_layer = {
        u"convolutional": ConvolutionalLayer,
        u"maxpool": MaxpoolLayer,
        u"net": NetLayer,
        u"route": RouteLayer,
        u"shortcut": ShortcutLayer,
        u"upsample": UpsampleLayer,
        u"yolo": YoloLayer,
        u"yolo_tpu": YoloTpuLayer,
    }

    with open(cfg_path, u"r") as fd:
        layer = NetLayer(index=-1, type_index=-1)
        for line in fd:
            line = line.strip().split(u"#")[0]
            if line == u"":
                continue

            if line[0] == u"[":
                layer_type = line[1:-1]
                count[u"total"] += 1
                count[layer_type] += 1

                layer = meta_layer[layer_type](
                    index=count[u"total"] - 1, type_index=count[layer_type] - 1
                )
                metalayers[layer.name] = layer
                metalayers[count[u"total"] - 1] = layer

            else:
                # layer option
                option, value = line.split(u"=")
                option = option.strip()
                value = value.strip()
                try:
                    metalayers[layer.name][option] = value
                except KeyError, error:
                    print(
                        u"parse_cfg: [" + unicode(layer.name) + u" " + unicode(option) + u" is not" +
                        u" supported, it will not affect current computation."
                    )
                    pass

    # Build layer
    for index in xrange(count[u"total"]):
        layer = metalayers[index]

        output_shape = metalayers[index - 1].output_shape
        if layer.type in (u"route", u"shortcut"):
            if len(layer.layers) > 1:
                output_shape = [
                    metalayers[i].output_shape for i in layer.layers
                ]
            else:
                output_shape = metalayers[layer.layers[0]].output_shape
        layer[u"input_shape"] = output_shape

    model_name = pathlib.Path(cfg_path).stem

    return metalayers, count, model_name


# def parse_names(names_path: str) -> Dict[int, str]:
def parse_names(names_path):
    u"""
    @return {id: class name}
    """
    # names: Dict[int, str] = {}
    names = {}
    with open(names_path, u"r") as fd:
        index = 0
        for class_name in fd:
            class_name = class_name.strip()
            if len(class_name) != 0:
                names[index] = class_name
                index += 1

    return names