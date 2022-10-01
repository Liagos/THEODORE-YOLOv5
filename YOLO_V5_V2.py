import torch
import warnings
import torch.nn as nn
from torchsummary import summary

config = [(-1, 64, 6, 2, 2),
          (-1, 128, 3, 2, 1),
          ["C3", 3, True],
          (-1, 256, 3, 2, 1),
          ["C3", 6, True],
          (-1, 512, 3, 2, 1),
          ["C3", 9, True],
          (-1, 1024, 3, 2, 1),
          ["C3", 3, True],
          "SPPF",
          (-1, 512, 1, 1, 0),
          "U",
          ["C3", 3, False],
          (1, 256, 1, 1, 0),
          "U",
          ["C3", 3, False],
          "S",
          (2, 256, 3, 2, 1),
          ["C3", 3, False],
          "S",
          (3, 512, 3, 2, 1),
          ["C3", 3, False],
          "S"
          ]


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes, anchors_per_scale=3):
        super().__init__()
        self.pred = nn.Sequential(Conv(in_channels, in_channels * 2, k=1, s=1),
                                  Conv(in_channels * 2, (num_classes + 5) * 3, k=1, act=False)
                                  )
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale

    def forward(self, x):
        return (self.pred(x).reshape(x.shape[0], self.anchors_per_scale, self.num_classes + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    # act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, route=1):
        super().__init__()

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.route = route
        self.c2 = c2
        self.k = k
        self.s = s

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.n = n

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.num_repeats = n
        self.shortcut = shortcut
        self.c2 = c2
        self.c1 = c1
        # self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class YOLOV5(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(YOLOV5, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self.CreateConvLayers()
        self.false = []

    def CreateConvLayers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        false_list = []

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride, padding = module[1:]
                route = module[0]
                layers.append(Conv(c1=in_channels,
                                   c2=out_channels,
                                   k=kernel_size,
                                   s=stride,
                                   p=padding,
                                   route=route))
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                shortcut = module[-1]
                if not shortcut:
                    false_list.append(shortcut)
                    if len(false_list) == 3:
                        in_channels = in_channels*2
                        out_channels = out_channels*2
                    elif len(false_list) == 4:
                        in_channels = in_channels*2
                        out_channels = out_channels*2
                layers.append(C3(c1=in_channels,
                                c2=out_channels,
                                n=num_repeats,
                                shortcut=shortcut))
                in_channels = out_channels
            elif isinstance(module, str):
                if module == "SPPF":
                    layers.append(SPPF(c1=in_channels,
                                       c2=out_channels,
                                       ))
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 2
                elif module == "S":
                    layers.append(ScalePrediction(in_channels, num_classes=self.num_classes))

        return layers

    def forward(self, x):
        outputs = []
        route_connections = []
        conv_routes = []
        sppf_out = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, C3) and (layer.num_repeats == 6 or layer.num_repeats == 9):
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
            elif isinstance(layer, Conv) and (layer.c2 == 512 and layer.k == 1 and layer.s == 1):
                conv_routes.append(x)
            elif isinstance(layer, Conv) and (layer.c2 == 256 and layer.k == 1 and layer.s == 1):
                conv_routes.append(x)
            elif isinstance(layer, Conv) and layer.route == 2:
                x = torch.cat([x, conv_routes[-1]], dim=1)
                conv_routes.pop()
            elif isinstance(layer, Conv) and layer.route == 3:
                x = torch.cat([x, conv_routes[-1]], dim=1)
                conv_routes.pop()
        return outputs


if __name__ == "__main__":
    model = YOLOV5(in_channels=3, num_classes=6)
    summary(model, [3, 640, 640])

    # x = torch.rand([2, 3, 640, 640])
    # out = model(x)
    # print(out[0].shape)
    # print(out[1].shape)
    # print(out[2].shape)
