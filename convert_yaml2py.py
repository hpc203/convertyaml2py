from common import *
import yaml
import os

def convert_list2str(s):
    out = []
    for x in s:
        if not isinstance(x, str):
            out.append(str(x))
        else:
            out.append('\'' + x + '\'')
    return ', '.join(out)

def conver_listtostr(l):
    out = []
    for i, data in enumerate(l):
        if isinstance(data, list):
            out.append('['+', '.join(map(str, data))+']')
        else:
            out.append(str(data))
    return '[' + ', '.join(out) + ']'

def parse_model2py(d, ch, model_name, upsample_concat_use_nn_module=False):  # model_dict, input_channels(3)
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    wtxt = open(model_name + '.py', 'w')
    wtxt.write('from common import *\n' + '\n')
    wtxt.write('class My_YOLO(nn.Module):\n')
    init_str = '    def __init__(self, num_classes='+str(nc)+', anchors='+conver_listtostr(anchors)+', training=False):\n'
    # wtxt.write('    def __init__(self):\n')
    wtxt.write(init_str)
    wtxt.write('        super().__init__()\n')
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print

        fname = t[t.rfind('.') + 1:]
        if t.startswith('common.') and m is not Detect:
            if fname not in ('Upsample', 'Concat') or upsample_concat_use_nn_module:
                wtxt.write('        self.seq' + str(i) + '_' + fname + ' = ' + fname + '(' + convert_list2str(args) + ')\n')
        elif t.startswith('torch.nn.modules') and m is not Detect:
            if fname not in ('Upsample', 'Concat') or upsample_concat_use_nn_module:
                wtxt.write('        self.seq' + str(i) + '_' + fname + ' = nn.' + fname + '(' + convert_list2str(args) + ')\n')
        elif m is Detect:
            wtxt.write('        self.yolo_layers = ' + fname + '(nc=num_classes, anchors=anchors, ch=' + conver_listtostr(args[2]) + ', training=training)\n')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    wtxt.write('    def forward(self, x):\n')
    wtxt.write('        x = self.seq0_Focus(x)\n')
    wtxt.write('        x = self.seq1_Conv(x)\n')
    wtxt.write('        x = self.seq2_BottleneckCSP(x)\n')
    wtxt.write('        x = self.seq3_Conv(x)\n')
    wtxt.write('        xRt0 = self.seq4_BottleneckCSP(x)\n')
    wtxt.write('        x = self.seq5_Conv(xRt0)\n')
    wtxt.write('        xRt1 = self.seq6_BottleneckCSP(x)\n')
    wtxt.write('        x = self.seq7_Conv(xRt1)\n')
    wtxt.write('        x = self.seq8_SPP(x)\n')
    wtxt.write('        x = self.seq9_BottleneckCSP(x)\n')
    wtxt.write('        xRt2 = self.seq10_Conv(x)\n')
    if upsample_concat_use_nn_module:
        wtxt.write('        route = self.seq11_Upsample(xRt2)\n')
        wtxt.write('        x = self.seq12_Concat([route, xRt1])\n')
    else:
        wtxt.write('        route = F.interpolate(xRt2, size=(int(xRt2.shape[2] * 2), int(xRt2.shape[3] * 2)), mode=\'nearest\')\n')
        wtxt.write('        x = torch.cat([route, xRt1], dim=1)\n')
    wtxt.write('        x = self.seq13_BottleneckCSP(x)\n')
    wtxt.write('        xRt3 = self.seq14_Conv(x)\n')
    if upsample_concat_use_nn_module:
        wtxt.write('        route = self.seq15_Upsample(xRt3)\n')
        wtxt.write('        x = self.seq16_Concat([route, xRt0])\n')
    else:
        wtxt.write('        route = F.interpolate(xRt3, size=(int(xRt3.shape[2] * 2), int(xRt3.shape[3] * 2)), mode=\'nearest\')\n')
        wtxt.write('        x = torch.cat([route, xRt0], dim=1)\n')
    wtxt.write('        out1 = self.seq17_BottleneckCSP(x)\n')
    wtxt.write('        route = self.seq18_Conv(out1)\n')
    if upsample_concat_use_nn_module:
        wtxt.write('        x = self.seq19_Concat([route, xRt3])\n')
    else:
        wtxt.write('        x = torch.cat([route, xRt3], dim=1)\n')
    wtxt.write('        out2 = self.seq20_BottleneckCSP(x)\n')
    wtxt.write('        route = self.seq21_Conv(out2)\n')
    if upsample_concat_use_nn_module:
        wtxt.write('        x = self.seq22_Concat([route, xRt2])\n')
    else:
        wtxt.write('        x = torch.cat([route, xRt2], dim=1)\n')
    wtxt.write('        out3 = self.seq23_BottleneckCSP(x)\n')
    wtxt.write('        output = self.yolo_layers([out1, out2, out3])\n')
    wtxt.write('        return output\n')
    wtxt.close()
    return nn.Sequential(*layers), sorted(save)

if __name__ == '__main__':
    choices = ['yolov5s', 'yolov5l', 'yolov5m', 'yolov5x']
    model_cfg, ch = choices[0]+'.yaml', 3
    with open(model_cfg) as f:
        md = yaml.load(f, Loader=yaml.FullLoader)
    model_name = os.path.splitext(os.path.basename(model_cfg))[0]
    model, save = parse_model2py(md, [ch], model_name, upsample_concat_use_nn_module=False)
    print('generate '+model_name+'.py successfully')