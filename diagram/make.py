from graphviz import Digraph
import argparse
import os


base_dir = os.path.dirname(os.path.realpath(__file__))


def draw_graph(format, filename, spline_type):

    g = Digraph('Current State of FPGA Deployment of Neural Networks', filename=filename, format=format,
                graph_attr={'splines': spline_type, 'overlap': 'false', 'concentrate': 'true'})
    g.node_attr.update(shape='box', color='black')
    g.attr(compound='true')

    with g.subgraph(name='cluster_0') as c:
        c.attr(style='bold', color='dodgerblue')
        c.attr(rankdir='LR')
        c.node('PyTorch')
        c.node('Caffe2')
        c.node('Glow')
        c.edge('PyTorch', 'Caffe2',
               style='invisible', dir="none")
        c.attr(label='Facebook')

        with g.subgraph(name='cluster_1') as c:
            c.attr(style='bold', color='navyblue')
            c.node('CNTK')
            c.node('ONNX')
            c.attr(label='Microsoft')

    with g.subgraph(name='cluster_2') as c:
        c.attr(style='bold', color='orange')
        c.node('Tensorflow')
        c.node('Tensorflow Lite')
        c.node('Keras')
        # c.edge('Tensorflow', 'Tensorflow Lite',
        #        style='invisible', dir="none", rank='same')
        # c.edge('Keras', 'Tensorflow Lite',
        #        style='invisible', dir="none", rank='same')
        c.attr(label='Google')

    with g.subgraph(name='cluster_3') as c:
        c.attr(style='bold', color='red')
        c.node('MxNet')
        c.node('Chainer')
        c.node('Caffe')
        # c.edge('MxNet', 'Chainer',
        #        style='invisible', dir="none", rank='same')
        # c.edge('Chainer', 'Caffe',
        #        style='invisible', dir="none", rank='same')
        c.attr(label='Other')

    with g.subgraph(name='cluster_4') as c:
        c.attr(style='bold', color='royalblue1')
        c.node('Distiller')
        c.node('OpenVino')
        # c.edge('Distiller', 'OpenVino',
        #        style='invisible', dir="none", rank='same')
        c.attr(label='Intel')

    with g.subgraph(name='cluster_5') as c:
        c.attr(style='bold', color='crimson')
        with c.subgraph(name='cluster_6') as f:
            f.node('Brevitas')
            f.node('FINN-HLS')

            f.attr(label='FINN')

        with c.subgraph(name='cluster_7') as v:

            with v.subgraph(name='cluster_8') as d:
                d.node('AI Optimizer')
                d.node('AI Quantizer')
                d.node('AI Compiler')
                d.edge('AI Optimizer', 'AI Quantizer',
                       style='invisible', dir="none")
                d.edge('AI Quantizer', 'AI Compiler',
                       style='invisible', dir="none")
                d.attr(label='Model Optimization')

            with v.subgraph(name='cluster_9') as o:
                o.node('Deep Learning Processing Unit')
                o.attr(label='Overlay')

            v.node('AI Profiler')
            v.node('Vitis Accellerated Libraries')
            v.node('Xilinx Runtime Library')

            v.edge('AI Compiler', 'Xilinx Runtime Library', ltail='cluster_8')

            v.edge('Vitis Accellerated Libraries',
                   'AI Quantizer', lhead='cluster_8')
            v.edge('AI Quantizer', 'AI Profiler', ltail='cluster_8')
            v.edge('AI Profiler', 'AI Quantizer', lhead='cluster_8')
            v.edge('Xilinx Runtime Library',
                   'Deep Learning Processing Unit', lhead='cluster_9')
            # v.edge('AI Profiler', 'Vitis Accellerated Libraries',
            #        style='invisible', dir="none")
            # v.edge('Vitis Accellerated Libraries', 'Xilinx Runtime Library',
            #        style='invisible', dir="none")

            v.attr(label='Vitis')

        c.attr(label='Xilinx')

    with g.subgraph(name='cluster_10') as c:
        c.attr(style='bold', color='lightskyblue')
        c.node('Relay')
        c.node('VTA')
        c.attr(label='TVM')

    g.node('FPGA Deployment', shape='hexagon')

    g.node('_', style='invis')
    g.edge('_', 'Relay', style='invis', dir='none')
    g.edge('_', 'Tensorflow', style='invis', dir='none')

    # Edges

    # ONNX
    g.edges([('PyTorch', 'ONNX'), ('CNTK', 'ONNX'), ('MxNet', 'ONNX'),
             ('Caffe2', 'ONNX'), ('Tensorflow', 'ONNX'), ('Chainer', 'ONNX'),
             ('Keras', 'ONNX')])

    # TVM
    g.edges([('PyTorch', 'VTA'), ('Tensorflow', 'VTA'),
             ('ONNX', 'VTA'), ('Keras', 'VTA'), ('MxNet', 'VTA'), ('Tensorflow Lite', 'VTA'), ('Relay', 'VTA'), ('Caffe2', 'VTA')])

    # Intel
    # OpenVino
    g.edges([('Tensorflow', 'OpenVino'), ('ONNX', 'OpenVino'),
             ('Caffe', 'OpenVino')])
    # Distiller
    g.edges([('PyTorch', 'Distiller')])

    # Glow
    g.edges([('PyTorch', 'Glow')])

    # Xilinx
    # FINN
    g.edges([('PyTorch', 'Brevitas'), ('Brevitas', 'FINN-HLS')])
    # Vitis
    g.edge('Caffe', 'AI Quantizer', lhead='cluster_8')
    g.edge('Tensorflow', 'AI Quantizer', lhead='cluster_8')
    g.edge('PyTorch', 'AI Quantizer', lhead='cluster_8', style='dashed')

    # FPGA
    g.edges([('Deep Learning Processing Unit', 'FPGA Deployment'), ('OpenVino', 'FPGA Deployment'),
             ('VTA', 'FPGA Deployment'), ('Glow', 'FPGA Deployment'), ('FINN-HLS', 'FPGA Deployment')])

    return g


if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_format', '-o',
                        help='Choose the output type either svg, png, or pdf', default='png')
    parser.add_argument('-filename', '-f', type=str,
                        help='Output filename', default='/diagram.gv')
    parser.add_argument('--line_type', '-l',
                        help='Choose from the standard spline choices with graphviz: line, polyline, curved, spline, ortho', default='spline')
    parser.add_argument(
        '--show', '-s', help='Choose to view instantly y/n', default='y')

    args = parser.parse_args()

    file_choices = ['svg', 'png', 'pdf']
    line_choices = ['line', 'polyline', 'curved', 'spline', 'ortho']
    filename = base_dir+args.filename
    format = args.output_format

    print(base_dir+args.filename)
    g = draw_graph(format=args.output_format, filename=filename,
                   spline_type=args.line_type)

    g.save()
    g.render(filename=filename, format=format,
             view=True if args.show == 'y' else False)
