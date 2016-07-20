#!/Applications/anaconda/bin/python

from distutils.core import setup, Extension
#from distutils.extension import Extension
import commands

def pkgconfig(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    for token in commands.getoutput("pkg-config --libs --cflags %s" % ' '.join(packages)).split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw


opencv=pkgconfig('opencv') # Get opencv include dirs, library, library dirs

include_dirs =['src'] # Darknet src folder
include_dirs.extend(opencv['include_dirs']) # extend with open cv includes
libraries=opencv['libraries'] # open cv is the only 3ed needed for this python compilation
library_dirs=opencv['library_dirs'] # open cv library dirs

extra_compile_args=['-g','-O3','-DOPENCV','-DNDEBUG']


module1 = Extension('yolo',
                    include_dirs = include_dirs,
                    libraries = libraries,
                    library_dirs = library_dirs,
                    extra_compile_args=extra_compile_args,
                    sources = [
                               'yolowrapper.c',
                               'src/rnn.c',
                               'src/rnn_layer.c',
                               'src/dice.c',
                               'src/im2col.c',
                               'src/writing.c',
                               'src/cuda.c',
                               'src/classifier.c',
                               'src/softmax_layer.c',
                               'src/local_layer.c',
                               'src/yolo.c',
                               'src/batchnorm_layer.c',
                               'src/tag.c',
                               'src/gemm.c',
                               'src/maxpool_layer.c',
                               'src/swag.c',
                               'src/detection_layer.c',
                               'src/demo.c',                               
                               'src/matrix.c',
                               'src/cost_layer.c',
                               'src/deconvolutional_layer.c',
                               'src/data.c',
                               'src/shortcut_layer.c',
                               'src/crnn_layer.c',
                               'src/avgpool_layer.c',
                               'src/normalization_layer.c',
                               'src/option_list.c',
                               'src/coco.c',
                               'src/gru_layer.c',
                               'src/art.c',
                               'src/crop_layer.c',
                               'src/cifar.c',
                               'src/blas.c',
                               'src/connected_layer.c',
                               'src/col2im.c',
                               'src/activation_layer.c',
                               'src/imagenet.c',
                               'src/nightmare.c',
                               'src/network.c',
                               'src/dropout_layer.c',
                               'src/list.c',
                               'src/go.c',
                               'src/convolutional_layer.c',
                               'src/parser.c',
                               'src/rnn_vid.c',
                               'src/layer.c',
                               'src/image.c',
                               'src/utils.c',
                               'src/route_layer.c',
                               'src/activations.c',
                               'src/captcha.c',
                               'src/box.c',
                               'src/compare.c'
                               ])

setup (name = 'yolo',
       version = '1.0',
       description = 'yolo python wrapper',
       ext_modules = [module1])

'''
import python.YOLO
myYolo=YOLO('cfg/yolo-small.cfg', 'yolo-small.weights')
image=cv2.imread('data/dog.jpg')
res=myYolo.test(image)
'''




