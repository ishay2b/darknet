'''
    YOLO python wrapper calling yolo.so
'''

import yolo
from ctypes import *
import numpy
import cv2


class Detection():
    def __repr__(self):
        return "x(%d), y(%d), w(%d), h(%d), c(%d), prob(%d%%)" %(self.x, self.y, self.w, self.h, self.index, self.prob)
    pass


class YOLO():
    def __init__(self, cfg, weights, h=448, w=448):
        self.dll=cdll.LoadLibrary(yolo.__file__) # Load the as c dll using the path
        
        # Let ctypes know we expect a pointer back so it does not truncate to int
        self.dll.load_network.restype=c_void_p
        # Load the network
        self.net=c_void_p(self.dll.load_network(c_char_p(cfg), c_char_p(weights)))

        # Set hard coded values
        self.h=h
        self.w=w
        self.c=3
        self.MAX_BOX=30 # Max number of returned detections
    

    def preprocess(self, image):
        ''' @input bgr interleved 0..255
            @output bgr split chanels, resized, 0..1, continues in memory
        '''
        image=cv2.resize(image, (self.h,self.w))
        image=numpy.array(cv2.split(image), dtype='f4', order='C') # make sure it is contiuos (C)
        image=image/255. # Scale to 0..1
        return image

    def testPreprocessed(self, image):
        '''
        extern int test_yolo_cv(network *net,
        int h,
        int w,
        int c,
        float *data,
        float thresh,
        float*predictions)
        '''
        self.res=numpy.zeros(self.MAX_BOX*8, dtype='f4', order='CW') # Allocate continuos writable storage for result (float*)
        num_objects=self.dll.test_yolo_cv(self.net,
                                          c_int(self.h),
                                          c_int(self.w),
                                          c_int(self.c),
                                          c_void_p(image.ctypes.data),
                                          c_float(0.1),
                                          c_void_p(self.res.ctypes.data)
                                          )
                                     
        return YOLO.parse_results(num_objects,self.res) # from float * to python objects
    
    def test(self, image):
        '''
             test a given image and return python Prediction list.
        '''
        return self.testPreprocessed(self.preprocess(image))


    @staticmethod
    def parse_results(num_objects, res):
        ''' parse the float * c into python code
            '''
        p=[]
        z=0
        for i in range(num_objects):
            d=Detection()
            d.x=res[z+0]
            d.y=res[z+1]
            d.w=res[z+2]
            d.h=res[z+3]
            d.index=res[z+4]
            d.prob=res[z+5]
            p.append(d)
            
            z+=8 # skip 8 due to alinment
        return p
    
    


if __name__=='__main__':
    ''' Usage example '''
    myYolo=YOLO('cfg/yolo-small.cfg', 'yolo-small.weights')
    image=cv2.imread('data/dog.jpg')
    res=myYolo.test(image)

    print "first res", res[0]

