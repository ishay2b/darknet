#include <stdio.h>

#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "dbox.h"
#include "demo.h"

#ifdef OPENCV
#include <Python.h>
#include "opencv2/highgui/highgui_c.h"
#endif

#include "DetectionBox.h"

extern char *voc_names[] ;
/*= {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"}; */
extern image voc_labels[];


static network globalNetwork; // Hack

extern network * load_network(const char* cfgfile, const char* weights)
{
    globalNetwork = parse_network_cfg(cfgfile);
    if(weights){
        load_weights(&globalNetwork, weights);
    }
    
    set_batch_network(&globalNetwork, 1);
    printf("c sending python %p\n", (void*)&globalNetwork);
    
    return &globalNetwork;
}

const int MAX_DET_BOX=30;
typedef DetectionBox DetBoxArray[MAX_DET_BOX];

int translate_detections(DetectionBox *detBoxArray, image im, int num, float thresh, BOX *boxes, float **probs, char **names, int classes)
{
    /*
        unify all the data to a simple returned float array of 30*8
     */
    int i=0;
    int j=0; //The counter per valid objects
    
    for(i = 0; i < num && j<MAX_DET_BOX; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){
            int width = pow(prob, 1./2.)*10+1;
            width = 8;
    
            BOX b = boxes[i];
            
            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;
            
            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;
            
            printf("name (%s), prob(%.0f%%), x(%d), y(%d), w(%d), h(%d) \n",
                   names[class], prob*100, left, top, right-left, bot-top);

            DetectionBox *debox=&detBoxArray[j];
            debox->box.x=left;
            debox->box.y=top;
            debox->box.w=right-left;
            debox->box.h=bot-top;
            debox->class_index=class; // assiging int to float
            debox->probability=prob*100;
            j++; // Next valid object please
        }
    }
    return j;
}

extern int test_yolo_cv(network *net,
                         int h,
                         int w,
                         int c,
                         float *data,
                         float thresh,
                         DetBoxArray* detBoxArray){
    
    image im;
    im.h=h;
    im.w=w;
    im.c=c;
    im.data=data;
    
    
    detection_layer l = net->layers[net->n-1];
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    BOX *boxes = calloc(l.side*l.side*l.n, sizeof(BOX));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

    //image im = load_image_color(input,0,0);
    //image sized = resize_image(im, net->w, net->h);
    float *X = im.data;
    time=clock();
    float *predictions = network_predict(*net, X);
    printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
    convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
    if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    int num_objects=translate_detections(detBoxArray, im,  l.side*l.side*l.n, thresh, boxes, probs, voc_names, 20);
    return num_objects;
//    draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
}



///   predcated PyObject usage - left for debug///

static PyObject *yoloError;

static PyObject *
hello(PyObject *self, PyObject *args)
{
    float *ptr=NULL;
    PyObject *ptrPy=NULL;
    
    //Extract the 3 parameters need
    if (!PyArg_UnpackTuple(args, "hello", 1, 1, &ptrPy))
    {
        return NULL;
    }
    PyArg_Parse(ptrPy, "O", &ptr); // Parse as an object

    
    printf("%f, %f, %f",ptr[0], ptr[1], ptr[2]);
    
	return Py_BuildValue("d",  ptr[0]+ptr[1]+ptr[2]);
}


static PyMethodDef yoloMethods[] = {
    {"hello",  hello, METH_VARARGS, "hello"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyObject *yoloContext;

PyMODINIT_FUNC
inityolo(void)
{
    yoloContext = Py_InitModule("yolo", yoloMethods);
   // import_array();  // Must be present for NumPy.  Called first after above line.
    
    if (yoloContext == NULL)
        return;
    yoloError = PyErr_NewException("yolo.error", NULL, NULL);
    Py_INCREF(yoloError);
    
    PyModule_AddObject(yoloContext, "error", yoloError);
}

