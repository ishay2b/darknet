#include <stdio.h>
#include <Python.h>

#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "dbox.h"
#include "demo.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif


static PyObject *yoloError;

extern char *voc_names[] ;/*= {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"}; */
extern image voc_labels[];


static network globalNetwork;

static PyObject *
load_network(PyObject *self, PyObject *args) //char *cfgfile, char *weightfile
{
    
    PyObject *cfgfileO=NULL;;
    PyObject *weightfileO = NULL;
    
    if (!PyArg_UnpackTuple(args, "load_network", 2, 2, &cfgfileO, &weightfileO))
    {
        return NULL;
    }
    
    const char* cfgfile = PyString_AsString(cfgfileO);
    const char* weightfile = PyString_AsString(weightfileO);
    
    globalNetwork = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&globalNetwork, weightfile);
    }
    return Py_BuildValue("O", &globalNetwork);
    
}


static void test_yolo(network *net, char *filename, float thresh){
    detection_layer l = net->layers[net->n-1];
    set_batch_network(net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    BOX *boxes = calloc(l.side*l.side*l.n, sizeof(BOX));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net->w, net->h);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(*net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
        save_image(im, "predictions");
        show_image(im, "predictions");
        
        show_image(sized, "resized");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}



static PyObject *
test_yolo_wrapper(PyObject *self, PyObject *args)
{
    
    PyObject *netpy=NULL;// &args[0]; //load_network("cfg/yolo.cfg", "yolo.weights");
    PyObject *imagePathO=NULL;
    PyObject *threshO=NULL;

    if (!PyArg_UnpackTuple(args, "load_network", 3, 3, &netpy, &imagePathO,&threshO))
    {
        return NULL;
    }
    
    network *net=NULL;
    PyArg_Parse(netpy, "O", &net);
    const char* imagePath = PyString_AsString(imagePathO);
    
    float thresh=0.4;
    PyArg_Parse(threshO, "f", &thresh);
    
    test_yolo(net, imagePath, thresh);
    return Py_BuildValue("i", 0);
}


static PyObject *
hello(PyObject *self, PyObject *args)
{
	return Py_BuildValue("s",  "hello, world");
}



static PyMethodDef yoloMethods[] = {
    {"test",  test_yolo_wrapper, METH_VARARGS, "Execute a shell command."},
    {"load_network",  load_network, METH_VARARGS, "Execute a shell command."},
    
    {"hello",  hello, METH_VARARGS, "hello"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
inityolo(void)
{
    PyObject *m;

    m = Py_InitModule("yolo", yoloMethods);
    if (m == NULL)
        return;

    yoloError = PyErr_NewException("yolo.error", NULL, NULL);
    Py_INCREF(yoloError);
    PyModule_AddObject(m, "error", yoloError);
}

