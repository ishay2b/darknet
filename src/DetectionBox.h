#ifndef DETECTION_BOX_H
#define DETECTION_BOX_H
/*
#include "DetectionBox.h" 
 */

#include "dbox.h"


typedef struct DetectionBox{ // Actually will be treated as array of 8 floats in python
    BOX box;
    float class_index; //Should be int but int will ruin the 8 floats struc wanted
    float probability;
    
    float pad1 ; // alignment
    float pad2 ; // alignment
    
} DetectionBox;

DetectionBox detbox(BOX a, int class_indexP, float probabilityP);

#define MAX_DET_BOX 30
typedef DetectionBox DetBoxArray[MAX_DET_BOX];


#endif
