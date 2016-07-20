#include "DetectionBox.h"

DetectionBox detbox(BOX a, int class_indexP, float probabilityP){
    DetectionBox b;
    b.box = a;
    b.class_index= class_indexP;
    b.probability =probabilityP;
    return b;
}