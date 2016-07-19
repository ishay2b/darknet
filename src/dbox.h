#ifndef BOX_H
#define BOX_H

typedef struct{
    float x, y, w, h;
} BOX;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

BOX float_to_box(float *f);
float box_iou(BOX a, BOX b);
float box_rmse(BOX a, BOX b);
dbox diou(BOX a, BOX b);
void do_nms(BOX *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort(BOX *boxes, float **probs, int total, int classes, float thresh);
BOX decode_box(BOX b, BOX anchor);
BOX encode_box(BOX b, BOX anchor);

#endif
