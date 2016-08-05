# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import sys, os
import YOLO
import argparse
import datetime
import imutils
import time
import cv2
import numpy

## GLOBALS
ALLOWED_GAP = 3
HARD_THRESHOLD = 0.8
SOFT_THRESHOLD = 0.5
PROB_THRESH = 40
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
TARGET_CLASS = CLASSES.index('person')

## end GLOBALS

#=============== HELPER FUNCTIONS=====================
def toPoints(bb):
    # returns [X1,Y1,X2,Y2]
    return [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]

def toRect(bbPoints):
    #returns [X1,Y1,W,H]
    return [bbPoints[0],\
            bbPoints[1],\
            bbPoints[2]-bbPoints[0],\
            bbPoints[3]-bbPoints [1]]
    
def rectArea(bb):
    return bb[2] * bb[3]

def rectUnion(bb1,bb2):
    bb1Points = toPoints(bb1)
    bb2Points = toPoints(bb2)
    rect = toRect(    [min(bb1Points[0],bb2Points[0]),
                    min(bb1Points[1],bb2Points[1]),
                    max(bb1Points[2],bb2Points[2]),
                    max(bb1Points[3],bb2Points[3])])
    return rect
    
def rectOverlap(bb1,bb2,threshhold=0.5):
    # convert to 4 points
    bb1Points = toPoints(bb1)
    bb2Points = toPoints(bb2)
    overlapbb = toRect([max(bb1Points[0], bb2Points[0]),
                        max(bb1Points[1], bb2Points[1]),
                        min(bb1Points[2], bb2Points[2]),
                        min(bb1Points[3], bb2Points[3])])
    if overlapbb[2] <= 0 or overlapbb[3] <= 0 :
        # width or height are negative. no overlap here
        return None
    if threshhold:
        if float(rectArea(overlapbb))/max(rectArea(bb1),rectArea(bb2)) > threshhold:
            return overlapbb
        else:
            return None
    return overlapbb

#==============================END HELPER============================

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=200, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
# if args.get("video", None) is None:
#     camera = cv2.VideoCapture(0)
#     time.sleep(0.25)
# 
# # otherwise, we are reading from a video file
# else:
#     camera = cv2.VideoCapture(args["video"])
vidFile = '/home/gili/dev/YOLO/darknet/MVI_2966.MOV'
camera = cv2.VideoCapture(vidFile)
# initialize the first frame in the video stream
firstFrame = None
frame_num = 0
tracks = {}
results = []

#init network
HOMEDIR = os.path.join(os.path.dirname(__file__),'../')
myYolo=YOLO.YOLO(HOMEDIR+'cfg/yolo-small.cfg', HOMEDIR+'yolo-small.weights')

# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    frame_num+=1
    # clean up tracks
    trackDel = []
    for t in tracks.keys():
        if tracks[t]['frames'][-1]['frameNum'] + ALLOWED_GAP < frame_num:
            trackDel.append(t)
    for t in trackDel:
        del(tracks[t])
    
    (grabbed, frame) = camera.read()
    text = "Unoccupied"
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break
    else:
        print "working on frame: %d" %frame_num
    frame = imutils.resize(frame,width=480)
    # if the first frame is None, initialize it
    if firstFrame is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        firstFrame = gray
    yoloRes = myYolo.test(frame)
    yoloRects = {}
    # TODO: add support for multiple occurences
    for y in yoloRes:
        if y.index == TARGET_CLASS and y.prob > PROB_THRESH:
            yoloRects[TARGET_CLASS] = [y.x, y.y, y.w, y.h]
            w=numpy.float32(640/480.0*y.w)
            x=numpy.float32(640/480.0*y.x)
            cv2.rectangle(frame, (x, y.y), (x + w, y.y + y.h), (0, 255, 0), 2)
            print "found a %s" %CLASSES[TARGET_CLASS]
    #        
    if not (tracks or yoloRects):
    #    theres no reason to perform tracking on this frame...go to next one
        pass
    else:
        # resize the frame, convert it to grayscale, and blur it
        #frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue
    
        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    
        # loop over the contours
        boundingBoxes=[]
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue
    
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            boundingBoxes.append(cv2.boundingRect(c))
        bb2delete =[]
        #-----Joining overlapping bounding boxes----
        for i in range(len(boundingBoxes)-1,0,-1):
            for j in range(i-1,-1,-1):
                if rectOverlap(boundingBoxes[i],boundingBoxes[j],threshhold=None):
                    rect = rectUnion(boundingBoxes[i],boundingBoxes[j])
                    if rect:
                        boundingBoxes[j] = rect
                        bb2delete.append(i)
                        break;
            
        for i in bb2delete:
            boundingBoxes.pop(i)
        #-----------------end joining----------------
        used = []
        for b in boundingBoxes:
            if b in used:
                continue
            for c in yoloRects:
                # check for match against yolo rect
                if rectOverlap(yoloRects[c], b):
                    print "Yolo detection matches a box"
                    #check for match with previously saved tracks
                    if tracks.get(c) and frame_num - tracks[c]['frames'][-1]['frameNum'] < ALLOWED_GAP:
                        if rectOverlap(tracks[c]['frames'][-1]['rect'], b):
                            "matched box matches an existing track"
                            #BINGO!!!! dump tracked frames into results
                            if len(tracks[c]['frames'])>1:
                                # do not append consecutive yoloDetections. that makes no sense
                                results.append(tracks[c])
                            del(tracks[c])
                            tracks[c] = {'frames':[{
                                                'rect': b,
                                                'class': c,
                                                'frameNum': frame_num,
                                                'yoloRect': yoloRects[c]
                                                                        }]}
                        else:
                            # for now assuming only 1 instance of each class
                            # a mismatch between tracker and new yolo classification
                            # implies tracker data is garbage.
                            del(tracks[c]) # not necessary but doing this explicitly as a reminder
                    
                    #start new tracking channel
                    tracks[c] = {'frames':[{
                                                'rect': b,
                                                'class': c,
                                                'frameNum': frame_num,
                                                'yoloRect': yoloRects[c]
                                                                        }]}
                    x, y, w, h = b[0], b[1], b[2], b[3]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    used.append(b) #want to make sure we don't try and use this twice
                    
        for b in boundingBoxes:
            if b in used:
                continue        
            for t in tracks:
                if tracks[t].get('frames'):
                    if rectOverlap(tracks[t]['frames'][-1]['rect'], b):
                        print "found a box that matches a track"
                        used.append(b)
                        tracks[t]['frames'].append({    'rect': b,
                                                'class': tracks[t]['frames'][-1]['class'],
                                                'frameNum': frame_num})
    
                
            x, y, w, h = b[0], b[1], b[2], b[3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
    
#         # draw the text and timestamp on the frame
#         cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#         cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
#             (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
        # show the frame and record if the user presses a key
        cv2.imshow("DETECTION",frame)
        #cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()