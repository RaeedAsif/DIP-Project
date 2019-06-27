import cv2
import numpy as np

def guassianBlur(img):
    frame=cv2.GaussianBlur(img,(7,7),0)
    return frame

def rgbTOhsv(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    return hsv

def masking(img,lower,upper):
    mask = cv2.inRange(img,lower,upper)
    return mask

def edgeDetection(img):
    edges=cv2.Canny(img,75,150,)
    return edges

def lineDetection(img):
    lines=cv2.HoughLinesP(img,1,np.pi/180,50,maxLineGap=50)
    return lines

def main():
    
    video=cv2.VideoCapture("road_car_view.mp4")

    while True:
        #reading frame
        ret,frame=video.read()
        #applying gaussian blur so that getting better result after smoothing image
        frame=guassianBlur(frame)
        #converting hsv colorspace so that detect whatever color we want from frame
        hsv=rgbTOhsv(frame)
        #setting lower and upper threshols for color to detect
        lower_yellow=np.array([10,94,140])
        upper_yellow=np.array([40,255,255])
        #creating mask
        mask=masking(hsv,lower_yellow,upper_yellow)
        #detecting edges
        cv2.imshow("mask",mask)

        edges=edgeDetection(mask)
        
        cv2.imshow("edges",edges)
        #making hough lines
        lines=lineDetection(edges)
        #drawing lines
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2=line[0]
                cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 3)
        #showing frame with drawn lines
        cv2.imshow("frame",frame)
        #exit if ESC press
        key=cv2.waitKey(25)
        if key==27:
            break
    #release video from memory    
    video.release()
    cv2.destroyAllWindows()

#calling main
if __name__=="__main__":
    main()