import cv2
import numpy as np

def dilatation(img):
    dilatation_size = 15
    dilatation_type = 0
    val_type = 0
    if val_type == 0:
        dilatation_type = cv2.MORPH_RECT
    elif val_type == 1:
        dilatation_type = cv2.MORPH_CROSS
    elif val_type == 2:
        dilatation_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    img_dilated = cv2.dilate(img, element)
    return img_dilated

frame = cv2.imread('wortels.jpg', 1)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(frame_gray,127,255,cv2.THRESH_BINARY)
dilated_mask = dilatation(mask)
contours, hierarchy = cv2.findContours(dilated_mask,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

frame_copy = frame.copy()
count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    if (area > 35000):
        color = (0, 255, 0)
        count += 1
    else:
        color = (0, 0, 255)
    cv2.rectangle(frame_copy,(x,y),(x+w,y+h),color,2)

    cv2.putText(frame_copy,f"Number of roots found: {count}",(20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)
    #plt.imshow(frame_copy[..., ::-1])
    cv2.imshow("frame", frame)
    #cv2.imshow("mask", mask)
    #cv2.imshow("res", res)

    key = cv2.waitKey(100)
    if key == 27:
        break

cv2.destroyAllWindows()

