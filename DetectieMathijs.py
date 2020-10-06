import cv2
import numpy as np


def nothing(x):
    pass

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

while True:
    frame = cv2.imread('wortels.jpg')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_h = 4
    l_s = 44
    l_v = 26
    u_h = 95
    u_s = 237
    u_v = 136
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, l_b, u_b)
    res = cv2.bitwise_and(frame, frame, mask=mask)


    dilated_mask = dilatation(mask)
    contours, hierarchy = cv2.findContours(dilated_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    frame_copy = frame.copy()
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)


        if (area > 22000):
            color = (0, 255, 0)
            count += 1

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(frame_copy, f"Area: {area}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1,
                        cv2.LINE_AA)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 2)
        else:
            color = (0, 0, 255)

    cv2.putText(frame_copy, f"Number of roots found: {count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                cv2.LINE_AA)

    cv2.imshow("frame", frame_copy)
    #cv2.imshow("mask", mask)
    #cv2.imshow("res", res)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()





