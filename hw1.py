import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

def histogram_equalization(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    mask = cv2.inRange(hsv, np.array([0, 0, 0]),np.array([180, 255, 46]))
    mask_opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None, iterations=1)
    _, contours, hierarchy = cv2.findContours(mask_opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.cvtColor(mask_opening, cv2.COLOR_GRAY2BGR)
    for i in contours:
        area = cv2.contourArea(i)
        if area < 400 and area > 15:
            cv2.drawContours(result, [i], 0, (0, 0, 255), 2) 
    return result

def adaptive_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def sharp_masking(image):
    delta = image.astype(np.int) - cv2.GaussianBlur(image, (0, 0), 3).astype(np.int)
    result = image.astype(np.int) + 3 * delta
    result = np.where(result < 0, 0, result) 
    result = np.where(result > 255, 255, result)
    enhanced = result.astype(np.uint8) 
    return enhanced

if __name__ == "__main__":
    cap = cv2.VideoCapture('./hw1_test.avi')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('./result.avi',cv2.VideoWriter_fourcc(*'XVID'), 20.0, 
                            (frame_width ,frame_height))
    frame_num = 0
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow('video file')
    while frame_num < total_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret is False:
            break
        frame = cv2.resize(frame, (frame_width//2, frame_height//2), interpolation=cv2.INTER_LINEAR) 
        top = np.concatenate((frame, histogram_equalization(frame)),axis=1)
        bot = np.concatenate((sharp_masking(frame), adaptive_thresholding(frame)),axis=1)
        ans = np.concatenate((top, bot), axis=0)
        cv2.putText(ans, "Original", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(ans, "histogram equalization & mask", (frame_width//2, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(ans, "sharp", (0, frame_height // 2 + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(ans, "adaptive thresholding", (frame_width//2 , frame_height // 2 + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow('video file', ans)
        out.write(ans)
        key = cv2.waitKey(20) & 0xFF
        if key is 27:
            break
        frame_num += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows