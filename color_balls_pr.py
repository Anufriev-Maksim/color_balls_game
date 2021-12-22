import cv2
import random

cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera")

lower = [(53, 70, 100),(160, 130, 200),(90, 120, 90)]
upper = [(63, 160, 180),(190, 200, 255),(130, 190, 190)]


def mask_ball(lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 2)
    return mask

sequence = ["R", "G", "B"] 
random.shuffle(sequence)

def contr_ball(cnts, image, coloring):
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        (curr_x, curr_y), radius = cv2.minEnclosingCircle(c)
        if radius > 10:
            cv2.circle(image, (int(curr_x) ,int(curr_y)), 5,
                            coloring ,2)
            cv2.circle(image, (int(curr_x) ,int(curr_y)), int(radius),
                            coloring,2)
            return int(curr_x)
    return 0

while cam.isOpened():
    _, image = cam.read()
    blurred = cv2.GaussianBlur(image, (11,11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    maskGr= mask_ball(lower[0],upper[0])
    maskPn= mask_ball(lower[1],upper[1])
    maskBl= mask_ball(lower[2],upper[2])

    mask = maskBl + maskGr + maskPn
    
    cntsBl = cv2.findContours(maskBl.copy(), cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    cntsGr = cv2.findContours(maskGr.copy(), cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    cntsOr = cv2.findContours(maskPn.copy(), cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)[-2]


    x_cords = [contr_ball(cntsBl,image, (255,0,0)), contr_ball(cntsGr,image, (0,255,0)),contr_ball(cntsOr,image, (0, 0, 255))]

    so_colors = {"R": x_cords[2], "G": x_cords[1], "B": x_cords[0]}
    curr_order = sorted(so_colors, key=so_colors.get)
    
    if curr_order == sequence:
        cv2.putText(image, "You win!", (60,100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),4)
        
    cv2.imshow("Camera", image)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1)
    
    

    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

