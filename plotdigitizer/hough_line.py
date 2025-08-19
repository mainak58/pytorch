import cv2
import numpy as np


img = cv2.imread("plotdigitizer/page3.png")
img_resized = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)

grey_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(grey_img, 50, 150, apertureSize=3)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=300, maxLineGap=10)

output = img_resized.copy()
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2) 

# Display the result
cv2.imshow("Detected Vertical Lines", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
