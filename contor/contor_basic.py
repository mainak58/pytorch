import cv2
import numpy as np

img = cv2.imread("cannyedge/page.png")
new_size = (800, 600)
img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

lower_blue = np.array([90, 30, 150])  
upper_blue = np.array([130, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = img_resized.copy()
cv2.drawContours(output, contours, -1, (0, 0, 255), 2)

cv2.imshow("Light Blue Mask", mask_clean)
cv2.imshow("Detected Light Blue Bars", output)
cv2.waitKey(0)
cv2.destroyAllWindows()



# import cv2

# img = cv2.imread("cannyedge/page.png")
# new_size = (800, 600)
# img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

# grey_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(grey_img, 50, 150)

# contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(img_resized, contours, -1, (0, 255, 0), 2)

# cv2.imshow("Edges", edges)
# cv2.imshow("Contours", img_resized)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



# import cv2

# img = cv2.imread("contor/page.png")

# new_size = (1000, 800)
# img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

# grey_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(grey_img, 200, 255, cv2.THRESH_BINARY_INV)


# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# # Find contours
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Draw only large enough contours (filter small noise)
# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area > 1000:  # adjust as needed
#         x, y, w, h = cv2.boundingRect(cnt)
#         cv2.rectangle(img_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # Show results
# cv2.imshow("Detected Bars", img_resized)
# cv2.imshow("Threshold", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

