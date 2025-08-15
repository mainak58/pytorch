import cv2

img = cv2.imread("cannyedge/page.png", cv2.IMREAD_GRAYSCALE)

new_size = (800, 600) 
img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

edges = cv2.Canny(img_resized, 80, 140)

cv2.imshow("original image", img_resized)
cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
