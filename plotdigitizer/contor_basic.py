import cv2
import numpy as np


img = cv2.imread("plotdigitizer/page3.png")
print("image size", img.shape)
new_size = (800, 600)
img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
output = img_resized.copy()

grey_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(grey_img, cv2.CV_64F, 1, 0, ksize=5)
sobel_x = cv2.convertScaleAbs(sobel_x)
_, thresh_x = cv2.threshold(sobel_x, 50, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
thresh_x_clean = cv2.morphologyEx(thresh_x, cv2.MORPH_CLOSE, kernel)
contours_x, _ = cv2.findContours(thresh_x_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bar_tops, bar_bottoms = [], []

for contour in contours_x:
    x, y, w, h = cv2.boundingRect(contour)
    if w < h and h > 300:
        bar_tops.append(y)
        bar_bottoms.append(y + h)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

sobel_y = cv2.Sobel(grey_img, cv2.CV_64F, 0, 1, ksize=5)
sobel_y = cv2.convertScaleAbs(sobel_y)
_, thresh_y = cv2.threshold(sobel_y, 50, 255, cv2.THRESH_BINARY)
thresh_y_clean = cv2.morphologyEx(thresh_y, cv2.MORPH_CLOSE, kernel)
contours_y, _ = cv2.findContours(thresh_y_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

score_lines = []

for contour in contours_y:
    x, y, w, h = cv2.boundingRect(contour)
    if w > h and w < 25.5 and h > 10:
        score_center_y = y + h // 2
        score_lines.append((x, y, w, h, score_center_y))
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

if bar_tops and score_lines:
    chart_top = min(bar_tops)
    chart_bottom = max(bar_bottoms)
    full_width = img_resized.shape[1]
    cv2.rectangle(output, (0, chart_top), (full_width, chart_bottom), (0, 255, 255), 2)

    red_bar_height = chart_bottom - chart_top

    score_lines.sort(key=lambda item: item[0])

    print("\nDetected Score Line Percentages (Left to Right):")
    for x, y, w, h, center_y in score_lines:
        percentage = ((chart_bottom - center_y) / red_bar_height) * 100
        percentage = round(percentage, 1)
        print(f"x={x}, y={center_y} → {percentage}%")

else:
    print("No valid vertical bars or score lines found.")

cv2.imshow("Bar Chart with Lines and Bounding Box", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
