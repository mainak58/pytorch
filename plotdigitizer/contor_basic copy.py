import cv2
import numpy as np

# Read the image
img = cv2.imread("plotdigitizer/best_matched_crop.png")

# Resize the image
new_size = (830, 600)
img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
output = img_resized.copy()

# Convert to grayscale
grey_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Sobel filter for vertical edges (x-direction)
sobel_x = cv2.Sobel(grey_img, cv2.CV_64F, 1, 0, ksize=5)
sobel_x = cv2.convertScaleAbs(sobel_x)
_, thresh_x = cv2.threshold(sobel_x, 50, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
thresh_x_clean = cv2.morphologyEx(thresh_x, cv2.MORPH_CLOSE, kernel)
contours_x, _ = cv2.findContours(thresh_x_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bar_tops, bar_bottoms = [], []

# Detect and draw vertical bars
for contour in contours_x:
    x, y, w, h = cv2.boundingRect(contour)
    if w < h and h > 200:
        bar_tops.append(y)
        bar_bottoms.append(y + h)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color for vertical bars

# Sobel filter for horizontal edges (y-direction)
sobel_y = cv2.Sobel(grey_img, cv2.CV_64F, 0, 1, ksize=5)
sobel_y = cv2.convertScaleAbs(sobel_y)
_, thresh_y = cv2.threshold(sobel_y, 50, 255, cv2.THRESH_BINARY)
thresh_y_clean = cv2.morphologyEx(thresh_y, cv2.MORPH_CLOSE, kernel)
contours_y, _ = cv2.findContours(thresh_y_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

score_lines = []

# Draw all horizontal lines (including overlapping ones)
for contour in contours_y:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 40 and h > 12 :  # Filter out tiny lines (you can adjust this as needed)
        # Draw horizontal lines in blue color
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color for horizontal lines
        score_lines.append((x, y, w, h))

# Draw the yellow bounding box for the chart area
if bar_tops:
    chart_top = min(bar_tops)
    chart_bottom = max(bar_bottoms)
    full_width = img_resized.shape[1]
    
    # Draw yellow bounding box around the chart
    cv2.rectangle(output, (0, chart_top), (full_width, chart_bottom), (0, 255, 255), 2)  # Yellow for the chart area

    red_bar_height = chart_bottom - chart_top

    score_lines.sort(key=lambda item: item[0])

    print("\nDetected Score Line Percentages (Left to Right):")
    for x, y, w, h in score_lines:
        score_center_y = y + h // 2
        percentage = ((chart_bottom - score_center_y) / red_bar_height) * 100
        percentage = round(percentage, 1)
        print(f"x={x}, y={score_center_y} → {percentage}%")
else:
    print("No valid vertical bars or score lines found.")

# Display the result with all horizontal lines and yellow bounding box
cv2.imshow("Bar Chart with All Horizontal Lines and Bounding Box", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
