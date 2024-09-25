import cv2
import numpy as np
import matplotlib.pyplot as plt

color_ranges = {
    "red": [(0, 50, 50), (10, 255, 255)],  # HSV range for red
    "blue": [(100, 150, 50), (140, 255, 255)],  # HSV range for blue
    "green": [(40, 50, 50), (80, 255, 255)],  # HSV range for green
    "yellow": [(20, 100, 100), (30, 255, 255)],  # HSV range for yellow
    "orange": [(10, 100, 100), (20, 255, 255)],  # HSV range for orange
    "white": [(0, 0, 200), (180, 30, 255)]  # HSV range for white
}

def load_image(image_path):
    image = cv2.imread(image_path)
    return image

def preprocess_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
    return blurred

def detect_color(hsv_image, color_ranges):
    color_grid = []
    for color_name, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype="uint8")
        upper_bound = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Add detected color and its location to the grid
                    color_grid.append(((cX, cY), color_name))

    return color_grid

def create_color_matrix(image, color_grid, grid_size=4):
    matrix = [["" for _ in range(grid_size)] for _ in range(grid_size)]
    height, width, _ = image.shape
    step_x = width // grid_size
    step_y = height // grid_size

    for (x, y), color_name in color_grid:
        grid_x = x // step_x
        grid_y = y // step_y

        if grid_x < grid_size and grid_y < grid_size:
            matrix[grid_y][grid_x] = color_name

    return matrix

def display_detected_colors(image, color_grid):
    output = image.copy()
    for (x, y), color_name in color_grid:
        cv2.circle(output, (x, y), 5, (255, 255, 255), -1)
        cv2.putText(output, color_name, (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def detect_puzzle_colors(image_path):
    image = load_image(image_path)
    hsv_image = preprocess_image(image)
    color_grid = detect_color(hsv_image, color_ranges)

    display_detected_colors(image, color_grid)
    matrix = create_color_matrix(image, color_grid)
    return matrix

# Example usage:
image_path = r"C:\Users\Asus\Documents\MATLAB\images\org_3.png"
color_matrix = detect_puzzle_colors(image_path)

# Print the resulting color matrix
for row in color_matrix:
    print(row)
