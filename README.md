## Car Segmentation and Feature Analysis Pipeline

This project provides a comprehensive tool for detecting, enhancing, and analyzing cars within images or video streams. It uses advanced computer vision techniques to separate cars from their backgrounds, identify specific parts like wheels and lights, and even swap components between different vehicles for analysis.

---

## Features

*   **Video Keyframe Extraction**: Automatically scans video files for moving vehicles and captures high-quality frames for processing.
*   **Intelligent Background Removal**: Analyzes the complexity of a scene to decide the best way to cut out the car, whether it’s a simple solid background or a complex city street.
*   **Dual-Domain Enhancement**: Combines lighting adjustments (Spatial Domain) with edge sharpening (Frequency Domain) to make details pop.
*   **Part Identification**: Detects and extracts specific car components, including:
    *   Wheels and tires (using circular detection).
    *   Windows and windshields (using texture and position analysis).
    *   Headlights and taillights (using color and brightness filtering).
*   **Part Swapping**: Reconstructs cars using parts from other vehicles to test visual compatibility or analysis.
*   **Quality Metrics**: Calculates mathematical scores (PSNR, SSIM, MSE) to measure how well the software preserves image quality during processing.
*   **Feature Matching**: Uses SIFT (Scale-Invariant Feature Transform) to visually compare parts between two different cars.

---

## Prerequisites

To run this code, you will need Python installed along with the following libraries:

*   **OpenCV** (`cv2`): For core image and video processing.
*   **NumPy**: For mathematical operations.
*   **Matplotlib**: For generating the technical analysis plots.
*   **Scikit-Image** (`skimage`): For structural similarity metrics and texture analysis.

You can install these via pip:
`pip install opencv-python numpy matplotlib scikit-image`

---

## How to Use

1.  **Prepare your files**: Place your car image (e.g., `cars1.jpg`) and a replacement background (e.g., `new_background.jpg`) in the same folder as the script.
2.  **Configure the mode**: In the `if __name__ == "__main__":` section at the bottom of the script:
    *   Set `VIDEO_MODE = True` if you are pointing to a video file.
    *   Set `VIDEO_MODE = False` for a single image.
3.  **Run the script**: Execute the python file. 
4.  **Check the output**: 
    *   The script will create a folder named `Extracted_Car_Data`.
    *   Inside, you will find subfolders for each car containing the individual parts.
    *   A final technical grid (`Final_Technical_Grid.png`) will be generated showing the step-by-step math and results.

---

## How it Works

The pipeline follows a specific order of operations:

1.  **Detection**: It finds the "core" of the car to determine if it is facing head-on or side-on.
2.  **Masking**: It creates a silhouette of the car. For complex backgrounds, it uses a combination of GrabCut and Watershed algorithms.
3.  **Enhancement**: It cleans up the "noise" in the paint while simultaneously sharpening the mechanical edges.
4.  **Extraction**: It slices the car into its parent chassis and child parts (wheels, windows, lights).
5.  **Comparison**: It uses feature matching to find similar textures between different parts and saves a visual report.
