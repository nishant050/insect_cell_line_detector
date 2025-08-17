# Insect Cell Line Detector v2

This repository contains an Insect Detection Application built using Streamlit, OpenCV, and various machine learning models including the new YOLOv11n, Faster R-CNN, and HOG+SVM. Users can upload images, select detection models, and visualize results with bounding boxes. The app is designed for easily identifying insect cell lines for researchers.

## Features

-   Upload images in various formats (JPEG, PNG, TIFF, etc.)
-   Load ground truth data in YOLO format for comparison
-   Select from multiple detection models:
    -   **YOLOv11n (New)**: A new model trained on two classes.
    -   Faster R-CNN
    -   HOG + SVM
-   Adjust confidence and IoU thresholds for predictions
-   Visualize detection results with bounding boxes, with different colors for different classes from the YOLOv11n model.
-   Download detection results as an Excel file.
-   Version checking for the `ultralytics` library to ensure compatibility.

## Technologies Used

-   **Streamlit**: For creating the web interface
-   **OpenCV**: For image processing
-   **Ultralytics YOLO**: For object detection (v8.0.0 or higher recommended)
-   **Torchvision**: For Faster R-CNN model
-   **scikit-image**: For HOG feature extraction
-   **imutils**: For non-max suppression

## Installation

To run this project locally, follow these steps:

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/insect-detection-app.git](https://github.com/your-username/insect-detection-app.git)
    cd insect-detection-app
    ```

2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  Make sure you have the model weights in the root directory: `yolov11n.pt`, `frcnn.pth`, `model.npy`, and `scalar.npy`.

5.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Usage

1.  Open the app in your browser at `http://localhost:8501`.
2.  Select a model from the sidebar. The new YOLOv11n model is available.
3.  Upload an image of an insect.
4.  Optionally upload a text file containing ground truth bounding boxes in YOLO format.
5.  Adjust the confidence and IoU thresholds in the sidebar.
6.  The predictions will appear automatically. The results can be downloaded as an Excel file.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

-   [Streamlit](https://streamlit.io/)
-   [OpenCV](https://opencv.org/)
-   [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
-   [Torchvision](https://pytorch.org/vision/stable/index.html)
-   [scikit-image](https://scikit-image.org/)
