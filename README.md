# Insect cell line detector
This repository contains an Insect Detection Application built using Streamlit, OpenCV, and various machine learning models including YOLOv8, Faster R-CNN and HOG+SVM. Users can upload images, select detection models, and visualize results with bounding boxes. The app is designed for easily identifying insects cell line for researchers.



# Insect Detection Model Interface

This project is an Insect Detection application built using Streamlit, OpenCV, and several machine learning models including YOLOv8, Faster R-CNN, and a custom HOG + SVM detector. The app allows users to upload images and detect insects within them, visualizing the results with bounding boxes for both predictions and ground truth data.

## Features

- Upload images in various formats (JPEG, PNG, TIFF, etc.)
- Load ground truth data in YOLO format for comparison
- Select from multiple detection models:
  - YOLOv8
  - Faster R-CNN
  - HOG + SVM
- Adjust confidence and IoU thresholds for predictions
- Visualize detection results with bounding boxes

## Technologies Used

- **Streamlit**: For creating the web interface
- **OpenCV**: For image processing
- **Ultralytics YOLO**: For object detection
- **Torchvision**: For Faster R-CNN model
- **scikit-image**: For HOG feature extraction
- **imutils**: For non-max suppression

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/insect-detection-app.git
   cd insect-detection-app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the app in your browser at `http://localhost:8501`.
2. Upload an image of an insect.
3. Optionally upload a text file containing ground truth bounding boxes in YOLO format.
4. Select the desired detection model and adjust the confidence and IoU thresholds.
5. Click "Make Predictions" to see the results.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [scikit-image](https://scikit-image.org/)
