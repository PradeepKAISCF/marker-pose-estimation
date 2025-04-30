# marker-pose-estimation
Marker-based human pose estimation using a hybrid ResNet + Inception deep learning model with multi-output regression and Gaussian heatmaps.

# Marker-Based Human Pose Estimation 🎯

This project implements a real-time pose estimation system using a hybrid deep learning model (ResNet50 + Inception) to predict 9 human joint keypoints from images with physical markers.

## 🔧 Features
- Custom dataset creation with OpenCV (HSV masking + contour detection)
- Hybrid CNN model with ResNet + Inception modules
- Multi-output regression using Huber Loss and MAE monitoring
- Real-time visualization with keypoint overlays and Gaussian heatmaps
- Training/validation loss plots per joint for detailed analysis

## 🖼️ Sample Output

<p align="center">
  <img src="outputs/sample_heatmap.png" width="400"/>
  <img src="outputs/keypoints_overlay.png" width="400"/>
</p>

## 📁 Project Structure

├── train.py ├── predict.py ├── models/ ├── utils/ └── data/
