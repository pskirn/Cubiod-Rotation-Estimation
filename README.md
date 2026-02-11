# Cuboid Rotation Estimation Project

## 📋 Overview

This project implements an algorithm to estimate the rotation of a 3D cuboid from depth sensor data, as requested for the 10xConstruction technical assessment.

## 🎯 Deliverables

### ✅ Completed Items

1. **Python Script** (`cuboid_rotation_estimator.py`)

   - Well-commented implementation
   - ROS2 Humble compatible

2. **Results Table**

   - `results_table.csv` - CSV format with all measurements
   - Contains: Image number, normal angle, visible area, normal vectors

3. **Rotation Axis Vector**

   - `rotation_axis.csv` - CSV format
   - 3D unit vector in camera coordinates

4. **Algorithm Documentation** (`algorithm_documentation.md`)

   - Detailed approach explanation
   - Mathematical formulations
   - Implementation details

5. **Visualizations**
   - `analysis_visualization.png` - Comprehensive 4-panel analysis
   - `angle_evolution.png` - Temporal angle variation

## 🚀 Quick Start

### Prerequisites

````bash
# Install required packages
pip3 install -r requirements_compatible.txt

### Running the Code

```bash
# Source ROS2 (if using ROS2 bag files)
source /opt/ros/humble/setup.bash

# Run the estimation
python3 cuboid_rotation_estimator.py
````

## 📁 Project Structure

```
cuboid_project/
├── cuboid_rotation_estimator.py    # Main algorithm implementation
├── algorithm_documentation.md      # Detailed algorithm description
├── requirements_compatible.txt     # Python dependencies
├── depth/                          # ROS2 bag file (input)
├── results_table.csv               # Output: measurements table
├── rotation_axis.csv               # Output: rotation axis
├── analysis_visualization.png      # Output: visualization plots
├── angle_evaolution.png            # Output: visualization plots
└── README.md                       # This file
```

## 🔍 Algorithm Summary

### Approach

1. **Depth to Point Cloud**: Convert 2D depth images to 3D points using pinhole camera model
2. **Segmentation**: Isolate cuboid using distance-based clustering
3. **Face Detection**: RANSAC algorithm to find planar surfaces
4. **Area Calculation**: Convex hull method for visible face area
5. **Angle Computation**: Measure angle between face normal and camera
6. **Rotation Axis**: Estimate from normal vector history using cross-products

### Key Features

- Robust to sensor noise (RANSAC)
- No prior knowledge of cuboid dimensions required
- Automatic unit conversion (mm to m)
- Comprehensive error handling

## 🛠️ Technical Details

### Camera Parameters (Standard)

- Focal Length: fx=525, fy=525 pixels
- Principal Point: cx=319.5, cy=239.5 pixels

### Algorithm Parameters

- RANSAC iterations: 100
- Inlier threshold: 0.01m
- Minimum face points: 50
- Segmentation percentile: 75%

## 📈 Visualizations

The algorithm generates comprehensive visualizations including:

1. Normal angle evolution over time
2. Visible area changes per frame
3. Angle-area correlation analysis
4. 3D rotation axis visualization

## ⚙️ Compatibility

- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.10+
- **ROS**: ROS2 Humble (optional, for bag file reading)

## 🔧 Troubleshooting

If you encounter dependency issues:

```bash
# Fix NumPy compatibility
pip3 uninstall numpy opencv-python scikit-learn scipy
pip3 install numpy==1.24.3 scipy==1.10.1 scikit-learn==1.2.2 opencv-python==4.8.1.78
```

## 📝 Notes

- The algorithm assumes a single cuboid rotating around a fixed axis
- Depth sensor should remain stationary during capture
- At least one face should be visible in each frame
- Results are saved in both CSV and text formats for convenience

---

_Submission Date: 6th October 2025_
