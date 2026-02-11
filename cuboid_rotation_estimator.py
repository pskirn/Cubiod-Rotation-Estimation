#!/usr/bin/env python3
"""
Cuboid Rotation Estimation from Depth Images - ROS2 Humble Version

This script processes depth images from a ROS2 bag file to:
1. Estimate normal angle and visible area of the largest face
2. Determine the axis of rotation of a rotating cuboid
"""

import numpy as np
import cv2
import rclpy
import rosbag2_py
import os
import sys
import csv
import matplotlib
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge



class CuboidRotationEstimator:
    def __init__(self, bag_file_path: str):
        """
        Initialize the cuboid rotation estimator.
        
        Args:
            bag_file_path: Path to the ROS2 bag file containing depth images
        """
        self.bag_file = bag_file_path
        self.bridge = CvBridge()
        
        # Camera parameters (standard values for depth cameras)
        self.fx = 525.0  # Focal length x (pixels)
        self.fy = 525.0  # Focal length y (pixels)
        self.cx = 319.5  # Principal point x (pixels)
        self.cy = 239.5  # Principal point y (pixels)
        
        # Storage for results
        self.results = []
        self.rotation_data = []
        
    def depth_to_point_cloud(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Convert depth image to 3D point cloud.
        
        Args:
            depth_image: 2D array of depth values in meters
            
        Returns:
            Nx3 array of 3D points
        """
        height, width = depth_image.shape
        
        # Create mesh grid for pixel coordinates
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        
        # Filter out invalid depth values
        valid_mask = (depth_image > 0.1) & (depth_image < 10.0)
        
        # Convert to 3D points using pinhole camera model
        z = depth_image[valid_mask]
        x = (xx[valid_mask] - self.cx) * z / self.fx
        y = (yy[valid_mask] - self.cy) * z / self.fy
        
        points = np.stack([x, y, z], axis=-1)
        return points
    
    def segment_cuboid(self, points: np.ndarray) -> np.ndarray:
        """
        Segment the cuboid from the background using depth clustering.
        
        Args:
            points: Nx3 array of 3D points
            
        Returns:
            Filtered points belonging to the cuboid
        """
        if len(points) < 100:
            return points
        
        # Remove background points (far from center of mass)
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        
        # Use percentile-based thresholding
        threshold = np.percentile(distances, 75)
        mask = distances < threshold
        
        return points[mask]
    
    def find_planar_faces(self, points: np.ndarray, num_faces: int = 3) -> List[Dict]:
        """
        Find planar faces in the point cloud using RANSAC.
        
        Args:
            points: Nx3 array of 3D points
            num_faces: Maximum number of faces to detect
            
        Returns:
            List of detected faces with normal vectors and areas
        """
        faces = []
        remaining_points = points.copy()
        
        for _ in range(num_faces):
            if len(remaining_points) < 100:
                break
                
            # RANSAC plane fitting
            best_plane = self.ransac_plane_fit(remaining_points)
            
            if best_plane is None:
                break
                
            normal, d, inliers = best_plane
            
            # Extract face points
            face_points = remaining_points[inliers]
            
            # Calculate visible area using convex hull
            if len(face_points) > 3:
                # Project points to 2D plane for area calculation
                area = self.calculate_face_area(face_points, normal)
                
                faces.append({
                    'normal': normal,
                    'area': area,
                    'points': face_points,
                    'center': np.mean(face_points, axis=0)
                })
            
            # Remove inliers for next iteration
            remaining_points = remaining_points[~inliers]
        
        return faces
    
    def ransac_plane_fit(self, points: np.ndarray, iterations: int = 100, threshold: float = 0.01) -> Tuple:
        """
        RANSAC algorithm for plane fitting.
        
        Args:
            points: Nx3 array of 3D points
            iterations: Number of RANSAC iterations
            threshold: Distance threshold for inliers
            
        Returns:
            Tuple of (normal vector, d, inlier mask) or None
        """
        n_points = len(points)
        if n_points < 3:
            return None
            
        best_inliers = None
        best_count = 0
        
        for _ in range(iterations):
            # Randomly sample 3 points
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]
            
            # Calculate plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            if np.linalg.norm(normal) < 1e-6:
                continue
                
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, p1)
            
            # Calculate distances to plane
            distances = np.abs(np.dot(points, normal) + d)
            inliers = distances < threshold
            count = np.sum(inliers)
            
            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_normal = normal
                best_d = d
        
        if best_inliers is None or best_count < 50:
            return None
            
        return best_normal, best_d, best_inliers
    
    def calculate_face_area(self, points: np.ndarray, normal: np.ndarray) -> float:
        """
        Calculate the visible area of a face.
        
        Args:
            points: Points on the face
            normal: Normal vector of the face
            
        Returns:
            Area in square meters
        """
        # Project points onto a 2D plane perpendicular to the normal
        # Create orthonormal basis
        if abs(normal[2]) < 0.9:
            u = np.cross(normal, [0, 0, 1])
        else:
            u = np.cross(normal, [1, 0, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        # Project points to 2D
        points_2d = np.column_stack([
            np.dot(points, u),
            np.dot(points, v)
        ])
        
        # Calculate convex hull area
        if len(points_2d) >= 3:
            hull = cv2.convexHull(points_2d.astype(np.float32))
            area = cv2.contourArea(hull)
            return abs(area)
        
        return 0.0
    
    def calculate_normal_angle(self, normal: np.ndarray) -> float:
        """
        Calculate angle between face normal and camera normal (z-axis).
        
        Args:
            normal: Normal vector of the face
            
        Returns:
            Angle in degrees
        """
        camera_normal = np.array([0, 0, -1])  # Camera looks in -z direction
        
        # Ensure normal points towards camera
        if np.dot(normal, camera_normal) > 0:
            normal = -normal
        
        # Calculate angle
        cos_angle = np.dot(normal, camera_normal) / (np.linalg.norm(normal) * np.linalg.norm(camera_normal))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(abs(cos_angle)))
        
        return angle
    
    def estimate_rotation_axis(self, normals_history: List[np.ndarray]) -> np.ndarray:
        """
        Estimate the axis of rotation from face normal history.
        
        Args:
            normals_history: List of normal vectors over time
            
        Returns:
            Estimated rotation axis vector
        """
        if len(normals_history) < 3:
            return np.array([0, 1, 0])  # Default to y-axis
        
        # Method 1: Use cross products of consecutive normals
        axes = []
        for i in range(len(normals_history) - 1):
            n1 = normals_history[i]
            n2 = normals_history[i + 1]
            
            # Skip if normals are too similar
            if np.linalg.norm(n1 - n2) < 0.1:
                continue
                
            axis = np.cross(n1, n2)
            if np.linalg.norm(axis) > 1e-6:
                axis = axis / np.linalg.norm(axis)
                axes.append(axis)
        
        if axes:
            # Average the axes
            avg_axis = np.mean(axes, axis=0)
            avg_axis = avg_axis / np.linalg.norm(avg_axis)
            return avg_axis
        
        # Method 2: PCA on normal vectors
        pca = PCA(n_components=3)
        pca.fit(normals_history)
        
        # The axis with smallest variance is likely the rotation axis
        return pca.components_[2]
    
    def process_ros2_bag(self):
        """
        Process depth images from a ROS2 bag file.
        """
        print(f"Processing ROS2 bag file: {self.bag_file}")
        
        bag_path = self.bag_file

        # Create reader
        reader = rosbag2_py.SequentialReader()
        
        # Configure storage
        storage_options = rosbag2_py.StorageOptions(
            uri=bag_path,
            storage_id='sqlite3'
        )
        
        # Configure converter
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        # Open bag
        reader.open(storage_options, converter_options)
        
        # Get topic information
        topic_types = reader.get_all_topics_and_types()
        
        # Find depth image topic
        depth_topic = None
        type_map = {}
        
        for topic_info in topic_types:
            topic_name = topic_info.name
            topic_type = topic_info.type
            type_map[topic_name] = topic_type
            
            if 'depth' in topic_name.lower() or 'image' in topic_name.lower():
                depth_topic = topic_name
                print(f"Found depth topic: {topic_name} with type: {topic_type}")
        
        if not depth_topic:
            print("Warning: No depth topic found. Using first available topic.")
            if topic_types:
                depth_topic = topic_types[0].name
            else:
                print("Error: No topics found in bag file")
                return
        
        print(f"Using topic: {depth_topic}")
        
        # Process messages
        frame_count = 0
        normals_history = []
        
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            
            if topic != depth_topic:
                continue
            
            frame_count += 1
            
            # Get message type and deserialize
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            
            try:
                # Convert message to depth image
                if hasattr(msg, 'encoding'):
                    # Standard Image message
                    if 'compressed' in type_map[topic].lower():
                        # Compressed image
                        np_arr = np.frombuffer(msg.data, np.uint8)
                        depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    else:
                        # Raw image
                        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                else:
                    # Handle raw data
                    depth_image = np.frombuffer(msg.data, dtype=np.float32).reshape((480, 640))
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue
            
            # Convert to meters if necessary (some sensors provide mm)
            if np.median(depth_image[depth_image > 0]) > 100:
                depth_image = depth_image / 1000.0
            
            # Process depth image
            points = self.depth_to_point_cloud(depth_image)
            
            if len(points) < 100:
                print(f"Frame {frame_count}: Insufficient points")
                continue
            
            # Segment cuboid
            cuboid_points = self.segment_cuboid(points)
            
            # Find planar faces
            faces = self.find_planar_faces(cuboid_points)
            
            if not faces:
                print(f"Frame {frame_count}: No faces detected")
                continue
            
            # Find largest visible face
            largest_face = max(faces, key=lambda f: f['area'])
            
            # Calculate normal angle
            angle = self.calculate_normal_angle(largest_face['normal'])
            
            # Store results
            self.results.append({
                'frame': frame_count,
                'timestamp': timestamp / 1e9,  # Convert to seconds
                'angle': angle,
                'area': largest_face['area'],
                'normal': largest_face['normal']
            })
            
            normals_history.append(largest_face['normal'])
            
            print(f"Frame {frame_count}: Angle={angle:.2f}deg, Area={largest_face['area']:.4f} m^2")
        
        # Estimate rotation axis
        if normals_history:
            self.rotation_axis = self.estimate_rotation_axis(normals_history)
            print(f"\nEstimated rotation axis: {self.rotation_axis}")
        else:
            self.rotation_axis = np.array([0, 1, 0])  # Default
    
    
    
    def save_results(self):
        """
        Save results to required output files in multiple formats.
        """
        
        # Save results as CSV file
        with open('results_table.csv', 'w', newline='') as csvfile:
            fieldnames = ['Image_Number', 'Timestamp', 'Normal_Angle_degrees', 'Visible_Area_m2', 
                         'Normal_X', 'Normal_Y', 'Normal_Z']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'Image_Number': result['frame'],
                    'Timestamp': f"{result['timestamp']:.6f}",
                    'Normal_Angle_degrees': f"{result['angle']:.2f}",
                    'Visible_Area_m2': f"{result['area']:.6f}",
                    'Normal_X': f"{result['normal'][0]:.6f}",
                    'Normal_Y': f"{result['normal'][1]:.6f}",
                    'Normal_Z': f"{result['normal'][2]:.6f}"
                })
        
        
        # Save rotation axis as Text file
        with open('rotation_axis.txt', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Axis_Component', 'Value'])
            writer.writerow(['x', f"{self.rotation_axis[0]:.6f}"])
            writer.writerow(['y', f"{self.rotation_axis[1]:.6f}"])
            writer.writerow(['z', f"{self.rotation_axis[2]:.6f}"])
        
        print("\nResults saved to:")
        print("  - results_table.csv (CSV format)")
        print("  - rotation_axis.txt (Text file format)")
    
    def plot_results(self):
        """
        Create comprehensive visualizations of the results.
        """
        if not self.results:
            return
        
        frames = [r['frame'] for r in self.results]
        angles = [r['angle'] for r in self.results]
        areas = [r['area'] for r in self.results]
        timestamps = [r['timestamp'] for r in self.results]
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Subplot 1: Normal angle over time
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(frames, angles, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Frame Number', fontsize=12)
        ax1.set_ylabel('Normal Angle (degrees)', fontsize=12)
        ax1.set_title('Face Normal Angle vs Frame', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([min(frames)-1, max(frames)+1])
        
        # Add angle statistics
        ax1.text(0.02, 0.98, f'Mean: {np.mean(angles):.2f}°\nStd: {np.std(angles):.2f}°',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Subplot 2: Visible area over time
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(frames, areas, 'r-s', linewidth=2, markersize=6)
        ax2.set_xlabel('Frame Number', fontsize=12)
        ax2.set_ylabel('Visible Area (m²)', fontsize=12)
        ax2.set_title('Largest Face Area vs Frame', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([min(frames)-1, max(frames)+1])
        
        # Add area statistics
        ax2.text(0.02, 0.98, f'Mean: {np.mean(areas):.4f} m²\nStd: {np.std(areas):.4f} m²',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Subplot 3: Angle vs Area correlation
        ax3 = plt.subplot(2, 2, 3)
        scatter = ax3.scatter(angles, areas, c=frames, cmap='viridis', s=50)
        ax3.set_xlabel('Normal Angle (degrees)', fontsize=12)
        ax3.set_ylabel('Visible Area (m²)', fontsize=12)
        ax3.set_title('Angle vs Area Correlation', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for frame numbers
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Frame Number', fontsize=10)
        
        # Calculate and display correlation
        correlation = np.corrcoef(angles, areas)[0, 1]
        ax3.text(0.02, 0.98, f'Correlation: {correlation:.3f}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # Subplot 4: 3D visualization of rotation axis
        ax4 = plt.subplot(2, 2, 4, projection='3d')
        
        # Plot rotation axis
        axis_length = 1.5
        ax4.quiver(0, 0, 0, 
                  self.rotation_axis[0] * axis_length,
                  self.rotation_axis[1] * axis_length,
                  self.rotation_axis[2] * axis_length,
                  color='red', arrow_length_ratio=0.1, linewidth=3, label='Rotation Axis')
        
        # Plot normal vectors from some frames
        sample_indices = np.linspace(0, len(self.results)-1, min(10, len(self.results)), dtype=int)
        for idx in sample_indices:
            normal = self.results[idx]['normal']
            ax4.quiver(0, 0, 0, normal[0], normal[1], normal[2],
                      color='blue', alpha=0.3, arrow_length_ratio=0.05)
        
        # Set labels and limits
        ax4.set_xlabel('X', fontsize=10)
        ax4.set_ylabel('Y', fontsize=10)
        ax4.set_zlabel('Z', fontsize=10)
        ax4.set_title('Rotation Axis & Face Normals', fontsize=14, fontweight='bold')
        ax4.set_xlim([-1.5, 1.5])
        ax4.set_ylim([-1.5, 1.5])
        ax4.set_zlim([-1.5, 1.5])
        ax4.legend()
        
        # Add main title
        fig.suptitle('Cuboid Rotation Analysis Results', fontsize=16, fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('analysis_visualization.png', dpi=150, bbox_inches='tight')
        
        print("\nVisualizations saved to:")
        print("  - analysis_visualization.png")
        
        # Create additional detailed plot for angles only
        fig2, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timestamps, angles, 'b-', linewidth=2, label='Normal Angle')
        ax.fill_between(timestamps, angles, alpha=0.3)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Normal Angle (degrees)', fontsize=12)
        ax.set_title('Normal Angle Evolution Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Mark peaks
        peaks = []
        for i in range(1, len(angles)-1):
            if angles[i] > angles[i-1] and angles[i] > angles[i+1]:
                peaks.append(i)
                ax.plot(timestamps[i], angles[i], 'ro', markersize=8)
        
        plt.savefig('angle_evolution.png', dpi=150, bbox_inches='tight')
        print("  - angle_evolution.png")

def main():
    """
    Main function to run the cuboid rotation estimation.
    """
    # Path to your ROS bag file
    bag_file = "/home/sk/cuboid_project/depth"  # Update this path as needed
    
    print("CUBOID ROTATION ESTIMATION - ROS2 Humble Version")
    print("="*60)
    
    # Create estimator and process
    estimator = CuboidRotationEstimator(bag_file)
    estimator.process_ros2_bag()
    
    if estimator.results:
        estimator.save_results()
        estimator.plot_results()
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE - SUMMARY")
        print("="*60)
        print(f"Total frames processed: {len(estimator.results)}")
        print(f"Estimated rotation axis: [{estimator.rotation_axis[0]:.4f}, {estimator.rotation_axis[1]:.4f}, {estimator.rotation_axis[2]:.4f}]")
        
        # Calculate statistics
        angles = [r['angle'] for r in estimator.results]
        areas = [r['area'] for r in estimator.results]
        print(f"\nAngle Statistics:")
        print(f"  Mean angle: {np.mean(angles):.2f}deg")
        print(f"  Min angle:  {np.min(angles):.2f}deg")
        print(f"  Max angle:  {np.max(angles):.2f}deg")
        
        print(f"\nArea Statistics:")
        print(f"  Mean area: {np.mean(areas):.4f} m²")
        print(f"  Min area:  {np.min(areas):.4f} m²")
        print(f"  Max area:  {np.max(areas):.4f} m²")
        
        print("\n" + "="*60)
        print("All deliverables generated successfully!")
        print("="*60)
    else:
        print("\nNo frames could be processed. Please check your bag file.")

if __name__ == "__main__":
    main()
