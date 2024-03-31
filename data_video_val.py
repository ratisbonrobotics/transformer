import os
import numpy as np
import cv2

# Load the video tensor from the .npz file
tensor_filename = 'your_tensor_filename.npz'  # Replace with the actual tensor filename
tensor_path = os.path.join('tensors', tensor_filename)
tensor_data = np.load(tensor_path)
patches = tensor_data['patches']

# Get the dimensions of the reconstructed video
num_frame_groups, num_patches_h, num_patches_w, frame_group_size, patch_size, _, _ = patches.shape
height = num_patches_h * patch_size
width = num_patches_w * patch_size

# Create a VideoWriter object to save the reconstructed video
output_filename = 'reconstructed_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30  # Adjust the frames per second as needed
video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# Reconstruct the video frames from patches
for i in range(num_frame_groups):
    for j in range(frame_group_size):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for k in range(num_patches_h):
            for l in range(num_patches_w):
                frame[k*patch_size:(k+1)*patch_size, l*patch_size:(l+1)*patch_size, :] = patches[i, k, l, j]
        video_writer.write(frame)

# Release the VideoWriter object
video_writer.release()

print(f'Reconstructed video saved: {output_filename}')