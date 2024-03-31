import os
import csv
import random
import hashlib
from pytube import YouTube
import cv2
import numpy as np
import tqdm

video_ids = [row['video_id'] for row in csv.DictReader(open('howto100m.csv', 'r'))]
sampled_ids = random.sample(video_ids, 1)

for video_id in sampled_ids:
    try:
        url = f'https://www.youtube.com/watch?v={video_id}'
        video = YouTube(url)
        temp_filename = str(random.randint(0, 2**64-1)) + ".mp4"
        print(f'Downloading video: {video.title} from {url}')
        video.streams.get_lowest_resolution().download(output_path='videos', filename=temp_filename)
        with open(f'videos/{temp_filename}', 'rb') as file:
            file_hash = hashlib.sha256(file.read()).hexdigest()
        original_path, new_path = f'videos/{temp_filename}', f'videos/{file_hash}.mp4'
        os.rename(original_path, new_path)
        print(f'Video downloaded successfully')
       
        # Read the video frames using OpenCV
        video_path = new_path
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
       
        # Convert frames to a NumPy array
        frames = np.array(frames)
       
        # Reshape the array to add frame groups dimension
        frame_group_size = 8
        num_frame_groups = len(frames) // frame_group_size
        frames = frames[:num_frame_groups * frame_group_size]  # Truncate frames to fit frame groups
        frames = frames.reshape(num_frame_groups, frame_group_size, frames.shape[1], frames.shape[2], frames.shape[3])
       
        # Cut each frame group into patches
        patch_size = 16
        height, width, _ = frames[0, 0].shape
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
       
        patches = np.zeros((num_frame_groups, num_patches_h, num_patches_w, frame_group_size, patch_size, patch_size, 3), dtype=np.uint8)
       
        for i in tqdm.tqdm(range(num_frame_groups)):
            for j in range(num_patches_h):
                for k in range(num_patches_w):
                    patches[i, j, k] = frames[i, :, j*patch_size:(j+1)*patch_size, k*patch_size:(k+1)*patch_size, :]
       
        # Save the patches as a .npz file
        os.makedirs("tensors", exist_ok=True)
        tensor_filename = f'{file_hash}_patches.npz'
        tensor_path = os.path.join('tensors', tensor_filename)
        np.savez(tensor_path, patches=patches)
        print(f'Tensor saved: {tensor_path} with shape {patches.shape}')
       
    except Exception as e:
        print(f'Error downloading or processing video: {video_id}\nError message: {str(e)}')