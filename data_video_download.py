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
        os.makedirs("patches", exist_ok=True)
        for i, frame_group in tqdm.tqdm(enumerate(frames)):
            height, width, _ = frame_group[0].shape
            num_patches_h = height // patch_size
            num_patches_w = width // patch_size
           
            for j in range(num_patches_h):
                for k in range(num_patches_w):
                    patch = frame_group[:, j*patch_size:(j+1)*patch_size, k*patch_size:(k+1)*patch_size, :]
                    patch_filename = f'{file_hash}_group_{i}_patch_{j}_{k}.mp4'
                    patch_path = os.path.join('patches', patch_filename)
                   
                    # Create a VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 30  # Adjust the frame rate as needed
                    video_writer = cv2.VideoWriter(patch_path, fourcc, fps, (patch_size, patch_size))
                   
                    # Write frames to the video file
                    for frame in patch:
                        video_writer.write(frame)
                   
                    video_writer.release()
        print(f'Patches saved.')
       
    except Exception as e:
        print(f'Error downloading or processing video: {video_id}\nError message: {str(e)}')