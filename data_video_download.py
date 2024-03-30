import os
import csv
import random
import hashlib
from pytube import YouTube
import cv2
import numpy as np

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
        frame_group_size = 64
        num_frame_groups = len(frames) // frame_group_size
        frames = frames[:num_frame_groups * frame_group_size]  # Truncate frames to fit frame groups
        frames = frames.reshape(num_frame_groups, frame_group_size, frames.shape[1], frames.shape[2], frames.shape[3])
        
        # Save each frame group as a separate MP4 file
        os.makedirs("frame_groups", exist_ok=True)
        for i, frame_group in enumerate(frames):
            frame_group_filename = f'{file_hash}_group_{i}.mp4'
            frame_group_path = os.path.join('frame_groups', frame_group_filename)
            
            # Create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30  # Adjust the frame rate as needed
            height, width, _ = frame_group[0].shape
            video_writer = cv2.VideoWriter(frame_group_path, fourcc, fps, (width, height))
            
            # Write frames to the video file
            for frame in frame_group:
                video_writer.write(frame)
            
            video_writer.release()

        print(f'Frame groups saved.')
        
    except Exception as e:
        print(f'Error downloading or processing video: {video_id}\nError message: {str(e)}')