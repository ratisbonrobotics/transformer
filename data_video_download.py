import csv
import random
from pytube import YouTube
import hashlib
import os
import cv2
import numpy as np

def distort(video_path):
    cap = cv2.VideoCapture(video_path)
    fps, num_frames, frame_height, frame_width = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = [cap.read()[1] for _ in range(int(fps * 5)) if cap.isOpened()]
    cap.release()
    video_tensor = np.array(frames)
    noisy_video_tensor = np.clip(video_tensor + np.random.normal(0, 50, video_tensor.shape), 0, 255).astype(np.uint8)
    for frame in noisy_video_tensor:
        for _ in range(100):
            x, y = np.random.randint(0, frame_width - 20), np.random.randint(0, frame_height - 20)
            frame[y:y+20, x:x+20] = 0
    os.makedirs("videos/distorted/", exist_ok=True)
    output_path = f"videos/distorted/{video_path.split('/')[-1].split('.')[0]}.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    for frame in noisy_video_tensor:
        out.write(frame)
    out.release()

video_ids = [row['video_id'] for row in csv.DictReader(open('howto100m.csv', 'r'))]
sampled_ids = random.sample(video_ids, 2)

for video_id in sampled_ids:
    try:
        video = YouTube(f'https://www.youtube.com/watch?v={video_id}')
        temp_filename = str(random.randint(0, 2**64-1)) + ".mp4"
        print(f'Downloading video: {video.title}')
        video.streams.get_lowest_resolution().download(output_path='videos/original', filename=temp_filename)
        with open(f'videos/original/{temp_filename}', 'rb') as file:
            file_hash = hashlib.sha256(file.read()).hexdigest()
        original_path, new_path = f'videos/original/{temp_filename}', f'videos/original/{file_hash}.mp4'
        os.rename(original_path, new_path)
        distort(new_path)
        print('Video downloaded successfully.')
    except Exception as e:
        print(f'Error downloading video: {video_id}\nError message: {str(e)}')