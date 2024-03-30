import csv
import random
from pytube import YouTube
import hashlib
import os
import cv2
import numpy as np

def distort(video_path):
    cap = cv2.VideoCapture(video_path)

    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Initialize an empty list to store the frames
    frames = []

    # Read the video frames for 5 seconds
    frame_count = 0
    while cap.isOpened() and frame_count < int(fps * 5):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    # Release the video capture object
    cap.release()

    # Convert the frames list to a 4D tensor
    video_tensor = np.array(frames)

    # Add Gaussian noise to every frame
    mean = 0
    std_dev = 50
    noise = np.random.normal(mean, std_dev, video_tensor.shape)
    noisy_video_tensor = video_tensor + noise
    noisy_video_tensor = np.clip(noisy_video_tensor, 0, 255).astype(np.uint8)

    # Add black dots and squares to every frame
    square_size = 20
    num_squares = 100

    for frame in noisy_video_tensor:
        for _ in range(num_squares):
            x = np.random.randint(0, frame_width - square_size)
            y = np.random.randint(0, frame_height - square_size)
            frame[y:y+square_size, x:x+square_size] = 0

    # Print the shape of the video tensor
    print("Video Tensor Shape:", video_tensor.shape)

    # Save the distorted video tensor to disk
    os.makedirs("videos/distorted/", exist_ok=True)
    output_path = f"videos/distorted/{video_path.split('/')[-1].split('.')[0]}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    for frame in noisy_video_tensor:
        out.write(frame)
    out.release()

# Read the CSV file and extract video IDs
video_ids = []
with open('howto100m.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        video_ids.append(row['video_id'])

# Randomly sample 2 video IDs
sampled_ids = random.sample(video_ids, 2)

# Download the sampled videos
for video_id in sampled_ids:
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    try:
        # Create a YouTube object
        video = YouTube(video_url)
        # Get the highest resolution stream
        stream = video.streams.get_lowest_resolution()
        
        # Download the video with a temporary filename
        temp_filename = str(random.randint(0, 2**64-1)) + ".mp4"
        print(f'Downloading video: {video.title}')
        stream.download(output_path='videos/original', filename=temp_filename)
        
        # Calculate the file hash
        with open(f'videos/original/{temp_filename}', 'rb') as file:
            file_hash = hashlib.sha256(file.read()).hexdigest()
        
        # Rename the video file with the file hash
        original_path = f'videos/original/{temp_filename}'
        new_path = f'videos/original/{file_hash}.mp4'
        os.rename(original_path, new_path)

        distort(new_path)
        
        print('Video downloaded successfully.')
    except Exception as e:
        print(f'Error downloading video: {video_url}')
        print(f'Error message: {str(e)}')