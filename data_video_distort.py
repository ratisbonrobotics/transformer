import cv2
import numpy as np

# Open the video file
video_path = 'videos/original/c6b60ab690f3691dcb1362f75134d059b0fcd4783ee23b8c71d47e160dd46623.mp4'
cap = cv2.VideoCapture(video_path)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Calculate the number of frames for 10 seconds
num_frames_10s = int(fps * 5)

# Initialize an empty list to store the frames
frames = []

# Read the video frames for 10 seconds
frame_count = 0
while cap.isOpened() and frame_count < num_frames_10s:
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
std_dev = 100
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
output_path = 'videos/distorted/distorted_video_10s.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
for frame in noisy_video_tensor:
    out.write(frame)
out.release()