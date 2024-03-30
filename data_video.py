import csv
import random
from pytube import YouTube

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

        # Download the video
        print(f'Downloading video: {video.title}')
        stream.download(output_path='videos')
        print('Video downloaded successfully.')
    except Exception as e:
        print(f'Error downloading video: {video_url}')
        print(f'Error message: {str(e)}')