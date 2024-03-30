import os
import csv
import random
import hashlib
from pytube import YouTube

video_ids = [row['video_id'] for row in csv.DictReader(open('howto100m.csv', 'r'))]
sampled_ids = random.sample(video_ids, 2)
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
    except Exception as e:
        print(f'Error downloading video: {video_id}\nError message: {str(e)}')