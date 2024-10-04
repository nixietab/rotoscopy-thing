import cv2
import numpy as np
from skimage import exposure
from moviepy.editor import VideoFileClip
from tqdm import tqdm  # Import tqdm for the progress bar

# Function to enhance contrast
def enhance_contrast(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast enhancement
    enhanced = exposure.rescale_intensity(gray, in_range=(50, 200))
    
    return enhanced

# Function to detect edges and create mask
def get_edge_mask(frame):
    # Use Canny edge detection
    edges = cv2.Canny(frame, 100, 200)
    
    # Dilate edges to make them thicker
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    return edges

# Main function to process video
def process_video(input_path, output_video_path, final_output_path):
    # OpenCV processing part
    cap = cv2.VideoCapture(input_path)
    
    # Get the original frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get frame size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize the video writer with the same frame rate as the input video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Set up the progress bar
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enhance contrast
            contrast_frame = enhance_contrast(frame)
            
            # Detect edges
            edge_mask = get_edge_mask(contrast_frame)
            
            # Convert edge mask to 3-channel for masking original frame
            edge_mask_3ch = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
            
            # Apply the edge mask on the original frame (rotoscope effect)
            rotoscoped_frame = cv2.bitwise_and(frame, edge_mask_3ch)
            
            # Write the frame to output video
            out.write(rotoscoped_frame)
            
            # Update the progress bar
            pbar.update(1)
    
    # Release video objects
    cap.release()
    out.release()
    
    # Now merge the original audio back using moviepy
    merge_audio(input_path, output_video_path, final_output_path)

# Function to merge original audio with processed video
def merge_audio(original_video_path, processed_video_path, output_video_with_audio):
    # Load the original video
    original_clip = VideoFileClip(original_video_path)
    
    # Load the processed video (without audio)
    processed_clip = VideoFileClip(processed_video_path)
    
    # Set the audio of the processed video to the original audio
    final_clip = processed_clip.set_audio(original_clip.audio)
    
    # Write the final output with audio
    final_clip.write_videofile(output_video_with_audio, codec="libx264", audio_codec="aac")

# Run the function with your video file
input_video_path = 'input.mp4'   # Change this to your input video file path
output_video_path = 'rotoscoped_silent.mp4'  # Temporary video without audio
final_output_path = 'rotoscoped_with_audio.mp4'  # Final video with audio

process_video(input_video_path, output_video_path, final_output_path)
