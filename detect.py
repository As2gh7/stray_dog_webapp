from uuid_tracker import process_video_file, process_webcam_stream

def detect_from_video(input_path, output_path, log_path):
    process_video_file(input_path, output_path, log_path)

def detect_from_webcam(output_path, log_path):
    process_webcam_stream(output_path, log_path)
