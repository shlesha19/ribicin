import cv2

def remove_frame_and_save_png(input_video_path, output_video_path, timestamp_seconds):
    # 1. Open the source video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open the input video '{input_video_path}'.")
        print("Make sure the video file is in the exact same folder as this script.")
        return

    # 2. Extract video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate the exact frame index corresponding to 0:38
    target_frame_index = int(timestamp_seconds * fps)
    print(f"Video FPS: {fps}")
    print(f"Targeting frame index {target_frame_index} at {timestamp_seconds} seconds.")

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_index = 0

    # 3. Process the video stream frame by frame
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        # 4. Handle the target frame at 0:38
        if frame_index == target_frame_index:
            # Save the frame as a PNG image file
            cv2.imwrite('extracted_frame.png', frame)
            print(f"-> Saved frame {frame_index} as 'extracted_frame.png'")
            print(f"-> Skipped frame {frame_index} from the output video.")
            
            frame_index += 1
            continue  # Skip writing this frame to the output video

        # Write all other frames to the new video file
        out.write(frame)
        frame_index += 1

    # 5. Clean up and close all files
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\nProcessing finished successfully!")
    print(f"New video saved as: '{output_video_path}'")
    print("Extracted image saved as: 'extracted_frame.png'")

# --- Run the function ---
# 38 seconds corresponds to exactly 0:38
remove_frame_and_save_png('clip1.mp4', 'clip1_edited.mp4', 38)
