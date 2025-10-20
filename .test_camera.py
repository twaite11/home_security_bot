import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Find and enable the RGB camera stream
# Note: For some RealSense models, you might need to specify the device serial number
# or use a different resolution if 640x480 is not supported.
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("[INFO] Starting camera stream...")
# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Display the resulting frame
        cv2.imshow('RealSense Camera Test', color_image)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    print("[INFO] Stopping camera stream.")
    pipeline.stop()
    cv2.destroyAllWindows()

