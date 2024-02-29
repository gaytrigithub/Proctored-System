import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pyaudio

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    parser.add_argument("--camera-indexes", nargs="+", type=int, default=[0, 1, 2])  # Specify all camera indexes
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Initialize VideoCapture objects for all cameras
    cameras = [cv2.VideoCapture(index) for index in args.camera_indexes]
    for cap in cameras:
        if not cap.isOpened():
            print(f"Error: Failed to open camera with index {index}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    # Initialize audio stream
    p = pyaudio.PyAudio()
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    try:
        while True:
            frames = [cap.read()[1] for cap in cameras]
            audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
            for frame in frames:
                result = model(frame, agnostic_nms=True)[0]
                detections = sv.Detections.from_yolov8(result)
                labels = [
                    f"{model.model.names[class_id]} {confidence:0.2f}"
                    for _, confidence, class_id, _
                    in detections
                ]
                frame = box_annotator.annotate(
                    scene=frame, 
                    detections=detections, 
                    labels=labels
                )

                zone.trigger(detections=detections)
                frame = zone_annotator.annotate(scene=frame)  

                cv2.imshow("yolov8", frame)

            if cv2.waitKey(30) == 27:
                break
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
    finally:
        # Release resources
        for cap in cameras:
            cap.release()
        cv2.destroyAllWindows()
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
