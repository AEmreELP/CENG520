import cv2
import numpy as np


def compute_histogram(image, bins=[8, 8, 8]):
    """
    Compute a normalized histogram for the provided image window.
    Assumes image in BGR and converts to HSV.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def histogram_distance(hist1, hist2, metric='L2'):
    """
    Compare two histograms using the specified metric.
    Supported metrics: 'L1', 'L2', 'Chi-Square', 'Symmetric-Chi-Square'
    """
    if metric == 'L1':
        return cv2.norm(hist1, hist2, normType=cv2.NORM_L1)
    elif metric == 'L2':
        return cv2.norm(hist1, hist2, normType=cv2.NORM_L2)
    elif metric == 'Chi-Square':
        # Using OpenCV's compareHist, note that a lower value means more similarity.
        return cv2.compareHist(hist1.astype('float32'), hist2.astype('float32'), cv2.HISTCMP_CHISQR)
    elif metric == 'Symmetric-Chi-Square':
        # Calculate a symmetric chi-square distance manually.
        eps = 1e-10
        return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))
    else:
        raise ValueError("Unsupported distance metric: " + metric)


def histogram_tracker(video_source, distance_metric='L2', search_scale=1.5, step=4):
    """
    Tracker function that performs a sliding window search in subsequent frames.

    Parameters:
    - video_source: Video file path or camera index.
    - distance_metric: Distance metric to use among {'L1', 'L2', 'Chi-Square', 'Symmetric-Chi-Square'}.
    - search_scale: A multiplier for expanding the search area around the previous location.
    - step: The step size for sliding window (in pixels).
    """
    cap = cv2.VideoCapture("Sample_Video.mp4")

    if not cap.isOpened():
        print("Error opening video source")
        return

    # Read the first frame and let the user choose ROI
    ret, frame = cap.read()
    if not ret:
        print("Error reading the first frame")
        return

    # Let user select ROI using OpenCV's ROI selector (x, y, w, h)
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = roi
    cv2.destroyWindow("Select ROI")
    ref_roi = frame[y:y + h, x:x + w]
    ref_hist = compute_histogram(ref_roi)

    # Define the initial location of ROI (as the center)
    current_roi = (x, y, w, h)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Unpack current ROI and define search window dimensions
        roi_x, roi_y, roi_w, roi_h = current_roi

        # expand search area by search_scale, ensure boundaries stay within image
        center_x = roi_x + roi_w // 2
        center_y = roi_y + roi_h // 2
        search_w = int(roi_w * search_scale)
        search_h = int(roi_h * search_scale)
        search_x = max(center_x - search_w // 2, 0)
        search_y = max(center_y - search_h // 2, 0)
        search_x_end = min(search_x + search_w, frame.shape[1])
        search_y_end = min(search_y + search_h, frame.shape[0])
        search_window = frame[search_y:search_y_end, search_x:search_x_end]

        best_distance = float("inf")
        best_position = None

        # Slide window within the search area
        for i in range(0, search_window.shape[0] - roi_h, step):
            for j in range(0, search_window.shape[1] - roi_w, step):
                candidate = search_window[i:i + roi_h, j:j + roi_w]
                candidate_hist = compute_histogram(candidate)
                dist = histogram_distance(ref_hist, candidate_hist, distance_metric)
                if dist < best_distance:
                    best_distance = dist
                    # convert candidate coordinates relative to full frame
                    best_position = (search_x + j, search_y + i, roi_w, roi_h)

        # Update the current location if best_position is found
        if best_position is not None:
            current_roi = best_position

        # Draw rectangle for the tracked ROI
        x_new, y_new, w_new, h_new = current_roi
        tracking_frame = frame.copy()
        cv2.rectangle(tracking_frame, (x_new, y_new), (x_new + w_new, y_new + h_new), (0, 255, 0), 2)
        cv2.imshow("Histogram Tracker", tracking_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Histogram based tracker on a video sequence")
    parser.add_argument("--video", type=str, default=0, help="Path to video file or camera index (default: 0)")
    parser.add_argument("--metric", type=str, default="L2", choices=["L1", "L2", "Chi-Square", "Symmetric-Chi-Square"],
                        help="Distance metric for histogram matching")
    parser.add_argument("--search_scale", type=float, default=1.5, help="Scale factor for the search window")
    parser.add_argument("--step", type=int, default=4, help="Step size in sliding window")

    args = parser.parse_args()

    # Support camera index conversion if video argument is digit-like
    video_source = args.video
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    histogram_tracker(video_source, distance_metric=args.metric, search_scale=args.search_scale, step=args.step)
