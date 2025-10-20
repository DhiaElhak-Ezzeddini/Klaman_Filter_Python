import cv2
import numpy as np
from Detector import detect_inrange, detect_visage
from KalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Track(object):
    """
    A class to store information for each tracked object
    """
    track_id_counter = 0 # Global counter for unique track IDs
    
    def __init__(self, point, dt):
        self.id = Track.track_id_counter
        Track.track_id_counter += 1
        
        # Create a new Kalman Filter for this track
        self.kf = KalmanFilter(dt, point)
        
        # Frames since last update (for deleting lost tracks)
        self.age = 0
        
        # Total frames this track has been alive (for stable drawing)
        self.total_age = 0

# --- Configuration Constants ---
DT = 0.01             # Time step (can be tuned)
MAX_DISTANCE = 75    # Max distance (pixels) to associate a detection with a track
MAX_AGE = 20         # Max frames to keep a track without a detection
MIN_AGE_FOR_DRAW = 3 # Only draw tracks that are stable (at least 3 frames old)

# --- Main Tracking Logic ---
def main():
    VideoCap = cv2.VideoCapture(0)
    
    # List to store all active Track objects
    active_tracks = []
    Track.track_id_counter = 0 # Reset ID counter

    while True:
        ret, frame = VideoCap.read()
        if not ret:
            break

        # 1. DETECT
        # Get all detections (points) from the frame
        # detections, mask = detect_inrange(frame, 800)
        detections, mask = detect_visage(frame,use_profile=False)
        
        # --- PREDICT & ASSOCIATE ---
        
        predicted_positions = []
        track_indices_to_match = []
        
        # 2. PREDICT
        # Predict the next position for all existing tracks
        for i, track in enumerate(active_tracks):
            track.age += 1
            track.total_age += 1
            predicted_state = track.kf.predict()
            predicted_pos = (predicted_state[0, 0], predicted_state[1, 0])
            predicted_positions.append(np.array(predicted_pos))
            track_indices_to_match.append(i)

        # 3. ASSOCIATE
        # Match detections to predicted track positions
        
        matched_track_indices = set()
        matched_det_indices = set()
        
        if len(predicted_positions) > 0 and len(detections) > 0:
            # Calculate a cost matrix (e.g., Euclidean distance)
            # Rows = tracks, Cols = detections
            cost_matrix = np.zeros((len(predicted_positions), len(detections)))
            for i, track_pos in enumerate(predicted_positions):
                for j, det_pos in enumerate(detections):
                    dist = np.linalg.norm(track_pos - det_pos)
                    cost_matrix[i, j] = dist
            
            # Use the Hungarian algorithm to find the optimal assignment
            # It minimizes the total cost (distance)
            track_idx_map, det_idx = linear_sum_assignment(cost_matrix)
            
            # Filter out "bad" matches (where distance is > MAX_DISTANCE)
            for ti, di in zip(track_idx_map, det_idx):
                if cost_matrix[ti, di] < MAX_DISTANCE:
                    # This is a good match
                    original_track_index = track_indices_to_match[ti]
                    matched_track_indices.add(original_track_index)
                    matched_det_indices.add(di)

        # --- UPDATE LIFECYCLES ---
        
        # 4a. UPDATE Matched Tracks
        # These tracks were seen, so update them with the new detection
        for ti in matched_track_indices:
            # Find which detection this track was matched with
            # This is a bit complex, but finds the 'di' that corresponds to 'ti'
            ti_in_map = track_indices_to_match.index(ti)
            di_index_in_map = np.where(track_idx_map == ti_in_map)[0][0]
            di = det_idx[di_index_in_map]

            track = active_tracks[ti]
            track.age = 0 # Reset age since it was seen
            detection_point = np.expand_dims(detections[di], axis=-1)
            track.kf.update(detection_point)

        # 4b. CREATE New Tracks
        # These detections were not matched to any existing track
        unmatched_det_indices = set(range(len(detections))) - matched_det_indices
        for di in unmatched_det_indices:
            new_track = Track(detections[di], DT)
            active_tracks.append(new_track)
            
        # 4c. DELETE Old Tracks
        # These tracks were not matched and are too old
        # We iterate in reverse to avoid index issues when removing
        for i in sorted(range(len(active_tracks)), reverse=True):
            if active_tracks[i].age > MAX_AGE:
                del active_tracks[i]

        # --- 5. VISUALIZE ---
        
        # Draw current detections (red circles)
        for point in detections:
            cv2.circle(frame, (point[0], point[1]), 10, (0, 0, 255), 2)
            
        # Draw active tracks (green circles, lines, and IDs)
        for track in active_tracks:
            # Only draw stable tracks
            if track.total_age > MIN_AGE_FOR_DRAW:
                state = track.kf.E.astype(np.int32)
                pos = (state[0].item(), state[1].item())
                vel = (state[2].item(), state[3].item())

                # Draw circle at estimated position
                cv2.circle(frame, pos, 2, (0, 255, 0), 5)
                
                # Draw velocity vector
                cv2.arrowedLine(frame,
                                pos, (pos[0] + vel[0], pos[1] + vel[1]),
                                color=(0, 255, 0),
                                thickness=2,
                                tipLength=0.2)
                
                # Draw Track ID
                cv2.putText(frame, f"ID: {track.id}", (pos[0] + 10, pos[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frames
        cv2.imshow('image', frame)
        if mask is not None:
            cv2.imshow('mask', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    VideoCap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()