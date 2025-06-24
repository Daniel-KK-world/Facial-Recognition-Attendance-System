import face_recognition
import queue 
import threading 
import cv2 


class FaceProcessor:
    """Optimized but reliable face processing"""
    def __init__(self, attendance_system):
        self.attendance_system = attendance_system
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = False
        self.process_thread = None
        self.last_locations = []
        self.last_encodings = []
        
        # Tune these for your hardware
        self.downscale_factor = 0.3  # 30% of original size
        self.detection_every_n_frames = 20  # Process every other frame
        self.frame_counter = 0

    def start(self):
        self.running = True
        self.process_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.process_thread.start()

    def stop(self):
        self.running = False
        if self.process_thread:
            self.process_thread.join()

    def _process_frames(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                self.frame_counter += 1
                
                # Process frame
                small_frame = cv2.resize(frame, (0, 0), 
                                      fx=self.downscale_factor, 
                                      fy=self.downscale_factor)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Only do heavy processing every N frames
                if self.frame_counter % self.detection_every_n_frames == 0:
                    face_locations = face_recognition.face_locations(
                        rgb_small,
                        number_of_times_to_upsample=1,  # Balanced accuracy/speed
                        model="hog"
                    )
                    
                    # Scale locations back up
                    scale = int(1/self.downscale_factor)
                    self.last_locations = [(top*scale, right*scale, bottom*scale, left*scale) 
                                         for (top, right, bottom, left) in face_locations]
                    
                    # Get encodings for all faces
                    self.last_encodings = face_recognition.face_encodings(
                        rgb_small, 
                        face_locations,
                        num_jitters=1
                    )
                
                # Prepare results using cached data
                results = []
                for (loc, encoding) in zip(self.last_locations, self.last_encodings):
                    name, confidence = self.attendance_system.recognize_face(encoding)
                    
                    # Only do liveness check on primary face
                    is_live = None
                    if loc == self.last_locations[0]:  # First face only
                        is_live = self.attendance_system.detect_liveness(frame, loc)
                    
                    results.append({
                        "location": loc,
                        "name": name,
                        "confidence": confidence,
                        "is_live": is_live
                    })
                
                # Update results
                if not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put(results)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue