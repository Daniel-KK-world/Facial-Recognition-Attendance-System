import face_recognition
import queue 
import threading 
import cv2 
from datetime import datetime, timedelta

class FaceProcessor:
    """Optimized but reliable face processing with automated attendance"""
    def __init__(self, attendance_system):
        self.attendance_system = attendance_system
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = False
        self.process_thread = None
        self.last_locations = []
        self.last_encodings = []
        
        # Tune these for your hardware
        self.downscale_factor = 0.3
        self.detection_every_n_frames = 20
        self.frame_counter = 0
        
        # NEW: Automated attendance tracking
        self.face_presence_tracker = {}  # {name: {"last_seen": timestamp, "status": "present/absent"}}
        self.auto_check_delay = 30  # seconds before auto check-out
        self.last_auto_check = datetime.now()

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
                        number_of_times_to_upsample=1,
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
                
                # NEW: Automated attendance logic
                current_time = datetime.now()
                current_faces = set()
                
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
                    
                    # NEW: Track recognized faces for automated attendance
                    if name != "Unknown" and confidence > 0.7:
                        current_faces.add(name)
                        
                        # Auto check-in for new detections
                        if name not in self.face_presence_tracker:
                            status = self.attendance_system.get_attendance_status(name)
                            if status == "absent":
                                success, message = self.attendance_system.auto_check_in(name)
                                if success:
                                    print(f"✅ Auto check-in: {name}")
                        
                        # Update presence tracker
                        self.face_presence_tracker[name] = {
                            "last_seen": current_time,
                            "status": "present"
                        }
                
                # NEW: Check for departed faces (auto check-out)
                self._check_departures(current_faces, current_time)
                
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

    def _check_departures(self, current_faces, current_time):
        """Check for faces that have left and trigger auto check-out"""
        departed_faces = set(self.face_presence_tracker.keys()) - current_faces
        
        for name in departed_faces:
            last_seen = self.face_presence_tracker[name]["last_seen"]
            time_absent = (current_time - last_seen).total_seconds()
            
            # If person has been absent for the delay period
            if time_absent > self.auto_check_delay:
                status = self.attendance_system.get_attendance_status(name)
                
                # Only check out if they're currently checked in
                if status == "checked_in":
                    success, message = self.attendance_system.auto_check_out(name)
                    if success:
                        print(f"🚪 Auto check-out: {name} (absent for {time_absent:.0f}s)")
                    else:
                        # If auto check-out failed but person left early, force check-out
                        current_hour = current_time.hour
                        if current_hour < 16:  # Before normal work end
                            success, message = self.attendance_system.force_check_out(name)
                            if success:
                                print(f"🚪 Early departure: {name}")
                
                # Remove from tracker
                del self.face_presence_tracker[name]
            else:
                # Update status to absent but keep tracking
                self.face_presence_tracker[name]["status"] = "absent"