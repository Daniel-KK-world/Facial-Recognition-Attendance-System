import cv2
import pickle
import os
import numpy as np
import concurrent.futures
import hashlib 
from datetime import datetime
import csv 

import pickle 


class AttendanceSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.attendance_log = []
        self.anti_spoofing_threshold = 0.3  # Threshold to indicate that a user is real. 
        self.min_confidence = 0.6  # Minimum confidence for recognition
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.liveness_cache = {}  # {name: timestamp}
        self.liveness_timeout = 10000  # seconds between liveness checks per person
        self.admin_password = self.hash_password("admin123")  # NEW: Default admin password
        self.load_data()
        
    # NEW PASSWORD METHODS ============================================
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_admin_password(self, password):
        """Verify admin password"""
        return self.hash_password(password) == self.admin_password

    def change_admin_password(self, old_password, new_password):
        """Change admin password after verification"""
        if self.verify_admin_password(old_password):
            self.admin_password = self.hash_password(new_password)
            return True
        return False

    def load_data(self):
        """Load all required data files"""
        try:
            # Load face encodings
            if os.path.exists("data/facial_recognition.dat"):
                with open("data/facial_recognition.dat", "rb") as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data["encodings"]
                    self.known_face_names = data["names"]
            
            # Load attendance records
            if os.path.exists("data/attendance.csv"):
                with open("data/attendance.csv", "r") as f:
                    reader = csv.DictReader(f)
                    self.attendance_log = list(reader)
                    
            # Create files if they don't exist
            if not os.path.exists("facial_recognition.dat"):
                self.save_known_faces()
            if not os.path.exists("data/attendance.csv"):
                with open("data/attendance.csv", "w") as f:
                    f.write("Name,Date,Check-in,Check-out\n")
                    
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create fresh files if loading fails
            self.known_face_encodings = []
            self.known_face_names = []
            self.attendance_log = []
            self.save_data()

    def save_data(self):
        """Save all data files"""
        try:
            self.save_known_faces()
            self.save_attendance_data()
        except Exception as e:
            print(f"Error saving data: {e}")

    def save_known_faces(self):
        """Save face encodings to file"""
        try:
            data = {
                "encodings": self.known_face_encodings,
                "names": self.known_face_names
            }
            with open("data/facial_recognition.dat", "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving face data: {e}")

    def save_attendance_data(self):
        """Save attendance records to file"""
        try:
            if self.attendance_log:
                keys = self.attendance_log[0].keys()
                with open("data/attendance.csv", "w", newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.attendance_log)
        except Exception as e:
            print(f"Error saving attendance data: {e}")

    def register_new_user(self, name, face_encodings):
        """Register a new user with multiple face samples"""
        if not name or not face_encodings:
            return False
        
        # Average the encodings for better accuracy
        avg_encoding = np.mean(face_encodings, axis=0)
        
        self.known_face_names.append(name)
        self.known_face_encodings.append(avg_encoding)
        self.save_known_faces()
        return True

    def recognize_face(self, face_encoding):
        if not self.known_face_encodings:
            return "Unknown", 0
        
        # Convert to numpy array first for vectorized operations
        known_encodings = np.array(self.known_face_encodings)
        face_encoding = np.array(face_encoding)
        
        # Vectorized distance calculation (much faster)
        distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
        
        # Find best match
        best_match_idx = distances.argmin()
        best_distance = distances[best_match_idx]
        
        # Fast confidence calculation
        confidence = max(0, 1 - (best_distance / 0.9))  # More aggressive confidence
        
        # Early return for obvious mismatches
        if best_distance > 0.6:  # Higher threshold for quick rejection
            return "Unknown", 0
        
        if confidence >= self.min_confidence:
            return self.known_face_names[best_match_idx], confidence
        return "Unknown", confidence

    def detect_liveness(self, frame, face_location):
        """
        Simple liveness detection to prevent spoofing
        Returns True if face appears to be live
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get face region
        top, right, bottom, left = face_location
        face_region = gray[top:bottom, left:right]
        
        # Reducing resolution for processing
        small_face = cv2.resize(face_region, (100, 100))
        
        # Calculate variance of Laplacian (focus measure)
        fm = cv2.Laplacian(small_face, cv2.CV_64F).var()
        
        # If focus measure is too low, might be a static image. 
        return fm > self.anti_spoofing_threshold

    def record_attendance(self, name, action):
        """Record check-in/check-out with validation"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date = datetime.now().strftime("%Y-%m-%d")
        
        # Check if user already has an entry today
        existing_entry = None
        for record in self.attendance_log:
            if record["Name"] == name and record["Date"] == date:
                existing_entry = record
                break
        
        if action == "Check-in":
            if existing_entry and existing_entry["Check-in"] != "":
                return False, "Already checked in today"
            
            if not existing_entry:
                new_record = {
                    "Name": name,
                    "Date": date,
                    "Check-in": timestamp,
                    "Check-out": ""
                }
                self.attendance_log.append(new_record)
            else:
                existing_entry["Check-in"] = timestamp
                
            self.save_attendance_data()
            return True, "Checked in successfully"
            
        elif action == "Check-out":
            if not existing_entry or existing_entry["Check-in"] == "":
                return False, "Not checked in yet"
            if existing_entry["Check-out"] != "":
                return False, "Already checked out today"
            
            existing_entry["Check-out"] = timestamp
            self.save_attendance_data()
            return True, "Checked out successfully"
        
        return False, "Invalid action"
