import cv2
import pickle
import os
import numpy as np
import concurrent.futures
import hashlib 
from datetime import datetime, timedelta
import csv 

class AttendanceSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.attendance_log = []
        self.anti_spoofing_threshold = 0.3
        self.min_confidence = 0.6
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.liveness_cache = {}
        self.liveness_timeout = 10000
        self.admin_password = self.hash_password("admin123")
        self.load_data()
        
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_admin_password(self, password):
        return self.hash_password(password) == self.admin_password

    def change_admin_password(self, old_password, new_password):
        if self.verify_admin_password(old_password):
            self.admin_password = self.hash_password(new_password)
            return True
        return False

    def load_data(self):
        try:
            if os.path.exists("data/facial_recognition.dat"):
                with open("data/facial_recognition.dat", "rb") as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data["encodings"]
                    self.known_face_names = data["names"]
            
            if os.path.exists("data/attendance.csv"):
                with open("data/attendance.csv", "r") as f:
                    reader = csv.DictReader(f)
                    self.attendance_log = list(reader)
                    
            if not os.path.exists("data/facial_recognition.dat"):
                self.save_known_faces()
            if not os.path.exists("data/attendance.csv"):
                with open("data/attendance.csv", "w") as f:
                    f.write("Name,Date,Check-in,Check-out\n")
                    
        except Exception as e:
            print(f"Error loading data: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
            self.attendance_log = []
            self.save_data()

    def save_data(self):
        try:
            self.save_known_faces()
            self.save_attendance_data()
        except Exception as e:
            print(f"Error saving data: {e}")

    def save_known_faces(self):
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
        if not name or not face_encodings:
            return False
        
        avg_encoding = np.mean(face_encodings, axis=0)
        
        self.known_face_names.append(name)
        self.known_face_encodings.append(avg_encoding)
        self.save_known_faces()
        return True

    def recognize_face(self, face_encoding):
        if not self.known_face_encodings:
            return "Unknown", 0
        
        known_encodings = np.array(self.known_face_encodings)
        face_encoding = np.array(face_encoding)
        
        distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
        best_match_idx = distances.argmin()
        best_distance = distances[best_match_idx]
        
        confidence = max(0, 1 - (best_distance / 0.9))
        
        if best_distance > 0.6:
            return "Unknown", 0
        
        if confidence >= self.min_confidence:
            return self.known_face_names[best_match_idx], confidence
        return "Unknown", confidence

    def detect_liveness(self, frame, face_location):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        top, right, bottom, left = face_location
        face_region = gray[top:bottom, left:right]
        small_face = cv2.resize(face_region, (100, 100))
        fm = cv2.Laplacian(small_face, cv2.CV_64F).var()
        return fm > self.anti_spoofing_threshold

    def record_attendance(self, name, action):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date = datetime.now().strftime("%Y-%m-%d")
        
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

    # NEW AUTOMATED METHODS
    def auto_check_in(self, name):
        """Automatically check in if within work hours and not already checked in"""
        current_time = datetime.now()
        
        # Only auto check-in during morning hours (6 AM - 11 AM)
        if not (6 <= current_time.hour <= 11):
            return False, "Outside auto check-in hours"
        
        return self.record_attendance(name, "Check-in")

    def auto_check_out(self, name):
        """Automatically check out if checked in and not already checked out"""
        current_time = datetime.now()
        date = current_time.strftime("%Y-%m-%d")
        
        # Find today's record
        today_record = next((r for r in self.attendance_log 
                            if r["Name"] == name and r["Date"] == date), None)
        
        if not today_record or today_record["Check-in"] == "":
            return False, "Not checked in today"
        
        if today_record["Check-out"] != "":
            return False, "Already checked out"
        
        # Auto check-out conditions:
        check_in_time = datetime.strptime(today_record["Check-in"], "%Y-%m-%d %H:%M:%S")
        hours_worked = (current_time - check_in_time).total_seconds() / 3600
        
        # Check-out if: after 4 PM OR after 1 PM with at least 4 hours worked
        if current_time.hour >= 16 or (current_time.hour >= 13 and hours_worked >= 4):
            return self.record_attendance(name, "Check-out")
        
        return False, "Not yet time for auto check-out"

    def get_attendance_status(self, name):
        """Get current attendance status for a user"""
        date = datetime.now().strftime("%Y-%m-%d")
        
        for record in self.attendance_log:
            if record["Name"] == name and record["Date"] == date:
                if record["Check-out"]:
                    return "checked_out"
                elif record["Check-in"]:
                    return "checked_in"
        return "absent"

    def force_check_out(self, name):
        """Force check-out for early departures"""
        return self.record_attendance(name, "Check-out")