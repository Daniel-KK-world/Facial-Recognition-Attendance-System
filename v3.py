import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import tkinter.font as tkFont
from PIL import Image, ImageTk, ImageDraw
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import csv
import face_recognition
import pickle
import threading
import queue
from collections import deque
import random
import concurrent.futures
import time
import hashlib 
import ctypes


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
            if os.path.exists("facial_recognition.dat"):
                with open("facial_recognition.dat", "rb") as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data["encodings"]
                    self.known_face_names = data["names"]
            
            # Load attendance records
            if os.path.exists("attendance.csv"):
                with open("attendance.csv", "r") as f:
                    reader = csv.DictReader(f)
                    self.attendance_log = list(reader)
                    
            # Create files if they don't exist
            if not os.path.exists("facial_recognition.dat"):
                self.save_known_faces()
            if not os.path.exists("attendance.csv"):
                with open("attendance.csv", "w") as f:
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
            with open("facial_recognition.dat", "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving face data: {e}")

    def save_attendance_data(self):
        """Save attendance records to file"""
        try:
            if self.attendance_log:
                keys = self.attendance_log[0].keys()
                with open("attendance.csv", "w", newline='') as f:
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


class AttendanceUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("1280x720+100+50")
        self.root.title("KFCS Attendance Pro")
        self.root.configure(bg='white')
        self.root.tk.call('wm', 'iconphoto', self.root._w, tk.PhotoImage(width=1, height=1))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initialize systems
        self.attendance_system = AttendanceSystem()
        self.face_processor = FaceProcessor(self.attendance_system)
        self.face_processor.start()
        
        # Performance tracking
        self.frame_times = deque(maxlen=10)
        self.last_frame_time = datetime.now()
        
        # Custom fonts
        self.title_font = tkFont.Font(family="Segoe UI", size=24, weight="bold")
        self.button_font = tkFont.Font(family="Segoe UI", size=12)
        self.small_font = tkFont.Font(family="Segoe UI", size=10)
        
        # Create UI components
        self.create_main_container()
        self.kill_feather_icon()
        self.create_webcam_section()
        self.create_control_panel()
        self.create_status_bar()
        self.create_admin_button()
        self.create_user_button()
        self.create_logo()
        
        # Webcam init
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam!")
            self.root.destroy()
            return
        
        # Start webcam processing
        self.process_webcam()
        
        self.root.mainloop()
    
    def kill_feather_icon(self):
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("KFCS.Attendance.Pro")  # Unique ID
            self.root.iconbitmap(default="company_logo\steve_jobs_avatar_icon_263195.png")  # 1x1 transparent .ico file
        except Exception as e:
            print("Icon override failed (non-Windows?):", e)
    
    def on_close(self):
        """Cleanup on window close"""
        self.face_processor.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
    
    def create_main_container(self):
        """Create the main white container with shadow"""
        # Shadow effect
        self.shadow = tk.Frame(self.root, bg='#e0e0e0')
        self.shadow.place(x=52, y=52, width=1180, height=620)
        
        # Main white container
        self.main_frame = tk.Frame(
            self.root, 
            bg='white',
            bd=0,
            highlightthickness=0,
            relief='ridge'
        )
        self.main_frame.pack(
            side='left',
            fill='both',
            expand=True,
            padx=50,  # Same as control panel for balance
            pady=20
        )
    
    def create_webcam_section(self):
        """Create the webcam display area"""
        # Container for webcam with dark border
        self.webcam_container = tk.Frame(self.main_frame, bg='#333', bd=0)
        self.webcam_container.place(x=30, y=30, width=640, height=480)
        
        # Placeholder for webcam feed
        self.webcam_label = tk.Label(self.webcam_container, bg='#333')
        self.webcam_label.place(x=2, y=2, width=636, height=476)
        
        # Current user display
        self.current_user_label = tk.Label(self.webcam_container, 
                                         bg='#333', fg='white',
                                         font=self.button_font)
        self.current_user_label.place(x=10, y=10)
        
        # FPS display
        self.fps_label = tk.Label(self.webcam_container, 
                                 bg='#333', fg='white',
                                 font=self.small_font)
        self.fps_label.place(x=10, y=450)
    
    def process_webcam(self):
        """Process webcam frames with performance optimizations"""
        start_time = datetime.now()
        
        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            self.webcam_label.after(40, self.process_webcam)
            return
        
        # Put frame in processing queue (replace if not empty)
        if not self.face_processor.frame_queue.empty():
            try:
                self.face_processor.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.face_processor.frame_queue.put(frame.copy())
        
        # Get processing results if available
        face_results = []
        try:
            face_results = self.face_processor.result_queue.get_nowait()
        except queue.Empty:
            pass
        
        # Draw face boxes and labels
        current_user = None
        highest_confidence = 0
        
        for result in face_results:
            top, right, bottom, left = result["location"]
            name = result["name"]
            confidence = result["confidence"]
            is_live = result["is_live"]
            
            # Track user with highest confidence
            if name != "Unknown" and confidence > highest_confidence:
                current_user = name
                highest_confidence = confidence
            
            # Draw face rectangle
            color = (0, 255, 0) if is_live else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw name and confidence
            label = f"{name}"
            cv2.putText(frame, label, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        # Update current user display
        if current_user:
            self.current_user = current_user
            self.current_user_label.config(text=f"User: {current_user}")
        else:
            self.current_user = None
            self.current_user_label.config(text="No recognized user")
        
        # Convert to PhotoImage
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update label
        self.webcam_label.imgtk = imgtk
        self.webcam_label.configure(image=imgtk)
        
        # Calculate and display FPS
        frame_time = (datetime.now() - start_time).total_seconds()
        self.frame_times.append(frame_time)
        avg_fps = 1 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
        self.fps_label.config(text=f"FPS: {avg_fps:.1f}")
        
        # Repeat every 40ms (25 FPS target)
        self.webcam_label.after(100, self.process_webcam)
    
    def create_control_panel(self):
        """Create the right-side control panel"""
        self.control_panel = tk.Frame(self.main_frame, bg='white', bd=0)
        self.control_panel.pack(
            side='right',
            fill='y',
            padx=(20, 0),  # Only right padding
            pady=10,
          
        )
        
        # Welcome message
        tk.Label(self.control_panel, text="KFCS Attendance Pro", 
                font=self.title_font, bg='white', fg='#333').pack(pady=10)
        
        # Modern buttons
        self.login_btn = self.create_modern_button(
            self.control_panel, "CHECK IN", "#4CAF50", self.check_in)
        self.login_btn.pack(pady=15, ipady=10)
        
        self.logout_btn = self.create_modern_button(
            self.control_panel, "CHECK OUT", "#F44336", self.check_out)
        self.logout_btn.pack(pady=15, ipady=10)
        
        self.register_btn = self.create_modern_button(
            self.control_panel, "REGISTER NEW USER", "#2196F3", self.register_user)
        self.register_btn.pack(pady=15, ipady=10)
        
        # Separator
        ttk.Separator(self.control_panel, orient='horizontal').pack(fill='x', pady=20)
        
        # Stats frame
        self.stats_frame = tk.Frame(self.control_panel, bg='white')
        self.stats_frame.pack()
        
        tk.Label(self.stats_frame, text="Today's Stats", font=self.button_font,
                bg='white').grid(row=0, columnspan=2, pady=5)
                
        tk.Label(self.stats_frame, text="Checked in:", bg='white').grid(row=1, column=0, sticky='e')
        self.checked_in_label = tk.Label(self.stats_frame, text="0", bg='white', fg='#4CAF50')
        self.checked_in_label.grid(row=1, column=1, sticky='w')
        
        tk.Label(self.stats_frame, text="Pending:", bg='white').grid(row=2, column=0, sticky='e')
        self.pending_label = tk.Label(self.stats_frame, text="0", bg='white', fg='#FF9800')
        self.pending_label.grid(row=2, column=1, sticky='w')
        
        # Update stats
        self.update_stats()
    
    def update_stats(self):
        """Update the statistics display"""
        today = datetime.now().strftime("%Y-%m-%d")
        checked_in = 0
        pending = 0
        
        for record in self.attendance_system.attendance_log:
            if record["Date"] == today:
                if record["Check-in"] != "":
                    checked_in += 1
                if record["Check-out"] == "" and record["Check-in"] != "":
                    pending += 1
        
        self.checked_in_label.config(text=str(checked_in))
        self.pending_label.config(text=str(pending))
        
        # Update every minute
        self.root.after(60000, self.update_stats)
    
    def create_status_bar(self):
        """Create the status bar at bottom"""
        self.status = tk.Label(self.main_frame, 
                             text=f"System Ready | {len(self.attendance_system.known_face_names)} users registered | Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                             font=self.small_font, bg='white', fg='#666',
                             anchor='w')
        self.status.place(x=30, y=530, width=1120)
    
    def create_admin_button(self):
        """Create admin settings button"""
        self.admin_btn = tk.Label(self.main_frame, text="⚙", font=("Arial", 28), 
                                 bg='white', fg='#999', cursor="hand2")
        self.admin_btn.place(x=1120, y=572)
        self.admin_btn.bind("<Button-1>", lambda e: self.request_password())
    
    def create_user_button(self):
        """Create user profile button"""
        self.user_btn = tk.Label(self.main_frame, text="👤", font=('Arial', 28),
                                bg='white', fg='#999', cursor="hand2")
        self.user_btn.place(x=10, y=570)
        self.user_btn.bind("<Button-1>", lambda e: self.show_user_panel())
    
    def create_logo(self):
        """Create company logo"""
        try:
            self.logo_img = Image.open("company_logo/KFCS.ico").resize((200,150))
            self.logo_tk = ImageTk.PhotoImage(self.logo_img)
        except:
            # Create placeholder if logo not found
            self.logo_img = Image.new('RGB', (200, 80), color='#2196F3')
            draw = ImageDraw.Draw(self.logo_img)
            draw.text((10, 10), "KFCS Logo", fill="white")
            self.logo_tk = ImageTk.PhotoImage(self.logo_img)
            
        tk.Label(self.control_panel, image=self.logo_tk, bg='white').pack(pady=20)
    
    def create_modern_button(self, parent, text, color, command):
        """Helper to create modern-looking buttons"""
        btn = tk.Button(parent, text=text, command=command, 
                       font=self.button_font, bg=color, fg='white',
                       activebackground=self.lighten_color(color), 
                       activeforeground='white',
                       bd=0, relief='flat', highlightthickness=0,
                       padx=30, pady=5, cursor="hand2")
        
        # Hover effects
        btn.bind("<Enter>", lambda e: btn.config(bg=self.lighten_color(color)))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn
    
    def lighten_color(self, color, amount=0.2):
        """Lighten a hex color"""
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        lighter = tuple(min(255, int(c + (255 - c) * amount)) for c in rgb)
        return f'#{lighter[0]:02x}{lighter[1]:02x}{lighter[2]:02x}'
    
    def check_in(self):
        """Handle check-in action"""
        if not hasattr(self, 'current_user') or not self.current_user:
            messagebox.showwarning("Warning", "No recognized user detected!")
            return
        
        success, message = self.attendance_system.record_attendance(self.current_user, "Check-in")
        if success:
            messagebox.showinfo("Success", message)
            self.update_stats()
        else:
            messagebox.showwarning("Warning", message)
    
    def check_out(self):
        """Handle check-out action"""
        if not hasattr(self, 'current_user') or not self.current_user:
            messagebox.showwarning("Warning", "No recognized user detected!")
            return
        
        success, message = self.attendance_system.record_attendance(self.current_user, "Check-out")
        if success:
            messagebox.showinfo("Success", message)
            self.update_stats()
        else:
            messagebox.showwarning("Warning", message)
    
    def register_user(self):
        """Register a new user with face capture"""
        name = simpledialog.askstring("Register New User", "Enter user's full name:", parent=self.root)
        if not name:
            return
        
        # Capture face samples
        samples = []
        messagebox.showinfo("Instructions", "Please look directly at the camera. We'll capture 5 samples.")
        
        for i in range(5):
            ret, frame = self.cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if len(face_locations) == 1:
                    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    samples.append(face_encoding)
                    messagebox.showinfo("Sample Captured", f"Sample {i+1}/5 captured")
                else:
                    messagebox.showerror("Error", "Could not detect face. Please try again.")
                    return
        
        if len(samples) == 5:
            success = self.attendance_system.register_new_user(name, samples)
            if success:
                messagebox.showinfo("Success", f"User {name} registered successfully!")
                self.status.config(text=f"System Ready | {len(self.attendance_system.known_face_names)} users registered | Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
    def request_password(self):
        """Request admin password and verify"""
        password = simpledialog.askstring("Admin Authentication", 
                                        "Enter Admin Password:", 
                                        show='*',
                                        parent=self.root)
        if password is None:  # User cancelled
            return
            
        if self.attendance_system.verify_admin_password(password):
            self.show_admin_panel()
        else:
            messagebox.showerror("Access Denied", "Incorrect admin password!")

    
    def show_admin_panel(self):
        """Show the admin panel with management features"""
        admin_win = tk.Toplevel(self.root)
        admin_win.geometry("900x600+200+100")
        admin_win.title("Admin Dashboard")
        admin_win.configure(bg='#f5f5f5')
        
        # Add password change option
        def change_password():
            old = simpledialog.askstring("Change Password", "Enter current password:", show='*')
            if not old:
                return
                
            new_pass = simpledialog.askstring("Change Password", "Enter new password:", show='*')
            if not new_pass:
                return
                
            confirm = simpledialog.askstring("Change Password", "Confirm new password:", show='*')
            
            if new_pass != confirm:
                messagebox.showerror("Error", "New passwords don't match!")
                return
                
            if self.attendance_system.change_admin_password(old, new_pass):
                messagebox.showinfo("Success", "Password changed successfully!")
            else:
                messagebox.showerror("Error", "Incorrect current password!")
        
        # Password change button
        ttk.Button(admin_win, text="Change Admin Password", 
                command=change_password).pack(pady=10)
    
        # Notebook (tabbed interface)
        notebook = ttk.Notebook(admin_win)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # --- Tab 1: Attendance Reports ---
        reports_frame = ttk.Frame(notebook)
        notebook.add(reports_frame, text="Attendance Reports")
        
        # Date range selection
        date_frame = ttk.Frame(reports_frame)
        date_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(date_frame, text="From:").pack(side='left')
        self.start_date = ttk.Entry(date_frame)
        self.start_date.pack(side='left', padx=5)
        
        ttk.Label(date_frame, text="To:").pack(side='left')
        self.end_date = ttk.Entry(date_frame)
        self.end_date.pack(side='left', padx=5)
        
        ttk.Button(date_frame, text="Filter", 
                  command=lambda: self.filter_attendance(tree)).pack(side='left', padx=10)
        
        # Treeview for data display
        columns = ("Date", "Name", "Check-in", "Check-out", "Hours")
        tree = ttk.Treeview(reports_frame, columns=columns, show="headings")
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        vsb = ttk.Scrollbar(reports_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        
        # Insert data
        for record in sorted(self.attendance_system.attendance_log, 
                           key=lambda x: x["Date"], reverse=True):
            hours = self.calculate_hours(record["Check-in"], record["Check-out"])
            tree.insert("", "end", values=(
                record["Date"],
                record["Name"],
                record["Check-in"],
                record["Check-out"],
                f"{hours:.1f}" if hours else ""
            ))
        
        
        # Export button
        ttk.Button(reports_frame, text="Export to Excel", 
                  command=lambda: self.export_to_excel(tree)).pack(pady=10)
        
        # --- Tab 2: User Management ---
        user_frame = ttk.Frame(notebook)
        notebook.add(user_frame, text="User Management")
        
        # User list
        user_list = ttk.Treeview(user_frame, columns=("Name"), show="headings")
        user_list.heading("Name", text="Name")
        user_list.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add registered users
        for name in set(self.attendance_system.known_face_names):
            user_list.insert("", "end", values=(name,))
        
        # Action buttons
        btn_frame = tk.Frame(user_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Remove User", 
                  command=lambda: self.remove_user(user_list)).pack(side='left', padx=5)
        
        # --- Tab 3: System Settings ---
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="System Settings")
        
        # Confidence threshold setting
        ttk.Label(settings_frame, text="Recognition Confidence Threshold:").pack(pady=5)
        self.confidence_slider = ttk.Scale(settings_frame, from_=0.5, to=1.0, 
                                         value=self.attendance_system.min_confidence)
        self.confidence_slider.pack(pady=5)
        
        ttk.Button(settings_frame, text="Save Settings", 
                  command=self.save_settings).pack(pady=10)
                
        #request password function
        def request_password():
            entered = simpledialog.askstring("Admin Login", "Enter Admin Password:", show='*')
            ADMIN_PASSWORD = 'admin123'
            if entered == ADMIN_PASSWORD:
                self.show_admin_pannel()
            else:
                messagebox.showerror("Access Denied", "Incorrect password!") 
            
    def filter_attendance(self, tree):
        """Filter attendance records by date range"""
        start_date = self.start_date.get()
        end_date = self.end_date.get()
        
        # Clear existing items
        for item in tree.get_children():
            tree.delete(item)
        
        # Filter and insert records
        for record in self.attendance_system.attendance_log:
            if (not start_date or record["Date"] >= start_date) and \
               (not end_date or record["Date"] <= end_date):
                hours = self.calculate_hours(record["Check-in"], record["Check-out"])
                tree.insert("", "end", values=(
                    record["Date"],
                    record["Name"],
                    record["Check-in"],
                    record["Check-out"],
                    f"{hours:.1f}" if hours else ""
                ))
    
    def calculate_hours(self, check_in, check_out):
        """Calculate hours worked from check-in/check-out times"""
        if not check_in or not check_out:
            return None
        
        try:
            in_time = datetime.strptime(check_in, "%Y-%m-%d %H:%M:%S")
            out_time = datetime.strptime(check_out, "%Y-%m-%d %H:%M:%S")
            return (out_time - in_time).total_seconds() / 3600
        except:
            return None
    
    def export_to_excel(self, tree):
        """Export attendance data to Excel on desktop"""
        try:
            # Get all items from treeview
            items = tree.get_children()
            data = []
            columns = tree["columns"]
            
            # Get column headers (using the 'text' from each column heading)
            headers = [tree.heading(col)["text"] for col in columns]
            
            for item in items:
                values = tree.item(item, "values")
                data.append(dict(zip(headers, values)))  # Use headers instead of column IDs
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Get desktop path (works for Windows, macOS, and Linux)
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            
            # Create filename with timestamp
            filename = f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            filepath = os.path.join(desktop, filename)
            
            # Export to Excel
            df.to_excel(filepath, index=False)
            
            messagebox.showinfo(
                "Export Successful", 
                f"Attendance report has been saved to your desktop:\n{filename}"
            )
        except Exception as e:
            messagebox.showerror(
                "Export Failed", 
                f"Could not export report:\n{str(e)}"
            )
    
    def remove_user(self, user_list):
        """Remove selected user from system"""
        selected = user_list.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a user to remove")
            return
        
        name = user_list.item(selected[0], 'values')[0]
        
        if messagebox.askyesno("Confirm", f"Remove user {name}? This cannot be undone."):
            # Remove from known faces
            indices = [i for i, x in enumerate(self.attendance_system.known_face_names) if x == name]
            for index in sorted(indices, reverse=True):
                del self.attendance_system.known_face_names[index]
                del self.attendance_system.known_face_encodings[index]
            
            # Save changes
            self.attendance_system.save_known_faces()
            
            # Update UI
            user_list.delete(selected[0])
            self.status.config(text=f"System Ready | {len(self.attendance_system.known_face_names)} users registered | Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            messagebox.showinfo("Success", f"User {name} removed successfully")
    
    def save_settings(self):
        """Save system settings"""
        self.attendance_system.min_confidence = float(self.confidence_slider.get())
        messagebox.showinfo("Success", "Settings saved successfully")
    
    def show_user_panel(self):
        """Show user-specific attendance dashboard"""
        if not hasattr(self, 'current_user') or not self.current_user:
            messagebox.showwarning("Warning", "No recognized user detected!")
            return
            
        user_win = tk.Toplevel(self.root)
        user_win.geometry("900x600")
        user_win.title(f"{self.current_user}'s Attendance Dashboard")
        user_win.configure(bg='#f0f2f5')

        # Header
        header = tk.Frame(user_win, bg='#2c3e50', height=100)
        header.pack(fill='x')
        
        tk.Label(header, 
                text=f"{self.current_user}", 
                font=("Arial", 20, 'bold'), 
                bg='#2c3e50', fg='white').pack(side='left', padx=20, pady=10)
        
        # Today's status card
        status_frame = tk.Frame(user_win, bg='white', bd=2, relief='groove')
        status_frame.pack(fill='x', padx=20, pady=10)
        
        today = datetime.now().strftime("%Y-%m-%d")
        today_record = next((r for r in self.attendance_system.attendance_log 
                           if r["Name"] == self.current_user and r["Date"] == today), None)
        
        status_text = ("✅ Currently working" if today_record and not today_record["Check-out"] else
                      "🟢 Checked out" if today_record else 
                      "🔴 Not checked in today")
        
        tk.Label(status_frame, text="TODAY'S STATUS", 
                font=("Arial", 12, 'bold'), bg='white').grid(row=0, column=0, sticky='w', padx=10)
        tk.Label(status_frame, text=status_text, 
                font=("Arial", 14), bg='white').grid(row=1, column=0, sticky='w', padx=10, pady=5)
        
        # Metric cards
        metrics_frame = tk.Frame(user_win, bg='#f0f2f5')
        metrics_frame.pack(fill='x', padx=20, pady=10)
        
        # Card 1: Present Days
        present_days = len([r for r in self.attendance_system.attendance_log 
                          if r["Name"] == self.current_user and r["Check-in"]])
        self._create_metric_card(metrics_frame, "Present Days", present_days, "#4CAF50", 0, 0)
        
        # Card 2: Avg Hours
        avg_hours = self._calculate_avg_hours(self.current_user)
        self._create_metric_card(metrics_frame, "Avg Hours/Day", f"{avg_hours:.1f}h", "#2196F3", 0, 1)
        
        # Card 3: Late Arrivals
        late_days = len([r for r in self.attendance_system.attendance_log 
                        if r["Name"] == self.current_user and self._is_late(r["Check-in"])])
        self._create_metric_card(metrics_frame, "Late Arrivals", late_days, "#FF9800", 0, 2)

        # Attendance history
        history_frame = tk.Frame(user_win)
        history_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        columns = ("Date", "Check-in", "Check-out", "Hours", "Status")
        tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor='center')
        
        # Insert user's attendance records
        for record in sorted(
            [r for r in self.attendance_system.attendance_log 
             if r["Name"] == self.current_user],
            key=lambda x: x["Date"], 
            reverse=True
        ):
            hours = self.calculate_hours(record["Check-in"], record["Check-out"])
            status = self._get_status_icon(record["Check-in"], record["Check-out"])
            tree.insert("", "end", values=(
                record["Date"],
                record["Check-in"] or "-",
                record["Check-out"] or "-",
                f"{hours:.1f}" if hours else "-",
                status
            ))
        
        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Action buttons
        actions_frame = tk.Frame(user_win, bg='#f0f2f5')
        actions_frame.pack(fill='x', pady=10)
        
        ttk.Button(actions_frame, 
                  text="Export My Attendance", 
                  command=lambda: self.export_user_data(self.current_user)).pack(side='left', padx=10)
        
        ttk.Button(actions_frame, 
                  text="Request Correction", 
                  command=self.request_correction).pack(side='left')
    
    def _create_metric_card(self, parent, title, value, color, row, col):
        """Helper to create metric cards"""
        card = tk.Frame(parent, bg='white', bd=1, relief='groove')
        card.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
        
        tk.Label(card, text=title, font=("Arial", 10), bg='white').pack(pady=(10,0))
        tk.Label(card, text=value, font=("Arial", 18, 'bold'), bg='white', 
                fg=color).pack(pady=5)
    
    def _calculate_avg_hours(self, user_name):
        """Calculate average working hours for a user"""
        total_hours = 0
        count = 0
        
        for record in self.attendance_system.attendance_log:
            if record["Name"] == user_name and record["Check-in"] and record["Check-out"]:
                hours = self.calculate_hours(record["Check-in"], record["Check-out"])
                if hours:
                    total_hours += hours
                    count += 1
        
        return total_hours / count if count > 0 else 0
    
    def _is_late(self, check_in_time):
        """Check if check-in was late (after 9:30 AM)"""
        if not check_in_time:
            return False
        
        try:
            check_in = datetime.strptime(check_in_time, "%Y-%m-%d %H:%M:%S")
            return check_in.time() > datetime.strptime("09:30:00", "%H:%M:%S").time()
        except:
            return False
    
    def _get_status_icon(self, check_in, check_out):
        """Get status icon for attendance record"""
        if not check_in:
            return "❌ Absent"
        elif check_in and not check_out:
            return "⚠️ Missing Check-out"
        else:
            hours = self.calculate_hours(check_in, check_out)
            if hours and hours < 4:
                return "⚠️ Short Day"
            elif self._is_late(check_in):
                return "⚠️ Late"
            else:
                return "✅ Complete"
    
    def export_user_data(self, user_name):
        """Export user's attendance data to Excel"""
        try:
            # Filter user's records
            user_records = [r for r in self.attendance_system.attendance_log 
                          if r["Name"] == user_name]
            
            if not user_records:
                messagebox.showwarning("Warning", "No attendance records found")
                return
            
            # Create DataFrame
            df = pd.DataFrame(user_records)
            filename = f"{user_name}_attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(filename, index=False)
            messagebox.showinfo("Success", f"Data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def request_correction(self):
        """Handle attendance correction requests"""
        messagebox.showinfo("Request Sent", "Your correction request has been submitted to HR")


# Run the application
if __name__ == "__main__":
    app = AttendanceUI()