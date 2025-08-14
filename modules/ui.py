from modules import FaceProcessor, AttendanceSystem
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import tkinter.font as tkFont
from PIL import Image, ImageTk
import cv2
import os
import pandas as pd
from datetime import datetime
import face_recognition
import queue
from collections import deque
import ctypes
from modules.face_processor import FaceProcessor

class AttendanceUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("1280x800+100+50")
        self.root.title("KFCS Attendance Pro")
        self.root.configure(bg='#f5f7fa')
        self.root.tk.call('wm', 'iconphoto', self.root._w, tk.PhotoImage(width=1, height=1))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Custom color scheme
        self.primary_color = "#4e73df"
        self.secondary_color = "#1cc88a"
        self.danger_color = "#e74a3b"
        self.warning_color = "#f6c23e"
        self.dark_color = "#5a5c69"
        self.light_color = "#f8f9fc"
        
        # Initialize systems
        self.attendance_system = AttendanceSystem()
        self.face_processor = FaceProcessor(self.attendance_system)
        self.face_processor.start()
        
        # Performance tracking
        self.frame_times = deque(maxlen=10)
        self.last_frame_time = datetime.now()
        
        # Custom fonts
        self.title_font = tkFont.Font(family="Segoe UI", size=24, weight="bold")
        self.subtitle_font = tkFont.Font(family="Segoe UI", size=14)
        self.button_font = tkFont.Font(family="Segoe UI", size=12, weight="bold")
        self.small_font = tkFont.Font(family="Segoe UI", size=10)
        
        # Create UI components
        self.create_header()
        self.create_main_container()
        self.create_webcam_section()
        self.create_control_panel()
        self.create_status_bar()
        self.create_admin_button()
        self.create_user_button()
        
        # Webcam init
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam!")
            self.root.destroy()
            return
        
        # Start webcam processing
        self.process_webcam()
        
        self.root.mainloop()
    
    def create_header(self):
        """Create the top header bar"""
        self.header = tk.Frame(self.root, bg=self.primary_color, height=60)
        self.header.pack(fill='x', padx=0, pady=0)
        
        # Logo and title
        self.logo_label = tk.Label(
            self.header, 
            text="KFCS Attendance Pro", 
            font=self.title_font,
            bg=self.primary_color,
            fg='white',
            padx=20
        )
        self.logo_label.pack(side='left')
        
        # Current time display
        self.time_label = tk.Label(
            self.header,
            font=self.subtitle_font,
            bg=self.primary_color,
            fg='white',
            padx=20
        )
        self.time_label.pack(side='right')
        self.update_time()
    
    def update_time(self):
        """Update the current time display"""
        current_time = datetime.now().strftime("%A, %B %d %Y | %I:%M:%S %p")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)
    
    def on_close(self):
        """Cleanup on window close"""
        self.face_processor.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
    
    def create_main_container(self):
        """Create the main container with shadow"""
        self.main_frame = tk.Frame(
            self.root, 
            bg=self.light_color,
            bd=0,
            highlightthickness=0,
            relief='ridge'
        )
        self.main_frame.pack(
            fill='both',
            expand=True,
            padx=20,
            pady=20
        )
    
    def create_webcam_section(self):
        """Create the webcam display area"""
        # Container for webcam with card styling
        self.webcam_card = tk.Frame(
            self.main_frame, 
            bg='white',
            bd=0,
            highlightthickness=0,
            relief='groove'
        )
        self.webcam_card.pack(
            side='left',
            fill='both',
            expand=True,
            padx=10,
            pady=10
        )
        
        # Title for webcam section
        tk.Label(
            self.webcam_card,
            text="Face Recognition",
            font=self.subtitle_font,
            bg='white',
            fg=self.dark_color
        ).pack(pady=(10, 5))
        
        # Webcam feed container
        self.webcam_container = tk.Frame(self.webcam_card, bg='#333', bd=0)
        self.webcam_container.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Placeholder for webcam feed
        self.webcam_label = tk.Label(self.webcam_container, bg='#333')
        self.webcam_label.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Current user display
        self.current_user_frame = tk.Frame(self.webcam_card, bg='white')
        self.current_user_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(
            self.current_user_frame,
            text="Current User:",
            font=self.small_font,
            bg='white',
            fg=self.dark_color
        ).pack(side='left', padx=10)
        
        self.current_user_label = tk.Label(
            self.current_user_frame,
            text="Not detected",
            font=self.small_font,
            bg='white',
            fg=self.primary_color
        )
        self.current_user_label.pack(side='left')
        
        # FPS display
        self.fps_label = tk.Label(
            self.current_user_frame,
            text="FPS: 0.0",
            font=self.small_font,
            bg='white',
            fg=self.dark_color
        )
        self.fps_label.pack(side='right', padx=10)
    
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
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        # Update current user display
        if current_user:
            self.current_user = current_user
            self.current_user_label.config(text=current_user, fg=self.secondary_color)
        else:
            self.current_user = None
            self.current_user_label.config(text="Not detected", fg=self.dark_color)
        
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
        self.webcam_label.after(40, self.process_webcam)
    
    def create_control_panel(self):
        """Create the right-side control panel"""
        self.control_panel = tk.Frame(
            self.main_frame, 
            bg='white',
            bd=0,
            highlightthickness=0,
            relief='groove'
        )
        self.control_panel.pack(
            side='right',
            fill='y',
            padx=10,
            pady=10,
            ipadx=10
        )
        
        # Title for control panel
        tk.Label(
            self.control_panel,
            text="Attendance Controls",
            font=self.subtitle_font,
            bg='white',
            fg=self.dark_color
        ).pack(pady=(10, 20))
        
        # Modern buttons
        button_frame = tk.Frame(self.control_panel, bg='white')
        button_frame.pack(fill='x', pady=10)
        
        self.login_btn = self.create_modern_button(
            button_frame, "CHECK IN", self.secondary_color, self.check_in)
        self.login_btn.pack(fill='x', pady=5, ipady=8)
        
        self.logout_btn = self.create_modern_button(
            button_frame, "CHECK OUT", self.danger_color, self.check_out)
        self.logout_btn.pack(fill='x', pady=5, ipady=8)
        
        self.register_btn = self.create_modern_button(
            button_frame, "REGISTER NEW USER", self.primary_color, self.register_user)
        self.register_btn.pack(fill='x', pady=5, ipady=8)
        
        # Stats frame
        stats_frame = tk.Frame(self.control_panel, bg='white')
        stats_frame.pack(fill='x', pady=20)
        
        tk.Label(
            stats_frame,
            text="Today's Attendance",
            font=self.subtitle_font,
            bg='white',
            fg=self.dark_color
        ).pack(pady=(0, 10))
        
        # Stats cards
        self.checked_in_card = self.create_stat_card(
            stats_frame, "Checked In", "0", self.secondary_color)
        self.checked_in_card.pack(side='left', fill='x', expand=True, padx=5)
        
        self.pending_card = self.create_stat_card(
            stats_frame, "Pending", "0", self.warning_color)
        self.pending_card.pack(side='left', fill='x', expand=True, padx=5)
        
        # Update stats
        self.update_stats()
    
    def create_stat_card(self, parent, title, value, color):
        """Create a statistic card"""
        card = tk.Frame(parent, bg='white', bd=1, relief='groove')
        
        tk.Label(
            card,
            text=title,
            font=self.small_font,
            bg='white',
            fg=self.dark_color
        ).pack(pady=(10, 0))
        
        tk.Label(
            card,
            text=value,
            font=("Segoe UI", 18, 'bold'),
            bg='white',
            fg=color
        ).pack(pady=5)
        
        return card
    
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
        
        # Update the stat cards
        for widget in self.checked_in_card.winfo_children():
            if isinstance(widget, tk.Label) and widget['text'].isdigit():
                widget.config(text=str(checked_in))
        
        for widget in self.pending_card.winfo_children():
            if isinstance(widget, tk.Label) and widget['text'].isdigit():
                widget.config(text=str(pending))
        
        # Update every minute
        self.root.after(60000, self.update_stats)
    
    def create_status_bar(self):
        """Create the status bar at bottom"""
        self.status = tk.Label(
            self.root, 
            text=f"System Ready | {len(self.attendance_system.known_face_names)} users registered | Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
            font=self.small_font, 
            bg=self.dark_color, 
            fg='white',
            anchor='w',
            padx=20
        )
        self.status.pack(side='bottom', fill='x', ipady=10)
    
    def create_admin_button(self):
        """Create admin settings button"""
        self.admin_btn = tk.Label(
            self.header, 
            text="⚙", 
            font=("Arial", 20), 
            bg=self.primary_color, 
            fg='white', 
            cursor="hand2"
        )
        self.admin_btn.pack(side='right', padx=10)
        self.admin_btn.bind("<Button-1>", lambda e: self.request_password())
    
    def create_user_button(self):
        """Create user profile button"""
        self.user_btn = tk.Label(
            self.header, 
            text="👤", 
            font=('Arial', 20),
            bg=self.primary_color, 
            fg='white', 
            cursor="hand2"
        )
        self.user_btn.pack(side='right', padx=10)
        self.user_btn.bind("<Button-1>", lambda e: self.show_user_panel())
    
    def create_modern_button(self, parent, text, color, command):
        """Helper to create modern-looking buttons"""
        btn = tk.Button(
            parent, 
            text=text, 
            command=command, 
            font=self.button_font, 
            bg=color, 
            fg='white',
            activebackground=self.lighten_color(color), 
            activeforeground='white',
            bd=0, 
            relief='flat', 
            highlightthickness=0,
            cursor="hand2",
            padx=20
        )
        
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
        password = simpledialog.askstring(
            "Admin Authentication", 
            "Enter Admin Password:", 
            show='*',
            parent=self.root
        )
        if password is None:  # User cancelled
            return
            
        if self.attendance_system.verify_admin_password(password):
            self.show_admin_panel()
        else:
            messagebox.showerror("Access Denied", "Incorrect admin password!")
    
    def show_admin_panel(self):
        """Show the admin panel with management features"""
        admin_win = tk.Toplevel(self.root)
        admin_win.geometry("1000x700")
        admin_win.title("Admin Dashboard")
        admin_win.configure(bg=self.light_color)
        
        # Style configuration
        style = ttk.Style()
        style.configure("Admin.TFrame", background=self.light_color)
        style.configure("Admin.TLabel", background=self.light_color)
        style.configure("Admin.TButton", font=self.button_font)
        
        # Main container
        main_frame = ttk.Frame(admin_win, style="Admin.TFrame")
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header = ttk.Frame(main_frame, style="Admin.TFrame")
        header.pack(fill='x', pady=(0, 20))
        
        ttk.Label(
            header,
            text="Admin Dashboard",
            font=self.title_font,
            style="Admin.TLabel"
        ).pack(side='left')
        
        # Password change button
        ttk.Button(
            header,
            text="Change Admin Password",
            command=self.change_admin_password,
            style="Admin.TButton"
        ).pack(side='right')
        
        # Notebook (tabbed interface)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True)
        
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
        
        ttk.Button(
            date_frame, 
            text="Filter", 
            command=lambda: self.filter_attendance(tree)
        ).pack(side='left', padx=10)
        
        # Treeview for data display
        tree_frame = ttk.Frame(reports_frame)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        columns = ("Date", "Name", "Check-in", "Check-out", "Hours")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        
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
        ttk.Button(
            reports_frame, 
            text="Export to Excel", 
            command=lambda: self.export_to_excel(tree)
        ).pack(pady=10)
        
        # --- Tab 2: User Management ---
        user_frame = ttk.Frame(notebook)
        notebook.add(user_frame, text="User Management")
        
        # User list
        user_tree_frame = ttk.Frame(user_frame)
        user_tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        user_list = ttk.Treeview(user_tree_frame, columns=("Name"), show="headings")
        user_list.heading("Name", text="Name")
        user_list.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(user_tree_frame, orient="vertical", command=user_list.yview)
        user_list.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        
        # Add registered users
        for name in set(self.attendance_system.known_face_names):
            user_list.insert("", "end", values=(name,))
        
        # Action buttons
        btn_frame = ttk.Frame(user_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(
            btn_frame, 
            text="Remove User", 
            command=lambda: self.remove_user(user_list)
        ).pack(side='left', padx=5)
        
        # --- Tab 3: System Settings ---
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="System Settings")
        
        # Confidence threshold setting
        ttk.Label(
            settings_frame, 
            text="Recognition Confidence Threshold:",
            font=self.subtitle_font
        ).pack(pady=10)
        
        self.confidence_slider = ttk.Scale(
            settings_frame, 
            from_=0.5, 
            to=1.0, 
            value=self.attendance_system.min_confidence
        )
        self.confidence_slider.pack(pady=5, padx=20, fill='x')
        
        ttk.Button(
            settings_frame, 
            text="Save Settings", 
            command=self.save_settings
        ).pack(pady=20)
    
    def change_admin_password(self):
        """Change admin password dialog"""
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
        """Export attendance data to Excel"""
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
        user_win.geometry("900x700")
        user_win.title(f"{self.current_user}'s Attendance Dashboard")
        user_win.configure(bg=self.light_color)

        # Header
        header = tk.Frame(user_win, bg=self.primary_color, height=80)
        header.pack(fill='x')
        
        tk.Label(
            header, 
            text=f"{self.current_user}", 
            font=("Segoe UI", 20, 'bold'), 
            bg=self.primary_color, 
            fg='white'
        ).pack(side='left', padx=20, pady=20)
        
        # Today's status card
        status_card = tk.Frame(
            user_win, 
            bg='white', 
            bd=1, 
            relief='groove'
        )
        status_card.pack(fill='x', padx=20, pady=20)
        
        today = datetime.now().strftime("%Y-%m-%d")
        today_record = next((r for r in self.attendance_system.attendance_log 
                           if r["Name"] == self.current_user and r["Date"] == today), None)
        
        status_text = ("✅ Currently working" if today_record and not today_record["Check-out"] else
                      "🟢 Checked out" if today_record else 
                      "🔴 Not checked in today")
        
        tk.Label(
            status_card, 
            text="TODAY'S STATUS", 
            font=("Segoe UI", 12, 'bold'), 
            bg='white'
        ).grid(row=0, column=0, sticky='w', padx=10, pady=(10, 0))
        
        tk.Label(
            status_card, 
            text=status_text, 
            font=("Segoe UI", 14), 
            bg='white'
        ).grid(row=1, column=0, sticky='w', padx=10, pady=(0, 10))
        
        # Metric cards
        metrics_frame = tk.Frame(user_win, bg=self.light_color)
        metrics_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        # Card 1: Present Days
        present_days = len([r for r in self.attendance_system.attendance_log 
                          if r["Name"] == self.current_user and r["Check-in"]])
        self._create_metric_card(metrics_frame, "Present Days", present_days, self.secondary_color, 0, 0)
        
        # Card 2: Avg Hours
        avg_hours = self._calculate_avg_hours(self.current_user)
        self._create_metric_card(metrics_frame, "Avg Hours/Day", f"{avg_hours:.1f}h", self.primary_color, 0, 1)
        
        # Card 3: Late Arrivals
        late_days = len([r for r in self.attendance_system.attendance_log 
                        if r["Name"] == self.current_user and self._is_late(r["Check-in"])])
        self._create_metric_card(metrics_frame, "Late Arrivals", late_days, self.warning_color, 0, 2)

        # Attendance history
        history_frame = tk.Frame(user_win)
        history_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        columns = ("Date", "Check-in", "Check-out", "Hours", "Status")
        tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=15)
        
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
        actions_frame = tk.Frame(user_win, bg=self.light_color)
        actions_frame.pack(fill='x', pady=(0, 20), padx=20)
        
        ttk.Button(
            actions_frame, 
            text="Export My Attendance", 
            command=lambda: self.export_user_data(self.current_user)
        ).pack(side='left', padx=5)
        
        ttk.Button(
            actions_frame, 
            text="Request Correction", 
            command=self.request_correction
        ).pack(side='left', padx=5)
    
    def _create_metric_card(self, parent, title, value, color, row, col):
        """Helper to create metric cards"""
        card = tk.Frame(
            parent, 
            bg='white', 
            bd=1, 
            relief='groove'
        )
        card.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
        parent.grid_columnconfigure(col, weight=1)
        
        tk.Label(
            card, 
            text=title, 
            font=("Segoe UI", 10), 
            bg='white'
        ).pack(pady=(10, 0))
        
        tk.Label(
            card, 
            text=value, 
            font=("Segoe UI", 18, 'bold'), 
            bg='white', 
            fg=color
        ).pack(pady=5)
        
        return card
    
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
            
            # Get desktop path
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            filename = f"{user_name}_attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            filepath = os.path.join(desktop, filename)
            
            # Export to Excel
            df.to_excel(filepath, index=False)
            
            messagebox.showinfo(
                "Success", 
                f"Your attendance data has been exported to:\n{filename}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def request_correction(self):
        """Handle attendance correction requests"""
        messagebox.showinfo("Request Sent", "Your correction request has been submitted to HR")


# Run the application
if __name__ == "__main__":
    app = AttendanceUI()