from modules import FaceProcessor, AttendanceSystem, AttendanceUI

def main():
    # Initialize the core logic class
    attendance_system = AttendanceSystem()
    
    # Pass attendance system to face processor
    processor = FaceProcessor(attendance_system)
    
    #And run. 
    ui = AttendanceUI()

if __name__ == "__main__":
    main() 