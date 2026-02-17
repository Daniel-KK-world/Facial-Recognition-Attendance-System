import sqlite3
from pathlib import Path
from datetime import datetime

# This tells Python where to save the database file
DB_PATH = Path(__file__).resolve().parents[1] / "attendance.db"

def init_database():
    """Create the database and table if they don't exist"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Create table (like a spreadsheet with columns)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        date TEXT NOT NULL,
        check_in TEXT,
        check_out TEXT
    )
    """)
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}")

def add_attendance_record(name, date, check_in=None, check_out=None):
    """Add or update an attendance record"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Check if record exists
    cursor.execute("SELECT id FROM attendance WHERE name = ? AND date = ?", (name, date))
    existing = cursor.fetchone()
    
    if existing:
        # Update existing record
        cursor.execute("""
        UPDATE attendance 
        SET check_in = COALESCE(?, check_in), 
            check_out = COALESCE(?, check_out)
        WHERE name = ? AND date = ?
        """, (check_in, check_out, name, date))
    else:
        # Create new record
        cursor.execute("""
        INSERT INTO attendance (name, date, check_in, check_out)
        VALUES (?, ?, ?, ?)
        """, (name, date, check_in, check_out))
    
    conn.commit()
    conn.close()

def get_all_records():
    """Get all attendance records (like reading the entire CSV)"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("SELECT name, date, check_in, check_out FROM attendance ORDER BY date DESC")
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to dictionary format (same as your CSV)
    records = []
    for row in rows:
        records.append({
            "Name": row[0],
            "Date": row[1],
            "Check-in": row[2] if row[2] else "",
            "Check-out": row[3] if row[3] else ""
        })
    return records

def get_today_records(date):
    """Get all records for a specific date"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("SELECT name, date, check_in, check_out FROM attendance WHERE date = ?", (date,))
    rows = cursor.fetchall()
    conn.close()
    
    records = []
    for row in rows:
        records.append({
            "Name": row[0],
            "Date": row[1],
            "Check-in": row[2] if row[2] else "",
            "Check-out": row[3] if row[3] else ""
        })
    return records

# Initialize when imported
init_database()