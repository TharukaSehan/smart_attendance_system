import sqlite3

conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM attendance")
rows = cursor.fetchall()

print("\n--- 📊 ATTENDANCE DATABASE RECORDS ---")
for row in rows:
    print(f"ID: {row[0]} | Name: {row[1]} | Time: {row[2]}")

conn.close()