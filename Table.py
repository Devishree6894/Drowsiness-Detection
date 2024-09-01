import sqlite3
from datetime import datetime

conn = sqlite3.connect('driver_status4.db')
cursor = conn.cursor()

# Create table to store status information if not exists
# cursor.execute('''DROP TABLE driver_status''')
# cursor.execute('''DELETE FROM driver_status''')
# cursor.execute('''CREATE TABLE IF NOT EXISTS driver_status
                # (id INTEGER PRIMARY KEY AUTOINCREMENT,
                # time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                # state TEXT)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS driver_status
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TIMESTAMP,
                state TEXT)''')


conn.commit()
conn.close()
