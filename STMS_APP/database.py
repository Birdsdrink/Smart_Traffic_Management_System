# dashboard/database.py

import mysql.connector

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "vehicle_data"
}

def insert_into_database(track_id, speed, timestamp, model, color, company, number_plate):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO vehicle_records (track_id, speed, date_time, vehicle_model, vehicle_color, vehicle_company, number_plate)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (track_id, speed, timestamp, model, color, company, number_plate))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ DB: Inserted vehicle {track_id}")
    except Exception as e:
        print(f"❌ DB Error: {e}")
