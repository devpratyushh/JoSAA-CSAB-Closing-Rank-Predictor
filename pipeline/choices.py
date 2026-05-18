import sqlite3
import os

DB_PATH = "user_choices.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS choices (
                institute TEXT,
                program TEXT,
                PRIMARY KEY (institute, program)
            )
        """)

def get_choices() -> set[tuple[str, str]]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT institute, program FROM choices")
        return {(row[0], row[1]) for row in cursor}

def toggle_choice(institute: str, program: str, is_starred: bool):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        if is_starred:
            conn.execute(
                "INSERT OR IGNORE INTO choices (institute, program) VALUES (?, ?)", 
                (institute, program)
            )
        else:
            conn.execute(
                "DELETE FROM choices WHERE institute=? AND program=?", 
                (institute, program)
            )
