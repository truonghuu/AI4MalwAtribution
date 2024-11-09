import os
import sqlalchemy
import sqlite3


def initialise_db():
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS results (hash TEXT PRIMARY KEY, name TEXT, score TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS names (name TEXT , hash TEXT, FOREIGN KEY (hash) REFERENCES results (hash))")
    connection.commit()

initialise_db()

def connect():
    try:
        connection = sqlite3.connect("database.db")
        cursor = connection.cursor()
        return connection,cursor
    except Exception as e:
        print(f"Connection Error: {e}")
    
# Search for a file by name
def searchByName(name):
    connection, cursor = connect()
    cursor.execute("SELECT hash FROM names WHERE name = ?", (name,))
    matching_hashes = cursor.fetchall()

    # Each hash is one upload with the specified name.
    # Get the other names that has been used to upload same file
    matching_hashes = dict(matching_hashes)
    for hash in matching_hashes:
        cursor.execute("SELECT name from names WHERE hash = ?", (hash,))
        alternate_names = cursor.fetchall()
        matching_hashes[hash] = alternate_names
    

    # Format: Search for notepad.exe
    # Found 3 hashes with that
    # matching_hashes: { hash1:"notepad.exe, npad.exe, notepadwindows.exe", hash2:"notepad.exe, paint.exe, nonsense.exe"}
    return matching_hashes

# Search by hash
def searchByHash(hash):
    connection, cursor = connect()
    cursor.execute("SELECT score FROM results WHERE hash = ?", (hash,))
    result = cursor.fetchall()
    return result