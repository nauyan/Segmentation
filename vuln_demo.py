import os
import sqlite3


def run_backup(user_input):
    # Command injection: unsanitized user input passed straight to the shell
    os.system("tar -czf backup.tar.gz " + user_input)


def get_user(username):
    conn = sqlite3.connect("app.db")
    # SQL injection: user input concatenated directly into the query
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    return conn.execute(query).fetchall()
