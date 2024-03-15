#!/usr/bin/env python3
"""
Writing a script that creates the database db_0 in your MySQL server.
"""

import mysql.connector

def create_database_if_missing():
    try:
        # Connect to MySQL server
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Ag017@@@"
        )

        # Create a cursor
        cursor = connection.cursor()

        # Execute the CREATE DATABASE query
        cursor.execute("CREATE DATABASE IF NOT EXISTS db_0")

        # Commit the changes
        connection.commit()

        print("Database db_0 created successfully (if it did not already exist).")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        # Close the cursor and connection
        cursor.close()
        connection.close()

# Call the function to create the database
create_database_if_missing()
