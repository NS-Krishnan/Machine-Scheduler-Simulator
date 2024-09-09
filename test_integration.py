import pytest
import sqlite3
from login import create_connection, create_user, validate_user

# Fixture to set up and tear down the in-memory database
@pytest.fixture(scope="module")
def db_connection():
    # Use an in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    # Create the users table
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY,
                     name TEXT NOT NULL,
                     password TEXT NOT NULL,
                     email TEXT NOT NULL UNIQUE,
                     phone TEXT NOT NULL);''')
    yield conn
    conn.close()

# Test case to check user registration and retrieval from the database
def test_user_registration_and_retrieval(db_connection):
    # New user data
    user_data = ('newuser', 'newpassword', 'newuser@example.com', '9876543210')
    
    # Register the new user
    user_id = create_user(db_connection, user_data)
    
    # Verify the user is added to the database
    cur = db_connection.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (user_id,))
    user = cur.fetchone()
    
    assert user is not None, "User should be present in the database"
    assert user[1] == 'newuser', "User name should match"
    assert user[2] == 'newpassword', "Password should match"
    assert user[3] == 'newuser@example.com', "Email should match"
    assert user[4] == '9876543210', "Phone number should match"

# Test case to validate user login with correct credentials
def test_validate_user_login(db_connection):
    # Assuming the user has already been added in the previous test
    is_valid, user_name = validate_user(db_connection, 'newuser', 'newpassword')
    
    assert is_valid, "User login should be valid"
    assert user_name == 'newuser', "The returned username should match the one in the database"
