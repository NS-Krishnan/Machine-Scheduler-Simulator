# tests/test_login.py
import pytest
from login import validate_user, create_user, create_connection

@pytest.fixture
def db_connection():
    conn = create_connection(':memory:')  # use an in-memory database for testing
    conn.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            password TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone TEXT NOT NULL
        );
    ''')
    yield conn
    conn.close()

def test_create_user(db_connection):
    user = ('testuser', 'password', 'test@example.com', '1234567890')
    user_id = create_user(db_connection, user)
    assert user_id is not None

def test_validate_user(db_connection):
    user = ('testuser', 'password', 'test@example.com', '1234567890')
    create_user(db_connection, user)
    is_valid, user_name = validate_user(db_connection, 'testuser', 'password')
    assert is_valid is True
    assert user_name == 'testuser'

    # Test invalid credentials
    is_valid, user_name = validate_user(db_connection, 'wronguser', 'wrongpassword')
    assert is_valid is False
