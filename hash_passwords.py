import streamlit_authenticator as stauth
import sys

def hash_password(password: str):
    """Handles both old and new versions of streamlit_authenticator."""
    try:
        # ✅ New versions (>= 0.3.0): static method
        return stauth.Hasher.hash(password)
    except AttributeError:
        # ✅ Old versions (< 0.3.0): class-based interface
        return stauth.Hasher([password]).generate()[0]

if len(sys.argv) > 1:
    password = sys.argv[1]
    hashed = hash_password(password)
    print(f"Password: {password}")
    print(f"Hashed: {hashed}")
else:
    print("Usage: python hash_passwords.py 'YourNewPassword'")
