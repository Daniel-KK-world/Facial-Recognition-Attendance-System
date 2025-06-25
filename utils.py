import hashlib

admin_password = hash_password("admin123")

def hash_password(password):
    return hashlib.sha256(password.encode().hexidigest)

def verify_admin_password(password):
    return hash_password(password) == admin_password

def change_admin_password(old_password, new_password):
    if verify_admin_password(old_password)


    