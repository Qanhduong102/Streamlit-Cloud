import bcrypt
import logging

# Tắt cảnh báo của Streamlit (không cần nếu không dùng Streamlit)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Mật khẩu cần mã hóa
password = 'Qanhduong102'

# Mã hóa bằng bcrypt
hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
print(hashed.decode('utf-8'))