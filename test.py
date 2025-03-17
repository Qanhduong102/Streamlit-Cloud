import streamlit as st
import streamlit_authenticator as stauth
import yaml

# Đọc file credentials.yaml
with open('credentials.yaml') as file:
    config = yaml.safe_load(file)

# Khởi tạo authenticator
authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    cookie_key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days']
)

# Hiển thị form đăng nhập
authentication_status = authenticator.login(fields={'Form name': 'Đăng nhập'}, location='main')

# Kiểm tra trạng thái đăng nhập
if st.session_state.get('authentication_status') is None:
    st.warning("Vui lòng nhập thông tin đăng nhập")
elif st.session_state.get('authentication_status') is False:
    st.error("Sai tên đăng nhập hoặc mật khẩu")
elif st.session_state.get('authentication_status'):
    st.write(f"Chào mừng {st.session_state['name']}!")
    authenticator.logout("Đăng xuất", "main")