import streamlit as st
import streamlit_authenticator as stauth
import yaml

st.set_page_config(page_title="Test Login")

with open("credentials.yaml", "r") as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name, auth_status, username = authenticator.login("Đăng nhập", "sidebar")  # Thử "sidebar"

if auth_status:
    st.success(f"Chào mừng {name}!")
    authenticator.logout("Đăng xuất", "sidebar")
elif auth_status == False:
    st.error("Sai thông tin đăng nhập!")
elif auth_status is None:
    st.warning("Nhập thông tin đăng nhập!")