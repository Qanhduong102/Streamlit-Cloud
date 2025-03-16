import streamlit as st
import pandas as pd
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import joblib
from io import BytesIO
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Cấu hình trang
st.set_page_config(page_title="Phân tích Hành vi Mua sắm", layout="wide", page_icon="📊", initial_sidebar_state="expanded")

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stSidebar {background-color: #e8eef3;}
    .stButton>button {background-color: #2e7bcf; color: white; border-radius: 5px;}
    .stButton>button:hover {background-color: #1e5b9f;}
    h1 {color: #2e7bcf; font-family: 'Arial', sans-serif;}
    .stTab {background-color: #ffffff; border-radius: 10px; padding: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
    .footer {
        background: linear-gradient(90deg, #2e7bcf, #4a90e2);
        color: #ffffff;
        text-align: center;
        padding: 15px 0;
        position: relative;
        bottom: 0;
        width: 100%;
        font-family: 'Arial', sans-serif;
        font-size: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
    }
    .footer a {
        color: #ffffff;
        text-decoration: none;
        margin: 0 10px;
        transition: color 0.3s ease;
    }
    .footer a:hover {
        color: #ffd700; /* Màu vàng khi hover để tạo điểm nhấn */
    }
    .footer .social-icons a {
        font-size: 18px;
        margin: 0 8px;
    }
    @media (max-width: 768px) {
        .footer {padding: 10px 0;}
        .footer .social-icons a {margin: 0 5px;}
    }
    </style>
""", unsafe_allow_html=True)

# Tải dữ liệu từ Google Sheets
@st.cache_data
def load_data_from_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open("Purchase Data").sheet1  # Thay "Purchase Data" bằng tên Sheets của bạn nếu khác
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    # Chuyển đổi kiểu dữ liệu cho khớp với code cũ
    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
    df['Total Purchase Amount'] = df['Product Price'].astype(float) * df['Quantity'].astype(float)
    df['Customer ID'] = df['Customer ID'].astype(int)
    df['Returns'] = df['Returns'].astype(float)
    df['Age'] = df['Age'].astype(int)
    df['Churn'] = df['Churn'].astype(int)
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    customer_segments = pd.read_csv('customer_segments.csv')  # Giữ file này local tạm thời
    return df, customer_segments

# Tải mô hình
@st.cache_resource
def load_models():
    churn_model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    revenue_model = joblib.load('revenue_model.pkl')
    return churn_model, scaler, revenue_model

df, customer_segments = load_data_from_sheets()
churn_model, scaler, revenue_model = load_models()

# Header
st.title("📊 Hệ thống Phân tích Hành vi Mua sắm Chuyên nghiệp")
st.markdown("**Khám phá dữ liệu, phân khúc khách hàng và dự đoán với giao diện tối ưu!**", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔍 Bộ lọc Dữ liệu")
    category_filter = st.multiselect("Danh mục sản phẩm", options=['Tất cả'] + sorted(df['Product Category'].unique()), default=['Tất cả'])
    gender_filter = st.multiselect("Giới tính", options=['Tất cả'] + sorted(df['Gender'].unique()), default=['Tất cả'])
    date_range = st.date_input("Phạm vi ngày", value=(df['Purchase Date'].min(), df['Purchase Date'].max()), 
                               min_value=df['Purchase Date'].min(), max_value=df['Purchase Date'].max())
    st.markdown("---")
    st.caption(f"Cập nhật lần cuối: {pd.Timestamp.now().strftime('%d/%m/%Y')}")

# Lọc dữ liệu
filtered_df = df.copy()
if 'Tất cả' not in category_filter:
    filtered_df = filtered_df[filtered_df['Product Category'].isin(category_filter)]
if 'Tất cả' not in gender_filter:
    filtered_df = filtered_df[filtered_df['Gender'].isin(gender_filter)]
filtered_df = filtered_df[(filtered_df['Purchase Date'] >= pd.to_datetime(date_range[0])) & 
                          (filtered_df['Purchase Date'] <= pd.to_datetime(date_range[1]))]

# Tổng quan
st.write(f"**Tổng quan dữ liệu lọc**: {len(filtered_df):,} giao dịch | Tổng doanh thu: {filtered_df['Total Purchase Amount'].sum():,.0f} VND")

# Tabs
tabs = st.tabs(["📈 Phân tích Cơ bản", "👥 Phân khúc Khách hàng", "⚠️ Dự đoán Churn", "📅 Xu hướng Thời gian", 
                "👤 Chi tiết Khách hàng", "📦 Phân tích Hoàn trả"])

# Tab 1: Phân tích Cơ bản
with tabs[0]:
    st.subheader("Phân tích Cơ bản")
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")
    with col1:
        revenue_by_category = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().reset_index()
        fig1 = px.bar(revenue_by_category, x='Product Category', y='Total Purchase Amount', 
                      title="Doanh thu theo Danh mục", color='Product Category', text_auto='.2s', height=400)
        fig1.update_traces(textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        purchases_by_day = filtered_df.groupby(filtered_df['Purchase Date'].dt.date)['Customer ID'].count().reset_index()
        fig2 = px.line(purchases_by_day, x='Purchase Date', y='Customer ID', 
                       title="Giao dịch theo Ngày", height=400, line_shape='spline')
        st.plotly_chart(fig2, use_container_width=True)
    with col3:
        top_spenders = filtered_df.groupby('Customer ID')['Total Purchase Amount'].sum().nlargest(5).reset_index()
        fig3 = px.bar(top_spenders, x='Customer ID', y='Total Purchase Amount', 
                      title="Top 5 Khách hàng", text=top_spenders['Customer ID'].astype(str), 
                      color_discrete_sequence=['#ff6f61'], height=400)
        fig3.update_traces(textposition='outside')
        st.plotly_chart(fig3, use_container_width=True)
    st.subheader("Gợi ý Hành động")
    low_transaction_day = filtered_df.groupby('Day of Week')['Customer ID'].count().idxmin()
    st.write(f"- Tăng khuyến mãi vào {low_transaction_day} (ngày ít giao dịch nhất).")
    top_category = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().idxmax()
    st.write(f"- Tập trung quảng bá {top_category} (danh mục doanh thu cao nhất).")

# Tab 2: Phân khúc Khách hàng
with tabs[1]:
    st.subheader("Phân khúc Khách hàng")
    with st.expander("🔎 Chi tiết các nhóm khách hàng", expanded=False):
        st.dataframe(customer_segments.groupby('Cluster').mean().style.format("{:.2f}").background_gradient(cmap='Blues'))
    avg_spending = customer_segments.groupby('Cluster')['Total Purchase Amount'].mean().reset_index()
    fig4 = px.bar(avg_spending, x='Cluster', y='Total Purchase Amount', 
                  title="Chi tiêu Trung bình theo Nhóm", color='Cluster', 
                  text=avg_spending['Total Purchase Amount'].round(2), height=400)
    fig4.update_traces(textposition='outside')
    st.plotly_chart(fig4, use_container_width=True)

# Tab 3: Dự đoán Churn
with tabs[2]:
    st.subheader("Dự đoán Khách hàng Rời bỏ")
    col1, col2 = st.columns([2, 1])
    with col1:
        customer_id = st.number_input("Nhập Customer ID:", min_value=1, step=1, format="%d")
    with col2:
        predict_button = st.button("Dự đoán", key="predict", use_container_width=True)
    if predict_button:
        customer_data = customer_segments[customer_segments['Customer ID'] == customer_id]
        if not customer_data.empty:
            X = scaler.transform(customer_data[['Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']])
            churn_pred = churn_model.predict(X)[0]
            st.success(f"Khách hàng {customer_id} {'có nguy cơ rời bỏ' if churn_pred else 'không rời bỏ'}", icon="✅")
            if churn_pred:
                st.write("**Gợi ý**: Gửi ưu đãi giảm giá hoặc email cá nhân hóa để giữ chân khách hàng này.")
        else:
            st.error(f"Không tìm thấy khách hàng {customer_id}!", icon="❌")

# Tab 4: Xu hướng Thời gian
with tabs[3]:
    st.subheader("Xu hướng Theo Thời gian")
    monthly_revenue = filtered_df.groupby(filtered_df['Purchase Date'].dt.to_period('M'))['Total Purchase Amount'].sum().reset_index()
    monthly_revenue['Month_Num'] = np.arange(len(monthly_revenue))
    monthly_revenue['Purchase Date'] = monthly_revenue['Purchase Date'].astype(str)
    fig5 = px.line(monthly_revenue, x='Purchase Date', y='Total Purchase Amount', 
                   title="Doanh thu Theo Tháng", height=400, line_shape='spline')
    st.plotly_chart(fig5, use_container_width=True)
    future_months = np.arange(len(monthly_revenue), len(monthly_revenue) + 3).reshape(-1, 1)
    future_pred = revenue_model.predict(future_months)
    st.write("Dự đoán doanh thu 3 tháng tới:")
    st.line_chart(pd.DataFrame({'Dự đoán': future_pred}, index=[f"Tháng {i+1}" for i in range(3)]))

# Tab 5: Chi tiết Khách hàng
with tabs[4]:
    st.subheader("Chi tiết Khách hàng")
    customer_id = st.number_input("Nhập Customer ID để xem chi tiết:", min_value=1, step=1)
    customer_data = filtered_df[filtered_df['Customer ID'] == customer_id]
    if not customer_data.empty:
        st.write(f"Tổng chi tiêu: {customer_data['Total Purchase Amount'].sum():,.0f} VND")
        st.dataframe(customer_data[['Purchase Date', 'Product Category', 'Total Purchase Amount', 'Returns']])
        fig = px.line(customer_data, x='Purchase Date', y='Total Purchase Amount', 
                      title=f"Lịch sử mua sắm của {customer_id}", height=400)
        st.plotly_chart(fig)
    else:
        st.warning("Không tìm thấy khách hàng này!")

# Tab 6: Phân tích Hoàn trả
with tabs[5]:
    st.subheader("Phân tích Hoàn trả")
    return_rate = filtered_df.groupby('Product Category')['Returns'].mean().reset_index()
    fig6 = px.bar(return_rate, x='Product Category', y='Returns', 
                  title="Tỷ lệ Hoàn trả theo Danh mục", text_auto='.2%', height=400)
    fig6.update_traces(textposition='outside')
    st.plotly_chart(fig6, use_container_width=True)

# Xuất báo cáo PDF
def generate_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFillColorRGB(0.18, 0.48, 0.81)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Báo cáo Phân tích Hành vi Mua sắm")
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Tổng doanh thu: {filtered_df['Total Purchase Amount'].sum():,.0f} VND")
    c.drawString(100, 710, f"Số giao dịch: {len(filtered_df):,}")
    c.drawString(100, 690, f"Top danh mục: {filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().idxmax()}")
    c.drawString(100, 670, f"Nhóm khách hàng chi tiêu cao nhất: Cluster {customer_segments.groupby('Cluster')['Total Purchase Amount'].mean().idxmax()}")
    c.drawString(100, 650, f"Dự đoán doanh thu tháng tới: {int(revenue_model.predict([[len(monthly_revenue)]])):,} VND")
    c.line(100, 640, 500, 640)
    c.drawString(100, 620, f"Ngày cập nhật: {pd.Timestamp.now().strftime('%d/%m/%Y')}")
    c.save()
    buffer.seek(0)
    return buffer

with st.sidebar:
    st.markdown("---")
    if st.button("📥 Xuất Báo cáo PDF", key="export", use_container_width=True):
        pdf_buffer = generate_pdf()
        st.download_button(label="Tải Báo cáo PDF", data=pdf_buffer, file_name="purchase_analysis_report.pdf", 
                           mime="application/pdf", use_container_width=True)
        st.success("Báo cáo đã sẵn sàng để tải!", icon="📄")

# Footer
st.markdown("""
    <div class="footer">
        <p>© 2025 - Phát triển bởi <a href="https://example.com" target="_blank">KTDL-9 Team</a></p>
        <div class="social-icons">
            <a href="https://facebook.com" target="_blank"><i class="fab fa-facebook-f"></i></a>
            <a href="https://twitter.com" target="_blank"><i class="fab fa-twitter"></i></a>
            <a href="https://linkedin.com" target="_blank"><i class="fab fa-linkedin-in"></i></a>
            <a href="mailto:contact@ktdl9team.com"><i class="fas fa-envelope"></i></a>
        </p>
    </div>
""", unsafe_allow_html=True)