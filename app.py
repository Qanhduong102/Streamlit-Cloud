import streamlit as st
import pandas as pd
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import joblib
from io import BytesIO

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
    </style>
""", unsafe_allow_html=True)

# Tải dữ liệu và mô hình
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_customer_data.csv', parse_dates=['Purchase Date'])
    customer_segments = pd.read_csv('customer_segments.csv')
    return df, customer_segments

@st.cache_resource
def load_models():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

df, customer_segments = load_data()
model, scaler = load_models()

# Header
st.title("📊 Hệ thống Phân tích Hành vi Mua sắm Chuyên nghiệp")
st.markdown("**Khám phá dữ liệu, phân khúc khách hàng và dự đoán churn với giao diện tối ưu!**", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔍 Bộ lọc Dữ liệu")
    category_filter = st.multiselect(
        "Danh mục sản phẩm",
        options=['Tất cả'] + sorted(df['Product Category'].unique()),
        default=['Tất cả'],
        help="Chọn nhiều danh mục để phân tích"
    )
    gender_filter = st.multiselect(
        "Giới tính",
        options=['Tất cả'] + sorted(df['Gender'].unique()),
        default=['Tất cả'],
        help="Lọc theo giới tính khách hàng"
    )
    date_range = st.date_input(
        "Phạm vi ngày",
        value=(df['Purchase Date'].min(), df['Purchase Date'].max()),
        min_value=df['Purchase Date'].min(),
        max_value=df['Purchase Date'].max(),
        help="Chọn khoảng thời gian"
    )
    st.markdown("---")
    uploaded_file = st.file_uploader("Tải lên file CSV mới", type="csv")
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file, parse_dates=['Purchase Date'])
        new_df['Total Purchase Amount'] = new_df['Product Price'] * new_df['Quantity']
        new_df.to_csv('cleaned_customer_data.csv', index=False)
        st.success("Đã cập nhật dữ liệu mới!", icon="✅")
        st.experimental_rerun()  # Tải lại trang để cập nhật dữ liệu
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

# Hiển thị thông tin tổng quan
st.write(f"**Tổng quan dữ liệu lọc**: {len(filtered_df):,} giao dịch | Tổng doanh thu: {filtered_df['Total Purchase Amount'].sum():,.0f} VND")

# Tabs
tabs = st.tabs(["📈 Phân tích Cơ bản", "👥 Phân khúc Khách hàng", "⚠️ Dự đoán Churn", "📅 Xu hướng Thời gian"])

# Tab 1: Phân tích cơ bản
with tabs[0]:
    st.subheader("Phân tích Cơ bản")
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")
    with col1:
        revenue_by_category = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().reset_index()
        fig1 = px.bar(revenue_by_category, x='Product Category', y='Total Purchase Amount', 
                      title="Doanh thu theo Danh mục", color='Product Category', 
                      text_auto='.2s', height=400)
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

# Tab 2: Phân khúc khách hàng
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
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            customer_id = st.number_input("Nhập Customer ID:", min_value=1, step=1, format="%d", help="Nhập ID để dự đoán")
        with col2:
            predict_button = st.button("Dự đoán", key="predict", use_container_width=True)
        if predict_button:
            customer_data = customer_segments[customer_segments['Customer ID'] == customer_id]
            if not customer_data.empty:
                X = scaler.transform(customer_data[['Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']])
                churn_pred = model.predict(X)[0]
                st.success(f"Khách hàng {customer_id} {'có nguy cơ rời bỏ' if churn_pred else 'không rời bỏ'}", icon="✅")
            else:
                st.error(f"Không tìm thấy khách hàng {customer_id}!", icon="❌")

# Tab 4: Xu hướng thời gian
with tabs[3]:
    st.subheader("Xu hướng Theo Thời gian")
    monthly_revenue = filtered_df.groupby(filtered_df['Purchase Date'].dt.to_period('M'))['Total Purchase Amount'].sum().reset_index()
    monthly_revenue['Purchase Date'] = monthly_revenue['Purchase Date'].astype(str)  # Chuyển Period thành chuỗi
    fig5 = px.line(monthly_revenue, x='Purchase Date', y='Total Purchase Amount', 
                   title="Doanh thu Theo Tháng", height=400, line_shape='spline')
    st.plotly_chart(fig5, use_container_width=True)

# Xuất báo cáo PDF
def generate_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFillColorRGB(0.18, 0.48, 0.81)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Báo cáo Phân tích Hành vi Mua sắm")
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Tổng doanh thu (lọc): {filtered_df['Total Purchase Amount'].sum():,.0f} VND")
    c.drawString(100, 710, f"Số giao dịch: {len(filtered_df):,}")
    c.drawString(100, 690, f"Ngày cập nhật: {pd.Timestamp.now().strftime('%d/%m/%Y')}")
    c.line(100, 680, 500, 680)
    c.save()
    buffer.seek(0)
    return buffer

with st.sidebar:
    st.markdown("---")
    if st.button("📥 Xuất Báo cáo PDF", key="export", use_container_width=True):
        pdf_buffer = generate_pdf()
        st.download_button(
            label="Tải Báo cáo PDF",
            data=pdf_buffer,
            file_name="purchase_analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        st.success("Báo cáo đã sẵn sàng để tải!", icon="📄")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>© 2025 - Phát triển bởi KTDL-9 Team</p>", unsafe_allow_html=True)