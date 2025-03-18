import streamlit as st
import streamlit_authenticator as stauth
import yaml
import pandas as pd
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import joblib
from io import BytesIO
import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import os
from google.oauth2 import service_account
import gspread
import json  # Đảm bảo import json để parse credentials_json

customer_segments = pd.DataFrame()

# Lấy thông tin từ biến môi trường
credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

# Tăng giới hạn số ô tối đa cho Pandas Styler
pd.set_option("styler.render.max_elements", 998336)

# Cấu hình trang
st.set_page_config(page_title="Phân tích Hành vi Mua sắm", layout="wide", page_icon="📊", initial_sidebar_state="expanded")

# Đọc file CSS
with open("styles.css", "r", encoding="utf-8") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Đọc file cấu hình tài khoản
with open("credentials.yaml", "r") as file:
    config = yaml.safe_load(file)

# Khởi tạo authenticator
authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    cookie_key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days']
)

# Giao diện đăng nhập
if st.session_state.get('authentication_status') is None or st.session_state.get('authentication_status') is False:
    st.markdown("""
        <div class="login-container">
            <div class="login-box">
                <div class="login-header">
                    <img src="https://img.icons8.com/fluency/96/analytics.png" alt="Logo">
                    <h2>Hệ thống Phân tích</h2>
                    <p>Đăng nhập để khám phá dữ liệu mua sắm</p>
                </div>
    """, unsafe_allow_html=True)

    # Form đăng nhập
    authentication_status = authenticator.login(
        fields={'Form name': '', 'Username': 'Tên người dùng', 'Password': 'Mật khẩu', 'Login': 'Đăng nhập'},
        location='main'
    )

    # Thông báo trạng thái
    if st.session_state.get('authentication_status') is False:
        st.markdown('<div class="error-message">Sai tên người dùng hoặc mật khẩu!</div>', unsafe_allow_html=True)
    elif st.session_state.get('authentication_status') is None:
        st.markdown('<div class="warning-message">Vui lòng nhập thông tin đăng nhập.</div>', unsafe_allow_html=True)

    st.markdown("""
            <div class="login-footer">
                © 2025 KTDL-9 Team. All rights reserved.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Nếu đăng nhập thành công
elif st.session_state.get('authentication_status'):
    name = st.session_state.get('name', 'Người dùng')
    st.markdown(f'<div class="success-message">Chào mừng {name}!</div>', unsafe_allow_html=True)
    authenticator.logout("Đăng xuất", "sidebar")

   # Tải dữ liệu (ưu tiên Google Sheets nếu có credentials, nếu không dùng file CSV)
    @st.cache_data
    def load_data():
        if credentials_json:
            try:
                credentials_dict = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(credentials_dict)
                gc = gspread.authorize(credentials)
                sheet = gc.open("Purchase Data").sheet1
                raw_data = sheet.get_all_records()

                clean_data = []
                for row in raw_data:
                    clean_row = {}
                    for key, value in row.items():
                        if isinstance(value, str):
                            clean_row[key] = ''.join(char for char in value if ord(char) >= 32 or char in '\n\t\r')
                        else:
                            clean_row[key] = value if value is not None else None
                    clean_data.append(clean_row)

                df = pd.DataFrame(clean_data)
                required_columns = ['Customer ID', 'Purchase Date', 'Product Category', 'Product Price', 'Quantity',
                              'Total Purchase Amount', 'Payment Method', 'Returns', 'Customer Name', 'Age',
                              'Gender', 'Churn', 'Year', 'Month', 'Day of Week']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    st.error(f"Các cột thiếu trong dữ liệu Google Sheets: {missing_cols}")
                    return None, None

                df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
                df['Product Price'] = pd.to_numeric(df['Product Price'], errors='coerce').fillna(0.0)
                df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0.0)
                df['Total Purchase Amount'] = pd.to_numeric(df['Total Purchase Amount'], errors='coerce').fillna(0.0)
                df['Customer ID'] = pd.to_numeric(df['Customer ID'], errors='coerce').fillna(0).astype(int)
                df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce').fillna(0.0)
                df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median()).round(0).astype(int)
                df['Gender'] = df['Gender'].astype(str).fillna('Unknown')
                df['Payment Method'] = df['Payment Method'].astype(str).fillna('Unknown')
                df['Customer Name'] = df['Customer Name'].astype(str).fillna('Unknown')
                df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce').fillna(0).astype(int)
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
                df['Month'] = pd.to_numeric(df['Month'], errors='coerce').fillna(0).astype(int)
                df['Day of Week'] = df['Day of Week'].astype(str).fillna('Unknown')

                # Tải customer_segments
                segment_sheet = gc.open("Customer Segments").sheet1
                raw_segment_data = segment_sheet.get_all_records()
                clean_segment_data = []
                for row in raw_segment_data:
                    clean_row = {}
                    for key, value in row.items():
                        if isinstance(value, str):
                            clean_row[key] = ''.join(char for char in value if ord(char) >= 32 or char in '\n\t\r')
                        else:
                            clean_row[key] = value if value is not None else None
                    clean_segment_data.append(clean_row)
                customer_segments = pd.DataFrame(clean_segment_data)
                required_segment_cols = ['Customer ID', 'Customer Name', 'Total Purchase Amount', 'Transaction Count',
                                  'Returns', 'Age', 'Gender', 'Cluster', 'Churn Probability']
                missing_segment_cols = [col for col in required_segment_cols if col not in customer_segments.columns]
                if missing_segment_cols:
                    st.error(f"Các cột thiếu trong Customer Segments từ Google Sheets: {missing_segment_cols}")
                    return None, None
            except Exception as e:
                print(f"Lỗi khi tải dữ liệu từ Google Sheets: {e}")
                st.info("Sử dụng file CSV cục bộ thay thế.")
                df = pd.read_csv("cleaned_customer_data.csv")
                df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
                df['Product Price'] = pd.to_numeric(df['Product Price'], errors='coerce').fillna(0.0)
                df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0.0)
                df['Total Purchase Amount'] = pd.to_numeric(df['Total Purchase Amount'], errors='coerce').fillna(0.0)
                df['Customer ID'] = pd.to_numeric(df['Customer ID'], errors='coerce').fillna(0).astype(int)
                df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce').fillna(0.0)
                df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median()).round(0).astype(int)
                df['Gender'] = df['Gender'].astype(str).fillna('Unknown')
                df['Payment Method'] = df['Payment Method'].astype(str).fillna('Unknown')
                df['Customer Name'] = df['Customer Name'].astype(str).fillna('Unknown')
                df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce').fillna(0).astype(int)
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
                df['Month'] = pd.to_numeric(df['Month'], errors='coerce').fillna(0).astype(int)
                df['Day of Week'] = df['Day of Week'].astype(str).fillna('Unknown')
                customer_segments = pd.read_csv('customer_segments.csv')
                required_segment_cols = ['Customer ID', 'Customer Name', 'Total Purchase Amount', 'Transaction Count',
                                  'Returns', 'Age', 'Gender', 'Cluster', 'Churn Probability']
                missing_segment_cols = [col for col in required_segment_cols if col not in customer_segments.columns]
                if missing_segment_cols:
                    st.error(f"Các cột thiếu trong customer_segments.csv: {missing_segment_cols}")
                    return None, None
        else:
            df = pd.read_csv("cleaned_customer_data.csv")
            df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
            df['Product Price'] = pd.to_numeric(df['Product Price'], errors='coerce').fillna(0.0)
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0.0)
            df['Total Purchase Amount'] = pd.to_numeric(df['Total Purchase Amount'], errors='coerce').fillna(0.0)
            df['Customer ID'] = pd.to_numeric(df['Customer ID'], errors='coerce').fillna(0).astype(int)
            df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce').fillna(0.0)
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median()).round(0).astype(int)
            df['Gender'] = df['Gender'].astype(str).fillna('Unknown')
            df['Payment Method'] = df['Payment Method'].astype(str).fillna('Unknown')
            df['Customer Name'] = df['Customer Name'].astype(str).fillna('Unknown')
            df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce').fillna(0).astype(int)
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
            df['Month'] = pd.to_numeric(df['Month'], errors='coerce').fillna(0).astype(int)
            df['Day of Week'] = df['Day of Week'].astype(str).fillna('Unknown')
            customer_segments = pd.read_csv('customer_segments.csv')
            required_segment_cols = ['Customer ID', 'Customer Name', 'Total Purchase Amount', 'Transaction Count',
                              'Returns', 'Age', 'Gender', 'Cluster', 'Churn Probability']
            missing_segment_cols = [col for col in required_segment_cols if col not in customer_segments.columns]
            if missing_segment_cols:
                st.error(f"Các cột thiếu trong customer_segments.csv: {missing_segment_cols}")
                return None, None
        return df, customer_segments

    # Tải mô hình
    @st.cache_resource
    def load_models():
        try:
            churn_model = joblib.load('churn_model.pkl')
            scaler = joblib.load('scaler.pkl')
            revenue_model = joblib.load('revenue_model.pkl')
            return churn_model, scaler, revenue_model
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình: {e}")
            return None, None, None

    df, customer_segments = load_data()
    churn_model, scaler, revenue_model = load_models()

    # Kiểm tra nếu mô hình hoặc dữ liệu không tải được
    if not all([df is not None, customer_segments is not None, churn_model is not None, scaler is not None, revenue_model is not None]):
        st.error("Không thể tải dữ liệu hoặc mô hình. Vui lòng kiểm tra file hoặc cấu hình.")
    else:
        # Header
        st.title("📊 Hệ thống Phân tích Hành vi Mua sắm ")
        st.markdown("**Khám phá dữ liệu, phân khúc khách hàng và dự đoán với giao diện tối ưu!**", unsafe_allow_html=True)

        # Sidebar
        with st.sidebar:
            st.header("🔍 Bộ lọc Dữ liệu")
            category_filter = st.multiselect("Danh mục sản phẩm", options=['Tất cả'] + sorted(df['Product Category'].unique()), default=['Tất cả'])
            gender_filter = st.multiselect("Giới tính", options=['Tất cả'] + sorted(df['Gender'].unique()), default=['Tất cả'])
            payment_filter = st.multiselect("Phương thức thanh toán", options=['Tất cả'] + sorted(df['Payment Method'].unique()), default=['Tất cả'])
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
        if 'Tất cả' not in payment_filter:
            filtered_df = filtered_df[filtered_df['Payment Method'].isin(payment_filter)]
            filtered_df = filtered_df[(filtered_df['Purchase Date'] >= pd.to_datetime(date_range[0])) & 
                          (filtered_df['Purchase Date'] <= pd.to_datetime(date_range[1]))]

        # Tổng quan
        st.write(f"**Tổng quan dữ liệu lọc**: {len(filtered_df):,} giao dịch | Tổng doanh thu: {filtered_df['Total Purchase Amount'].sum():,.0f} $")

        # Tabs
        tabs = st.tabs(["📈 Phân tích Cơ bản", "👥 Phân khúc Khách hàng", "⚠️ Dự đoán Churn", "📅 Xu hướng Thời gian", 
                        "👤 Chi tiết Khách hàng", "📦 Phân tích Hoàn trả"])

        # Tab 1: Phân tích Cơ bản
        with tabs[0]:
            st.subheader("Phân tích Cơ bản")
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1], gap="small")

            with col1:
                revenue_by_category = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().reset_index()
                fig1 = px.bar(revenue_by_category, x='Product Category', y='Total Purchase Amount', 
                      title="Doanh thu theo Danh mục", color='Product Category', text_auto='.2s', height=400)
                fig1.update_traces(textposition='outside')
                st.plotly_chart(fig1, use_container_width=True, key="chart_revenue_by_category")

            with col2:
                revenue_by_day = filtered_df.groupby(filtered_df['Purchase Date'].dt.date)['Total Purchase Amount'].sum().reset_index()
                fig2 = px.line(revenue_by_day, x='Purchase Date', y='Total Purchase Amount', 
                       title="Doanh thu Theo Ngày", height=400, line_shape='spline')
                st.plotly_chart(fig2, use_container_width=True, key="chart_revenue_by_day")

            with col3:
                top_spenders = filtered_df.groupby(['Customer ID', 'Customer Name']).agg({
                    'Total Purchase Amount': 'sum',
                    'Purchase Date': 'count',
                    'Product Category': lambda x: x.mode()[0]
                }).nlargest(5, 'Total Purchase Amount').reset_index()
                top_spenders.columns = ['Customer ID', 'Customer Name', 'Total Purchase Amount', 'Transaction Count', 'Favorite Category']
                fig3 = px.bar(top_spenders, x='Customer Name', y='Total Purchase Amount', 
                      title="Top 5 Khách hàng Chi tiêu Cao nhất", 
                      text=top_spenders['Customer Name'] + ' (' + top_spenders['Transaction Count'].astype(str) + ' GD)',
                      color_discrete_sequence=['#ff6f61'], height=400)
                fig3.update_traces(textposition='outside')
                st.plotly_chart(fig3, use_container_width=True, key="chart_top_spenders")

            with col4:
                revenue_by_payment = filtered_df.groupby('Payment Method')['Total Purchase Amount'].sum().reset_index()
                fig4 = px.pie(revenue_by_payment, values='Total Purchase Amount', names='Payment Method',
                      title="Doanh thu theo Phương thức Thanh toán", height=400)
                st.plotly_chart(fig4, use_container_width=True, key="chart_revenue_by_payment")

            # Bảng chi tiết Top 5 Khách hàng
            st.subheader("Chi tiết Top 5 Khách hàng")
            st.dataframe(top_spenders.style.format({
                'Total Purchase Amount': '{:,.0f} $',
                'Transaction Count': '{:,}',
            }), height=200, use_container_width=True)
        
            # Phân tích chi tiết danh mục theo ngày
            st.subheader("Chi tiết Danh mục Theo Ngày")
            selected_category = st.selectbox("Chọn danh mục để xem chi tiết:", 
                                          options=['Tất cả'] + sorted(filtered_df['Product Category'].unique()),
                                          index=0)
        
            if selected_category == 'Tất cả':
                category_by_day = filtered_df.groupby(filtered_df['Purchase Date'].dt.date)['Total Purchase Amount'].sum().reset_index()
            else:
                category_by_day = filtered_df[filtered_df['Product Category'] == selected_category].groupby(filtered_df['Purchase Date'].dt.date)['Total Purchase Amount'].sum().reset_index()
        
            fig_category_day = px.line(category_by_day, x='Purchase Date', y='Total Purchase Amount', 
                                    title=f"Doanh thu Theo Ngày của {'Tất cả Danh mục' if selected_category == 'Tất cả' else selected_category}", 
                                    height=400, line_shape='spline')
            st.plotly_chart(fig_category_day, use_container_width=True, key="chart_category_by_day")
        
            with st.expander(f"🔎 Xem dữ liệu chi tiết của {'Tất cả Danh mục' if selected_category == 'Tất cả' else selected_category}", expanded=False):
                if selected_category == 'Tất cả':
                    detailed_data = filtered_df.groupby(['Purchase Date', 'Product Category'])['Total Purchase Amount'].sum().unstack().fillna(0)
                    limited_data = detailed_data.head(50)
                    st.write(f"**Hiển thị 50 ngày đầu tiên (tổng số ngày: {len(detailed_data)})**")
                    st.dataframe(limited_data.style.format('{:,.0f} $'), height=400, use_container_width=True)
                else:
                    detailed_data = filtered_df[filtered_df['Product Category'] == selected_category].groupby('Purchase Date')['Total Purchase Amount'].sum().reset_index()
                    st.dataframe(detailed_data.style.format('{:,.0f} $'), height=400, use_container_width=True)
        
            st.subheader("Gợi ý Hành động")
            low_transaction_day = filtered_df.groupby('Day of Week')['Customer ID'].count().idxmin()
            low_day_revenue = filtered_df.groupby('Day of Week')['Total Purchase Amount'].sum().min()
            st.write(f"- Tăng khuyến mãi 15% vào {low_transaction_day} (doanh thu thấp nhất: {low_day_revenue:,.0f} $) qua email hoặc SMS.")
        
            top_category = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().idxmax()
            top_category_revenue = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().max()
            st.write(f"- Đẩy mạnh quảng bá {top_category} (doanh thu: {top_category_revenue:,.0f} $) qua mạng xã hội và banner trên website.")
        
            st.write("- **Chiến lược cho Top Khách hàng:**")
            for vip in top_spenders['Customer ID']:
                vip_data = filtered_df[filtered_df['Customer ID'] == vip]
                last_purchase = vip_data['Purchase Date'].max()
                fav_category = vip_data['Product Category'].mode()[0]
                if (pd.Timestamp.now() - last_purchase).days > 30:
                    st.write(f"  - Khách hàng {vip}: Không hoạt động {(pd.Timestamp.now() - last_purchase).days} ngày. Gửi ưu đãi 20% cho {fav_category}.")
                else:
                    st.write(f"  - Khách hàng {vip}: Duy trì hoạt động. Tặng điểm thưởng hoặc giảm giá 10% cho {fav_category} để khuyến khích mua tiếp.")

        # Tab 2: Phân khúc Khách hàng
        with tabs[1]:
            st.subheader("Phân khúc Khách hàng")

            with st.expander("🔎 Chi tiết các nhóm khách hàng", expanded=False):
                cluster_summary = customer_segments.groupby('Cluster').agg({
                    'Total Purchase Amount': 'mean',
                    'Transaction Count': 'mean',
                    'Returns': 'mean',
                    'Age': 'mean',
                    'Customer ID': 'count'
                }).rename(columns={
                    'Total Purchase Amount': 'Chi tiêu TB ($)',
                    'Transaction Count': 'Tần suất GD TB',
                    'Returns': 'Tỷ lệ Hoàn trả TB',
                    'Age': 'Độ tuổi TB',
                    'Customer ID': 'Số lượng KH'
                })
                st.dataframe(cluster_summary.style.format({
                    'Chi tiêu TB ($)': '{:,.0f}',
                    'Tần suất GD TB': '{:.2f}',
                    'Tỷ lệ Hoàn trả TB': '{:.2%}',
                    'Độ tuổi TB': '{:.1f}',
                    'Số lượng KH': '{:,}'
                }).background_gradient(cmap='Blues'))

            avg_spending = customer_segments.groupby('Cluster')['Total Purchase Amount'].mean().reset_index()
            fig4 = px.bar(avg_spending, x='Cluster', y='Total Purchase Amount', 
                        title="Chi tiêu Trung bình theo Nhóm", color='Cluster', 
                        text=avg_spending['Total Purchase Amount'].round(2), height=400)
            fig4.update_traces(textposition='outside')
            st.plotly_chart(fig4, use_container_width=True, key="chart_avg_spending")

            cluster_compare = customer_segments.groupby('Cluster').agg({
                'Total Purchase Amount': 'mean',
                'Returns': 'mean'
            }).reset_index()
            cluster_compare['Returns'] = cluster_compare['Returns'] * 100
            fig_compare = px.scatter(cluster_compare, x='Total Purchase Amount', y='Returns', 
                                 color='Cluster', size='Total Purchase Amount',
                                 title="So sánh Chi tiêu TB và Tỷ lệ Hoàn trả",
                                 labels={'Total Purchase Amount': 'Chi tiêu TB ($)', 'Returns': 'Tỷ lệ Hoàn trả (%)'},
                                 height=400)
            st.plotly_chart(fig_compare, use_container_width=True, key="chart_cluster_compare")

            df_with_clusters = filtered_df.merge(customer_segments[['Customer ID', 'Cluster']], on='Customer ID', how='left')
            cluster_trends = df_with_clusters.groupby(['Cluster', df_with_clusters['Purchase Date'].dt.to_period('M')])['Total Purchase Amount'].sum().reset_index()
            cluster_trends['Purchase Date'] = cluster_trends['Purchase Date'].astype(str)
            fig_trends = px.line(cluster_trends, x='Purchase Date', y='Total Purchase Amount', color='Cluster',
                              title="Xu hướng Chi tiêu Theo Tháng của Các Nhóm", height=400, line_shape='spline')
            st.plotly_chart(fig_trends, use_container_width=True, key="chart_cluster_trends")

            st.subheader("Gợi ý Hành động Theo Nhóm")
            for cluster in cluster_summary.index:
                spending = cluster_summary.loc[cluster, 'Chi tiêu TB ($)']
                frequency = cluster_summary.loc[cluster, 'Tần suất GD TB']
                returns = cluster_summary.loc[cluster, 'Tỷ lệ Hoàn trả TB']
                st.write(f"**Nhóm {cluster}:**")
                if spending > cluster_summary['Chi tiêu TB ($)'].mean() and frequency < cluster_summary['Tần suất GD TB'].mean():
                    st.write(f"- Chi tiêu cao nhưng ít giao dịch: Tặng mã giảm giá định kỳ để tăng tần suất mua sắm.")
                elif returns > cluster_summary['Tỷ lệ Hoàn trả TB'].mean():
                    st.write(f"- Tỷ lệ hoàn trả cao: Cải thiện chất lượng sản phẩm hoặc kiểm tra chính sách đổi trả.")
                else:
                    st.write(f"- Nhóm ổn định: Duy trì chính sách hiện tại hoặc thử nghiệm ưu đãi nhỏ.")

            selected_cluster = st.selectbox("Chọn nhóm để xem chi tiết:", options=cluster_summary.index)
            cluster_data = customer_segments[customer_segments['Cluster'] == selected_cluster]
            st.write(f"**Thông tin chi tiết Nhóm {selected_cluster}:**")
            st.dataframe(cluster_data[['Customer ID', 'Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']])
            cluster_purchases = filtered_df[filtered_df['Customer ID'].isin(cluster_data['Customer ID'])]
            fav_categories = cluster_purchases.groupby('Product Category')['Total Purchase Amount'].sum().reset_index()
            fig_fav = px.pie(fav_categories, values='Total Purchase Amount', names='Product Category',
                          title=f"Danh mục Yêu thích của Nhóm {selected_cluster}", height=400)
            st.plotly_chart(fig_fav, use_container_width=True, key="chart_fav_categories")

            st.subheader("Phân tích Theo Giới tính")
            gender_spending = filtered_df.groupby('Gender')['Total Purchase Amount'].mean().reset_index()
            fig_gender = px.bar(gender_spending, x='Gender', y='Total Purchase Amount', 
                        title="Chi tiêu Trung bình theo Giới tính", color='Gender', 
                        text=gender_spending['Total Purchase Amount'].round(2), height=400)
            fig_gender.update_traces(textposition='outside')
            st.plotly_chart(fig_gender, use_container_width=True, key="chart_gender_spending")

        # Tab 3: Dự đoán Churn
        with tabs[2]:
            st.markdown("### Nhập thông tin khách hàng mới")
            with st.form(key='new_customer_form'):
                col1, col2 = st.columns(2)
                with col1:
                    new_customer_id = st.number_input("Customer ID", min_value=1, step=1, format="%d", key="new_customer_id")
                    new_total_purchase = st.number_input("Total Purchase Amount ($)", min_value=0.0, step=100.0, format="%.2f", key="new_total_purchase")
                    new_transaction_count = st.number_input("Transaction Count", min_value=0, step=1, format="%d", key="new_transaction_count")
                with col2:
                    new_returns = st.number_input("Returns", min_value=0, step=1, format="%d", key="new_returns")
                    new_age = st.number_input("Age", min_value=18, max_value=100, step=1, format="%d", key="new_age")
                    new_customer_name = st.text_input("Customer Name", key="new_customer_name")

                    # Sử dụng ba cột để đặt nút ở giữa
                    col_left, col_center, col_right = st.columns([0, 2, 0])
                    with col_center:
                        analyze_button = st.form_submit_button("Phân tích nguy cơ Churn", use_container_width=True)

            # Xử lý khi nhấn nút Phân tích
            if analyze_button:
                # Tạo DataFrame cho khách hàng mới
                new_customer_data = pd.DataFrame({
                    'Customer ID': [new_customer_id],
                    'Customer Name': [new_customer_name],
                    'Total Purchase Amount': [new_total_purchase],
                    'Transaction Count': [new_transaction_count],
                    'Returns': [new_returns],
                    'Age': [new_age]
                })
    
                # Chuẩn hóa dữ liệu và dự đoán
                X_new = scaler.transform(new_customer_data[['Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']])
                churn_pred = churn_model.predict(X_new)[0]
    
                st.markdown("### Kết quả phân tích khách hàng mới")
                st.dataframe(new_customer_data.style.format({
                    'Total Purchase Amount': '{:,.0f}',
                    'Transaction Count': '{:.0f}',
                    'Returns': '{:.0f}',
                    'Age': '{:.0f}'
                }), use_container_width=True)

                # Dự đoán và giải thích
                if hasattr(churn_model, 'predict_proba'):
                    churn_prob = churn_model.predict_proba(X_new)[0][1] * 100
                    prediction_text = ("có nguy cơ rời bỏ cao" if churn_prob >= 75 else 
                          "có nguy cơ rời bỏ" if churn_prob >= 50 else 
                          "không có nguy cơ rời bỏ")
                    st.success(f"Khách hàng {new_customer_id} - {new_customer_name} "
                            f"{prediction_text} (Xác suất: {churn_prob:.2f}%)", icon="✅")
        
                    # Giải thích chi tiết
                    st.markdown("#### Giải thích dự đoán")
                    st.write(f"Xác suất churn {churn_prob:.2f}% dựa trên:")
                    factors = [
                        ('Total Purchase Amount', 'Tổng chi tiêu', 'thấp hơn', 'cao hơn'),
                        ('Transaction Count', 'Số giao dịch', 'ít hơn', 'nhiều hơn'),
                        ('Returns', 'Số lần hoàn trả', 'cao hơn', 'thấp hơn'),
                        ('Age', 'Độ tuổi', 'trẻ hơn', 'lớn hơn')
                    ]
        
                    for col, name, low_text, high_text in factors:
                        value = new_customer_data[col].iloc[0]
                        mean_value = customer_segments[col].mean()
                        diff_percent = ((value - mean_value) / mean_value) * 100
                        comparison = high_text if value > mean_value else low_text
                        impact = "tăng nguy cơ churn" if (col == 'Returns' and value > mean_value) or (col != 'Returns' and value < mean_value) else "giảm nguy cơ churn"
                        st.write(f"- {name}: {value:,.0f} ({comparison} trung bình {mean_value:,.0f} khoảng {abs(diff_percent):.1f}%), {impact}")
                else:
                    prediction_text = "có nguy cơ rời bỏ" if churn_pred else "không rời bỏ"
                    st.success(f"Khách hàng {new_customer_id} - {new_customer_name} {prediction_text}", icon="✅")

            st.subheader("Dự đoán Khách hàng Rời bỏ")

            col1, col2 = st.columns([3, 1], vertical_alignment="center")
            with col1:
                customer_id = st.number_input("Nhập Customer ID:", min_value=1, step=1, format="%d", key="customer_id_input")
            with col2:
                predict_button = st.button("Dự đoán", key="predict_button", use_container_width=True)

            if predict_button:
                customer_data = customer_segments[customer_segments['Customer ID'] == customer_id]
                if not customer_data.empty:
                    X = scaler.transform(customer_data[['Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']])
                    churn_pred = churn_model.predict(X)[0]
                    customer_name = customer_data['Customer Name'].iloc[0] if 'Customer Name' in customer_data.columns else 'Unknown'
            
                    # Hiển thị toàn bộ thông tin khách hàng
                    st.markdown("### Thông tin chi tiết của khách hàng")
                    st.dataframe(customer_data.style.format({
                        'Total Purchase Amount': '{:,.0f}',
                        'Transaction Count': '{:.0f}',
                        'Returns': '{:.0f}',
                        'Age': '{:.0f}'
                    }), use_container_width=True)

                    # Tính toán và giải thích xác suất
                    if hasattr(churn_model, 'predict_proba'):
                        churn_prob = churn_model.predict_proba(X)[0][1] * 100
                        prediction_text = ("có nguy cơ rời bỏ cao" if churn_prob >= 75 else 
                                 "có nguy cơ rời bỏ" if churn_prob >= 50 else 
                                 "không có nguy cơ rời bỏ")
                        st.success(f"Khách hàng {customer_id} - {customer_name} "
                          f"{prediction_text} (Xác suất: {churn_prob:.2f}%)", icon="✅")
                
                        # Giải thích chi tiết xác suất
                        st.markdown("### Giải thích dự đoán")
                        st.write(f"Xác suất churn {churn_prob:.2f}% được tính dựa trên các yếu tố:")
                
                        # So sánh với giá trị trung bình
                        factors = [
                            ('Total Purchase Amount', 'Tổng chi tiêu', 'thấp hơn', 'cao hơn'),
                            ('Transaction Count', 'Số giao dịch', 'ít hơn', 'nhiều hơn'),
                            ('Returns', 'Số lần hoàn trả', 'cao hơn', 'thấp hơn'),
                            ('Age', 'Độ tuổi', 'trẻ hơn', 'lớn hơn')
                        ]
                
                        explanations = []
                        for col, name, low_text, high_text in factors:
                            value = customer_data[col].iloc[0]
                            mean_value = customer_segments[col].mean()
                            diff_percent = ((value - mean_value) / mean_value) * 100
                    
                            if col == 'Returns':
                                impact = "tăng nguy cơ churn" if value > mean_value else "giảm nguy cơ churn"
                                comparison = high_text if value > mean_value else low_text
                            else:
                                impact = "giảm nguy cơ churn" if value > mean_value else "tăng nguy cơ churn"
                                comparison = high_text if value > mean_value else low_text
                        
                            explanations.append(
                                f"- {name}: {value:,.0f} ({comparison} trung bình {mean_value:,.0f} khoảng {abs(diff_percent):.1f}%), "
                                f"{impact}"
                            )
                
                        for exp in explanations:
                            st.write(exp)
                
                        # Ngưỡng phân loại
                        st.write("\n**Cách phân loại nguy cơ:**")
                        st.write("- ≥ 75%: Nguy cơ rời bỏ cao")
                        st.write("- 50-74%: Có nguy cơ rời bỏ")
                        st.write("- < 50%: Không có nguy cơ rời bỏ")
            
                    else:
                        prediction_text = "có nguy cơ rời bỏ" if churn_pred else "không rời bỏ"
                        st.success(f"Khách hàng {customer_id} - {customer_name} {prediction_text}", icon="✅")

                    # Phân tích nguyên nhân tiềm năng
                    if churn_pred or (hasattr(churn_model, 'predict_proba') and churn_prob >= 50):
                        st.markdown("### Nguyên nhân tiềm năng")
                        reasons = []
                        if customer_data['Transaction Count'].iloc[0] < customer_segments['Transaction Count'].mean():
                            reasons.append("- Tần suất giao dịch thấp hơn trung bình")
                        if customer_data['Returns'].iloc[0] > customer_segments['Returns'].mean():
                            reasons.append("- Tỷ lệ hoàn trả cao hơn trung bình")
                        if customer_data['Total Purchase Amount'].iloc[0] < customer_segments['Total Purchase Amount'].mean():
                            reasons.append("- Chi tiêu thấp hơn trung bình")
                
                        if reasons:
                            for reason in reasons:
                                st.write(reason)
                        else:
                            st.write("- Không có nguyên nhân cụ thể được xác định")

                    # Gợi ý hành động
                    if 'filtered_df' in globals() and not filtered_df.empty:
                        customer_filtered = filtered_df[filtered_df['Customer ID'] == customer_id]
                        if not customer_filtered.empty:
                            last_purchase = customer_filtered['Purchase Date'].max()
                            fav_category = customer_filtered['Product Category'].mode()[0] if not customer_filtered['Product Category'].mode().empty else 'Unknown'
                            days_inactive = (pd.Timestamp.now() - last_purchase).days
                            avg_spending = customer_data['Total Purchase Amount'].mean()
                            potential_loss = avg_spending * 12

                            st.markdown("### Phân tích bổ sung và Gợi ý")
                            st.write(f"**Doanh thu tiềm năng bị mất**: {potential_loss:,.0f} $ (ước tính 12 tháng)")
                            st.write(f"**Thông tin hành vi:**")
                            st.write(f"- Lần mua cuối: {days_inactive} ngày trước")
                            st.write(f"- Danh mục yêu thích: {fav_category}")
                            st.write("**Gợi ý hành động:**")
                            if days_inactive > 30:
                                st.write(f"- Gửi email ưu đãi 20% cho {fav_category} để tái kích hoạt")
                            else:
                                st.write(f"- Tặng mã giảm giá 10% cho {fav_category} để khuyến khích mua sắm")
                    else:
                        st.warning("Dữ liệu filtered_df không khả dụng để tính toán chi tiết.")
                else:
                    st.error(f"Không tìm thấy khách hàng {customer_id}!", icon="❌")

            st.markdown("---")
            st.write("**Top 10 Khách hàng có nguy cơ rời bỏ cao nhất**")
            # Chỉ tính Churn Probability một lần và lưu vào customer_segments nếu cần
            if 'Churn Probability' not in customer_segments.columns and hasattr(churn_model, 'predict_proba'):
                X_all = scaler.transform(customer_segments[['Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']])
                churn_probs = churn_model.predict_proba(X_all)[:, 1]
                customer_segments['Churn Probability'] = churn_probs * 100
            elif 'Churn Probability' not in customer_segments.columns:
                X_all = scaler.transform(customer_segments[['Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']])
                churn_preds = churn_model.predict(X_all)
                customer_segments['Churn Prediction'] = churn_preds

            if 'Churn Probability' in customer_segments.columns:
                top_churn = customer_segments.sort_values('Churn Probability', ascending=False).head(10)
                st.dataframe(top_churn[['Customer ID', 'Customer Name', 'Total Purchase Amount', 'Transaction Count', 'Returns', 'Age', 'Churn Probability']]
                     .style.format({'Churn Probability': '{:.2f}%', 'Total Purchase Amount': '{:,.0f}'}), height=300)
            else:
                top_churn = customer_segments[customer_segments['Churn Prediction'] == 1].head(10)
                st.dataframe(top_churn[['Customer ID', 'Customer Name', 'Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']], height=300)

            st.markdown("---")
            st.write("**Xu hướng Nguy cơ Churn Theo Thời gian**")
            if 'filtered_df' in globals() and not filtered_df.empty:
                df_with_churn = filtered_df.merge(customer_segments[['Customer ID', 'Churn Probability']], on='Customer ID', how='left')
                if not df_with_churn.empty and 'Churn Probability' in df_with_churn.columns:
                    churn_trend = df_with_churn.groupby(df_with_churn['Purchase Date'].dt.to_period('M'))['Churn Probability'].mean().reset_index()
                    churn_trend['Purchase Date'] = churn_trend['Purchase Date'].astype(str)
                    fig_churn_trend = px.line(churn_trend, x='Purchase Date', y='Churn Probability',
                                    title="Nguy cơ Churn Trung bình Theo Tháng", height=400, line_shape='spline')
                    st.plotly_chart(fig_churn_trend, use_container_width=True, key="chart_churn_trend")
                else:
                    st.warning("Không có dữ liệu Churn Probability để hiển thị xu hướng.")
            else:
                st.warning("Dữ liệu filtered_df không khả dụng để hiển thị xu hướng.")

            st.markdown("---")
            st.write("**Nguy cơ Churn Theo Phân khúc Khách hàng**")
            if 'Cluster' in customer_segments.columns and 'Churn Probability' in customer_segments.columns:
                churn_by_cluster = customer_segments.groupby('Cluster')['Churn Probability'].mean().reset_index()
                fig_churn_cluster = px.bar(churn_by_cluster, x='Cluster', y='Churn Probability',
                                 title="Nguy cơ Churn Trung bình Theo Nhóm", color='Cluster',
                                 text=churn_by_cluster['Churn Probability'].apply(lambda x: f"{x:.2f}%"), height=400)
                fig_churn_cluster.update_traces(textposition='outside')
                st.plotly_chart(fig_churn_cluster, use_container_width=True, key="chart_churn_by_cluster")
            else:
                st.warning("Không có dữ liệu Cluster hoặc Churn Probability để hiển thị phân tích.")

        # Tab 4: Xu hướng Thời gian
        with tabs[3]:
            st.subheader("Xu hướng Theo Thời gian")

            if 'Purchase Date' in filtered_df.columns and filtered_df['Purchase Date'].dt.hour.notnull().any():
                hourly_trends = filtered_df.groupby(filtered_df['Purchase Date'].dt.hour)['Total Purchase Amount'].sum().reset_index()
                hourly_trends.columns = ['Hour', 'Total Purchase Amount']
                fig_hourly = px.bar(hourly_trends, x='Hour', y='Total Purchase Amount', 
                                title="Doanh thu Theo Giờ trong Ngày", 
                                text=hourly_trends['Total Purchase Amount'].apply(lambda x: f"{x:,.0f}"), 
                                height=400)
                fig_hourly.update_traces(textposition='outside')
                st.plotly_chart(fig_hourly, use_container_width=True, key="chart_hourly_trends")
            else:
                st.warning("Dữ liệu không chứa thông tin giờ chi tiết để phân tích theo giờ.")

            monthly_revenue = filtered_df.groupby(filtered_df['Purchase Date'].dt.to_period('M'))['Total Purchase Amount'].sum().reset_index()
            monthly_revenue['Month_Num'] = np.arange(len(monthly_revenue))
            monthly_revenue['Purchase Date'] = monthly_revenue['Purchase Date'].astype(str)
            fig5 = px.line(monthly_revenue, x='Purchase Date', y='Total Purchase Amount', 
                        title="Doanh thu Theo Tháng", height=400, line_shape='spline')
            st.plotly_chart(fig5, use_container_width=True, key="chart_monthly_revenue")

            quarterly_trends = filtered_df.groupby(filtered_df['Purchase Date'].dt.to_period('Q'))['Total Purchase Amount'].sum().reset_index()
            quarterly_trends['Purchase Date'] = quarterly_trends['Purchase Date'].astype(str)
            fig_quarterly = px.bar(quarterly_trends, x='Purchase Date', y='Total Purchase Amount', 
                                title="Doanh thu Theo Quý", 
                                text=quarterly_trends['Total Purchase Amount'].apply(lambda x: f"{x:,.0f}"), 
                                height=400)
            fig_quarterly.update_traces(textposition='outside')
            st.plotly_chart(fig_quarterly, use_container_width=True, key="chart_quarterly_trends")

        # Tab 5: Chi tiết Khách hàng
        with tabs[4]:
            st.subheader("Chi tiết Khách hàng")
            customer_id = st.number_input("Nhập Customer ID để xem chi tiết:", min_value=1, step=1)
            customer_data = filtered_df[filtered_df['Customer ID'] == customer_id]
            if not customer_data.empty:
                st.write(f"**Tên khách hàng**: {customer_data['Customer Name'].iloc[0]}")
                st.write(f"**Giới tính**: {customer_data['Gender'].iloc[0]}")
                st.write(f"**Tổng chi tiêu**: {customer_data['Total Purchase Amount'].sum():,.0f} $")
                st.dataframe(customer_data[['Purchase Date', 'Product Category', 'Product Price', 'Quantity', 
                                   'Total Purchase Amount', 'Payment Method', 'Returns']])
                fig = px.line(customer_data, x='Purchase Date', y='Total Purchase Amount', 
                      title=f"Lịch sử mua sắm của {customer_data['Customer Name'].iloc[0]} (ID: {customer_id})", height=400)
                st.plotly_chart(fig, use_container_width=True, key="chart_customer_history")
            else:
                st.warning("Không tìm thấy khách hàng này!")

        # Tab 6: Phân tích Hoàn trả
        with tabs[5]:
            st.subheader("Phân tích Hoàn trả")
            return_rate = filtered_df.groupby('Product Category')['Returns'].mean().reset_index()
            fig6 = px.bar(return_rate, x='Product Category', y='Returns', 
                        title="Tỷ lệ Hoàn trả theo Danh mục", text_auto='.2%', height=400)
            fig6.update_traces(textposition='outside')
            st.plotly_chart(fig6, use_container_width=True, key="chart_return_rate_1")

            # Xóa hoặc sửa biểu đồ trùng lặp thứ hai
            return_vs_revenue = filtered_df.groupby('Product Category').agg({'Returns': 'mean', 'Total Purchase Amount': 'sum'}).reset_index()
            return_vs_revenue['Returns'] = return_vs_revenue['Returns'] * 100
            fig_compare = px.scatter(return_vs_revenue, x='Total Purchase Amount', y='Returns', 
                                 color='Product Category', size='Total Purchase Amount',
                                 title="Tỷ lệ Hoàn trả so với Doanh thu",
                                 labels={'Total Purchase Amount': 'Doanh thu ($)', 'Returns': 'Tỷ lệ Hoàn trả (%)'},
                                 height=400)
            st.plotly_chart(fig_compare, use_container_width=True, key="chart_return_vs_revenue")  # Sửa fig6 thành fig_compare
            st.write("**Gợi ý**: Danh mục có doanh thu cao nhưng tỷ lệ hoàn trả lớn cần cải thiện chất lượng sản phẩm.")

    def generate_pdf():
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Đăng ký font hỗ trợ tiếng Việt
        pdfmetrics.registerFont(TTFont('TimesNewRoman', 'times.ttf'))
        pdfmetrics.registerFont(TTFont('TimesNewRoman-Bold', 'timesbd.ttf'))

        # Hàm kiểm tra và tạo trang mới nếu cần
        def check_page_break(y_position, space_needed):
            if y_position - space_needed < 50:  # Ngưỡng dưới cùng của trang
                c.showPage()
                return height - 50  # Reset về đầu trang mới
            return y_position

        # Tính toán trước tất cả dữ liệu cần thiết
        total_revenue = filtered_df['Total Purchase Amount'].sum()
        total_revenue = 0 if pd.isna(total_revenue) else total_revenue
        transaction_count = len(filtered_df)
        top_category = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().idxmax() if not filtered_df.empty else "Không có dữ liệu"
        revenue_by_category = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().reset_index()
        top_spenders = filtered_df.groupby('Customer ID')['Total Purchase Amount'].sum().nlargest(5).reset_index()
        avg_spending = customer_segments.groupby('Cluster')['Total Purchase Amount'].mean().reset_index()
        return_rate = filtered_df.groupby('Product Category')['Returns'].mean().reset_index()
        low_transaction_day = filtered_df.groupby('Day of Week')['Customer ID'].count().idxmin()
        future_months = np.arange(len(monthly_revenue), len(monthly_revenue) + 3).reshape(-1, 1)
        future_pred = revenue_model.predict(future_months)

        # Tiêu đề báo cáo
        c.setFillColorRGB(0.18, 0.48, 0.81)
        c.setFont("TimesNewRoman-Bold", 16)
        c.drawString(100, 750, "Báo cáo Phân tích Hành vi Mua sắm")
        c.setFont("TimesNewRoman", 12)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(100, 730, f"Ngày cập nhật: {pd.Timestamp.now().strftime('%d/%m/%Y')}")
        c.line(100, 720, 500, 720)

        # 1. Tổng quan
        y_position = 700
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, y_position, "1. Tổng quan Dữ liệu")
        y_position -= 20
        c.setFont("TimesNewRoman", 12)
        c.drawString(100, y_position, f"Tổng doanh thu: {total_revenue:,.0f} $")
        y_position -= 20
        c.drawString(100, y_position, f"Số giao dịch: {transaction_count:,}")
        y_position -= 20
        c.drawString(100, y_position, f"Top danh mục: {top_category}")
        y_position -= 20
        top_payment = filtered_df.groupby('Payment Method')['Total Purchase Amount'].sum().idxmax()
        c.drawString(100, y_position, f"Phương thức thanh toán phổ biến: {top_payment}")
        y_position -= 20
        # 2. Phân tích Doanh thu theo Danh mục
        y_position = check_page_break(y_position, 20 + 20 * len(revenue_by_category))
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, y_position, "2. Doanh thu theo Danh mục Sản phẩm")
        y_position -= 20
        data = [["Danh mục", "Doanh thu ($)"]]
        for _, row in revenue_by_category.iterrows():
            data.append([row['Product Category'], f"{row['Total Purchase Amount']:,.0f}"])
    
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ]))
        table.wrapOn(c, width, height)
        table.drawOn(c, 100, y_position - len(data) * 20)
        y_position -= (len(data) * 20 + 20)

        # 3. Top 5 Khách hàng Chi tiêu Nhiều nhất
        y_position = check_page_break(y_position, 20 + 20 * 6)  # 6 dòng cho top 5 + header
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, y_position, "3. Top 5 Khách hàng Chi tiêu Nhiều nhất")
        y_position -= 20
        data = [["Customer ID", "Tổng Chi tiêu ($)"]]
        for _, row in top_spenders.iterrows():
            data.append([str(row['Customer ID']), f"{row['Total Purchase Amount']:,.0f}"])
    
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ]))
        table.wrapOn(c, width, height)
        table.drawOn(c, 100, y_position - len(data) * 20)
        y_position -= (len(data) * 20 + 20)

        # 4. Phân khúc Khách hàng
        y_position = check_page_break(y_position, 20 + 20 * len(avg_spending))
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, y_position, "4. Phân khúc Khách hàng")
        y_position -= 20
        data = [["Nhóm (Cluster)", "Chi tiêu Trung bình ($)"]]
        for _, row in avg_spending.iterrows():
            data.append([str(row['Cluster']), f"{row['Total Purchase Amount']:,.0f}"])
    
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ]))
        table.wrapOn(c, width, height)
        table.drawOn(c, 100, y_position - len(data) * 20)
        y_position -= (len(data) * 20 + 20)

        # 5. Tỷ lệ Hoàn trả theo Danh mục
        y_position = check_page_break(y_position, 20 + 20 * len(return_rate))
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, y_position, "5. Tỷ lệ Hoàn trả theo Danh mục")
        y_position -= 20
        data = [["Danh mục", "Tỷ lệ Hoàn trả (%)"]]
        for _, row in return_rate.iterrows():
            data.append([row['Product Category'], f"{row['Returns']:.2f}%"])
    
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ]))
        table.wrapOn(c, width, height)
        table.drawOn(c, 100, y_position - len(data) * 20)
        y_position -= (len(data) * 20 + 20)

        # 6. Gợi ý Hành động
        y_position = check_page_break(y_position, 60)
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, y_position, "6. Gợi ý Hành động")
        y_position -= 20
        c.setFont("TimesNewRoman", 12)
        c.drawString(100, y_position, f"- Tăng khuyến mãi vào {low_transaction_day} (ngày ít giao dịch nhất).")
        y_position -= 20
        c.drawString(100, y_position, f"- Tập trung quảng bá {top_category} (danh mục doanh thu cao nhất).")
        y_position -= 20

        # 7. Dự đoán Doanh thu
        y_position = check_page_break(y_position, 20 + 20 * 4)  # 4 dòng cho 3 tháng + header
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, y_position, "7. Dự đoán Doanh thu 3 Tháng Tới")
        y_position -= 20
        data = [["Tháng", "Doanh thu Dự đoán ($)"]]
        for i, pred in enumerate(future_pred):
            data.append([f"Tháng {i+1}", f"{int(pred):,.0f}"])
    
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ]))
        table.wrapOn(c, width, height)
        table.drawOn(c, 100, y_position - len(data) * 20)

        # Thêm phần phân tích theo Giới tính
        y_position = check_page_break(y_position, 20 + 20 * len(filtered_df['Gender'].unique()))
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, y_position, "8. Phân tích Theo Giới tính")
        y_position -= 20
        data = [["Giới tính", "Chi tiêu Trung bình ($)"]]
        for gender, spending in filtered_df.groupby('Gender')['Total Purchase Amount'].mean().items():
            data.append([gender, f"{spending:,.0f}"])
    
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ]))
        table.wrapOn(c, width, height)
        table.drawOn(c, 100, y_position - len(data) * 20)
        y_position -= (len(data) * 20 + 20)

        # Kết thúc và lưu PDF
        c.showPage()
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
            <div class="footer-container">
                <div class="footer-column">
                    <h4>StudySystem</h4>
                    <p>Phân tích hành vi mua sắm chuyên sâu để tối ưu hóa doanh thu và trải nghiệm khách hàng.</p>
                </div>
                <div class="footer-column">
                    <h4>Liên kết nhanh</h4>
                    <a href="https://example.com/about" target="_blank">Về chúng tôi</a>
                    <a href="https://example.com/privacy" target="_blank">Chính sách bảo mật</a>
                    <a href="https://example.com/contact" target="_blank">Liên hệ</a>
                </div>
                <div class="footer-column">
                    <h4>Liên hệ</h4>
                    <p>Email: <a href="mailto:contact@ktdl9team.com">contact@ktdl9team.com</a></p>
                    <p>Hotline: +84 123 456 789</p>
                    <div class="social-icons">
                        <a href="https://facebook.com" target="_blank"><i class="fab fa-facebook-f"></i></a>
                        <a href="https://twitter.com" target="_blank"><i class="fab fa-twitter"></i></a>
                        <a href="https://linkedin.com" target="_blank"><i class="fab fa-linkedin-in"></i></a>
                    </div>
                </div>
            </div>
            <div class="footer-bottom">
                <p>© 2025 - Phát triển bởi KTDL-9 Team. All rights reserved.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
elif st.session_state.get('authentication_status') is False:
    st.error("Tên người dùng hoặc mật khẩu không đúng!")
elif st.session_state.get('authentication_status') is None:
    st.warning("Vui lòng nhập tên người dùng và mật khẩu.")