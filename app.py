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
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from io import BytesIO

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

    # Tải dữ liệu từ Google Sheets
    @st.cache_data
    def load_data_from_sheets():
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
        client = gspread.authorize(creds)
        sheet = client.open("Purchase Data").sheet1  # Thay "Purchase Data" bằng tên Sheets của bạn nếu khác
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
        df['Total Purchase Amount'] = df['Product Price'].astype(float) * df['Quantity'].astype(float)
        df['Customer ID'] = df['Customer ID'].astype(int)
        df['Returns'] = df['Returns'].astype(float)
        df['Age'] = df['Age'].astype(int)
        df['Churn'] = df['Churn'].astype(int)
        df['Year'] = df['Year'].astype(int)
        df['Month'] = df['Month'].astype(int)
        customer_segments = pd.read_csv('customer_segments.csv')
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
    st.title("📊 Hệ thống Phân tích Hành vi Mua sắm ")
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
            revenue_by_day = filtered_df.groupby(filtered_df['Purchase Date'].dt.date)['Total Purchase Amount'].sum().reset_index()
            fig2 = px.line(revenue_by_day, x='Purchase Date', y='Total Purchase Amount', 
                       title="Doanh thu Theo Ngày", height=400, line_shape='spline')
            st.plotly_chart(fig2, use_container_width=True)
    
        with col3:
            top_spenders = filtered_df.groupby('Customer ID').agg({
                'Total Purchase Amount': 'sum',
                'Purchase Date': 'count',
                'Product Category': lambda x: x.mode()[0]
            }).nlargest(5, 'Total Purchase Amount').reset_index()
            top_spenders.columns = ['Customer ID', 'Total Purchase Amount', 'Transaction Count', 'Favorite Category']
            fig3 = px.bar(top_spenders, x='Customer ID', y='Total Purchase Amount', 
                      title="Top 5 Khách hàng Chi tiêu Cao nhất", 
                      text=top_spenders['Customer ID'].astype(str) + ' (' + top_spenders['Transaction Count'].astype(str) + ' GD)',
                      color_discrete_sequence=['#ff6f61'], height=400)
            fig3.update_traces(textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)
            st.write("**Chi tiết Top 5 Khách hàng:**")
            st.dataframe(top_spenders.style.format({
                'Total Purchase Amount': '{:,.0f} VND',
                'Transaction Count': '{:,}',
            }))
    
        # Phân tích chi tiết danh mục theo ngày
        st.subheader("Chi tiết Danh mục Theo Ngày")
        # Bộ lọc danh mục
        selected_category = st.selectbox("Chọn danh mục để xem chi tiết:", 
                                     options=['Tất cả'] + sorted(filtered_df['Product Category'].unique()),
                                     index=0)
    
        # Dữ liệu doanh thu theo ngày cho danh mục
        if selected_category == 'Tất cả':
            category_by_day = filtered_df.groupby(filtered_df['Purchase Date'].dt.date)['Total Purchase Amount'].sum().reset_index()
        else:
            category_by_day = filtered_df[filtered_df['Product Category'] == selected_category].groupby(filtered_df['Purchase Date'].dt.date)['Total Purchase Amount'].sum().reset_index()
    
        # Biểu đồ doanh thu theo ngày cho danh mục
        fig_category_day = px.line(category_by_day, x='Purchase Date', y='Total Purchase Amount', 
                               title=f"Doanh thu Theo Ngày của {'Tất cả Danh mục' if selected_category == 'Tất cả' else selected_category}", 
                               height=400, line_shape='spline')
        st.plotly_chart(fig_category_day, use_container_width=True)
    
        # Bảng dữ liệu chi tiết
        with st.expander(f"🔎 Xem dữ liệu chi tiết của {'Tất cả Danh mục' if selected_category == 'Tất cả' else selected_category}", expanded=False):
            if selected_category == 'Tất cả':
                detailed_data = filtered_df.groupby(['Purchase Date', 'Product Category'])['Total Purchase Amount'].sum().unstack().fillna(0)
            else:
                detailed_data = filtered_df[filtered_df['Product Category'] == selected_category].groupby('Purchase Date')['Total Purchase Amount'].sum().reset_index()
            st.dataframe(detailed_data.style.format('{:,.0f} VND'))
    
        st.subheader("Gợi ý Hành động")
        low_transaction_day = filtered_df.groupby('Day of Week')['Customer ID'].count().idxmin()
        low_day_revenue = filtered_df.groupby('Day of Week')['Total Purchase Amount'].sum().min()
        st.write(f"- Tăng khuyến mãi 15% vào {low_transaction_day} (doanh thu thấp nhất: {low_day_revenue:,.0f} VND) qua email hoặc SMS.")
    
        top_category = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().idxmax()
        top_category_revenue = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().max()
        st.write(f"- Đẩy mạnh quảng bá {top_category} (doanh thu: {top_category_revenue:,.0f} VND) qua mạng xã hội và banner trên website.")
    
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

        # Bảng chi tiết các nhóm
        with st.expander("🔎 Chi tiết các nhóm khách hàng", expanded=False):
            cluster_summary = customer_segments.groupby('Cluster').agg({
                'Total Purchase Amount': 'mean',
                'Transaction Count': 'mean',
                'Returns': 'mean',
                'Age': 'mean',
                'Customer ID': 'count'
            }).rename(columns={
                'Total Purchase Amount': 'Chi tiêu TB (VND)',
                'Transaction Count': 'Tần suất GD TB',
                'Returns': 'Tỷ lệ Hoàn trả TB',
                'Age': 'Độ tuổi TB',
                'Customer ID': 'Số lượng KH'
            })
            st.dataframe(cluster_summary.style.format({
                'Chi tiêu TB (VND)': '{:,.0f}',
                'Tần suất GD TB': '{:.2f}',
                'Tỷ lệ Hoàn trả TB': '{:.2%}',
                'Độ tuổi TB': '{:.1f}',
                'Số lượng KH': '{:,}'
            }).background_gradient(cmap='Blues'))

        # Biểu đồ chi tiêu trung bình
        avg_spending = customer_segments.groupby('Cluster')['Total Purchase Amount'].mean().reset_index()
        fig4 = px.bar(avg_spending, x='Cluster', y='Total Purchase Amount', 
                    title="Chi tiêu Trung bình theo Nhóm", color='Cluster', 
                     text=avg_spending['Total Purchase Amount'].round(2), height=400)
        fig4.update_traces(textposition='outside')
        st.plotly_chart(fig4, use_container_width=True)

        # So sánh doanh thu và tỷ lệ hoàn trả
        cluster_compare = customer_segments.groupby('Cluster').agg({
            'Total Purchase Amount': 'mean',
            'Returns': 'mean'
        }).reset_index()
        cluster_compare['Returns'] = cluster_compare['Returns'] * 100
        fig_compare = px.scatter(cluster_compare, x='Total Purchase Amount', y='Returns', 
                             color='Cluster', size='Total Purchase Amount',
                             title="So sánh Chi tiêu TB và Tỷ lệ Hoàn trả",
                             labels={'Total Purchase Amount': 'Chi tiêu TB (VND)', 'Returns': 'Tỷ lệ Hoàn trả (%)'},
                             height=400)
        st.plotly_chart(fig_compare, use_container_width=True)

        # Xu hướng chi tiêu theo thời gian
        df_with_clusters = filtered_df.merge(customer_segments[['Customer ID', 'Cluster']], on='Customer ID', how='left')
        cluster_trends = df_with_clusters.groupby(['Cluster', df_with_clusters['Purchase Date'].dt.to_period('M')])['Total Purchase Amount'].sum().reset_index()
        cluster_trends['Purchase Date'] = cluster_trends['Purchase Date'].astype(str)
        fig_trends = px.line(cluster_trends, x='Purchase Date', y='Total Purchase Amount', color='Cluster',
                         title="Xu hướng Chi tiêu Theo Tháng của Các Nhóm", height=400, line_shape='spline')
        st.plotly_chart(fig_trends, use_container_width=True)

        # Gợi ý hành động
        st.subheader("Gợi ý Hành động Theo Nhóm")
        for cluster in cluster_summary.index:
            spending = cluster_summary.loc[cluster, 'Chi tiêu TB (VND)']
            frequency = cluster_summary.loc[cluster, 'Tần suất GD TB']
            returns = cluster_summary.loc[cluster, 'Tỷ lệ Hoàn trả TB']
            st.write(f"**Nhóm {cluster}:**")
            if spending > cluster_summary['Chi tiêu TB (VND)'].mean() and frequency < cluster_summary['Tần suất GD TB'].mean():
                st.write(f"- Chi tiêu cao nhưng ít giao dịch: Tặng mã giảm giá định kỳ để tăng tần suất mua sắm.")
            elif returns > cluster_summary['Tỷ lệ Hoàn trả TB'].mean():
                st.write(f"- Tỷ lệ hoàn trả cao: Cải thiện chất lượng sản phẩm hoặc kiểm tra chính sách đổi trả.")
            else:
                st.write(f"- Nhóm ổn định: Duy trì chính sách hiện tại hoặc thử nghiệm ưu đãi nhỏ.")

        # Tính tương tác
        selected_cluster = st.selectbox("Chọn nhóm để xem chi tiết:", options=cluster_summary.index)
        cluster_data = customer_segments[customer_segments['Cluster'] == selected_cluster]
        st.write(f"**Thông tin chi tiết Nhóm {selected_cluster}:**")
        st.dataframe(cluster_data[['Customer ID', 'Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']])
        cluster_purchases = filtered_df[filtered_df['Customer ID'].isin(cluster_data['Customer ID'])]
        fav_categories = cluster_purchases.groupby('Product Category')['Total Purchase Amount'].sum().reset_index()
        fig_fav = px.pie(fav_categories, values='Total Purchase Amount', names='Product Category',
                     title=f"Danh mục Yêu thích của Nhóm {selected_cluster}", height=400)
        st.plotly_chart(fig_fav, use_container_width=True)

    # Tab 3: Dự đoán Churn
    with tabs[2]:
        st.subheader("Dự đoán Khách hàng Rời bỏ")

        # Dự đoán cho một khách hàng cụ thể
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
                if hasattr(churn_model, 'predict_proba'):
                    churn_prob = churn_model.predict_proba(X)[0][1] * 100  # Xác suất churn (%)
                    st.success(f"Khách hàng {customer_id} {'có nguy cơ rời bỏ' if churn_pred else 'không rời bỏ'} (Xác suất: {churn_prob:.2f}%)", icon="✅")
                else:
                    st.success(f"Khách hàng {customer_id} {'có nguy cơ rời bỏ' if churn_pred else 'không rời bỏ'}", icon="✅")
                if churn_pred:
                    st.write("**Gợi ý**: Gửi ưu đãi giảm giá hoặc email cá nhân hóa để giữ chân khách hàng này.")
            else:
                st.error(f"Không tìm thấy khách hàng {customer_id}!", icon="❌")

        # Top 10 khách hàng có nguy cơ churn cao
        st.markdown("---")
        st.write("**Top 10 Khách hàng có nguy cơ rời bỏ cao nhất**")
        X_all = scaler.transform(customer_segments[['Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']])
        if hasattr(churn_model, 'predict_proba'):
            churn_probs = churn_model.predict_proba(X_all)[:, 1]  # Lấy xác suất churn
            customer_segments['Churn Probability'] = churn_probs * 100
            top_churn = customer_segments.sort_values('Churn Probability', ascending=False).head(10)
            st.dataframe(top_churn[['Customer ID', 'Total Purchase Amount', 'Transaction Count', 'Returns', 'Age', 'Churn Probability']]
                     .style.format({'Churn Probability': '{:.2f}%', 'Total Purchase Amount': '{:,.0f}'}))
        else:
            churn_preds = churn_model.predict(X_all)
            customer_segments['Churn Prediction'] = churn_preds
            top_churn = customer_segments[customer_segments['Churn Prediction'] == 1].head(10)
            st.dataframe(top_churn[['Customer ID', 'Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']])

    # Tab 4: Xu hướng Thời gian
    with tabs[3]:
        st.subheader("Xu hướng Theo Thời gian")

        # Phân tích theo giờ (giả sử Purchase Date có định dạng datetime đầy đủ)
        if 'Purchase Date' in filtered_df.columns and filtered_df['Purchase Date'].dt.hour.notnull().any():
            hourly_trends = filtered_df.groupby(filtered_df['Purchase Date'].dt.hour)['Total Purchase Amount'].sum().reset_index()
            hourly_trends.columns = ['Hour', 'Total Purchase Amount']
            fig_hourly = px.bar(hourly_trends, x='Hour', y='Total Purchase Amount', 
                            title="Doanh thu Theo Giờ trong Ngày", 
                            text=hourly_trends['Total Purchase Amount'].apply(lambda x: f"{x:,.0f}"), 
                            height=400)
            fig_hourly.update_traces(textposition='outside')
            st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            st.warning("Dữ liệu không chứa thông tin giờ chi tiết để phân tích theo giờ.")

        # Phân tích hiện tại (doanh thu theo tháng)
        monthly_revenue = filtered_df.groupby(filtered_df['Purchase Date'].dt.to_period('M'))['Total Purchase Amount'].sum().reset_index()
        monthly_revenue['Month_Num'] = np.arange(len(monthly_revenue))
        monthly_revenue['Purchase Date'] = monthly_revenue['Purchase Date'].astype(str)
        fig5 = px.line(monthly_revenue, x='Purchase Date', y='Total Purchase Amount', 
                   title="Doanh thu Theo Tháng", height=400, line_shape='spline')
        st.plotly_chart(fig5, use_container_width=True)

        # Phân tích theo quý
        quarterly_trends = filtered_df.groupby(filtered_df['Purchase Date'].dt.to_period('Q'))['Total Purchase Amount'].sum().reset_index()
        quarterly_trends['Purchase Date'] = quarterly_trends['Purchase Date'].astype(str)
        fig_quarterly = px.bar(quarterly_trends, x='Purchase Date', y='Total Purchase Amount', 
                           title="Doanh thu Theo Quý", 
                           text=quarterly_trends['Total Purchase Amount'].apply(lambda x: f"{x:,.0f}"), 
                           height=400)
        fig_quarterly.update_traces(textposition='outside')
        st.plotly_chart(fig_quarterly, use_container_width=True)

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
        # Biểu đồ tỷ lệ hoàn trả hiện tại
        return_rate = filtered_df.groupby('Product Category')['Returns'].mean().reset_index()
        fig6 = px.bar(return_rate, x='Product Category', y='Returns', 
                  title="Tỷ lệ Hoàn trả theo Danh mục", text_auto='.2%', height=400)
        fig6.update_traces(textposition='outside')
        st.plotly_chart(fig6, use_container_width=True)

        # Thêm biểu đồ so sánh tỷ lệ hoàn trả và doanh thu
        return_vs_revenue = filtered_df.groupby('Product Category').agg({'Returns': 'mean', 'Total Purchase Amount': 'sum'}).reset_index()
        return_vs_revenue['Returns'] = return_vs_revenue['Returns'] * 100  # Chuyển sang phần trăm
        fig_compare = px.scatter(return_vs_revenue, x='Total Purchase Amount', y='Returns', 
                             color='Product Category', size='Total Purchase Amount',
                             title="Tỷ lệ Hoàn trả so với Doanh thu",
                             labels={'Total Purchase Amount': 'Doanh thu (VND)', 'Returns': 'Tỷ lệ Hoàn trả (%)'},
                             height=400)
        st.plotly_chart(fig_compare, use_container_width=True)
        st.write("**Gợi ý**: Danh mục có doanh thu cao nhưng tỷ lệ hoàn trả lớn cần cải thiện chất lượng sản phẩm.")

    def generate_pdf():
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Đăng ký font hỗ trợ tiếng Việt
        pdfmetrics.registerFont(TTFont('TimesNewRoman', 'times.ttf'))
        pdfmetrics.registerFont(TTFont('TimesNewRoman-Bold', 'timesbd.ttf'))

        # Tiêu đề báo cáo
        c.setFillColorRGB(0.18, 0.48, 0.81)
        c.setFont("TimesNewRoman-Bold", 16)
        c.drawString(100, 750, "Báo cáo Phân tích Hành tươi Mua sắm")
        c.setFont("TimesNewRoman", 12)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(100, 730, f"Ngày cập nhật: {pd.Timestamp.now().strftime('%d/%m/%Y')}")
        c.line(100, 720, 500, 720)

        # 1. Tổng quan
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, 700, "1. Tổng quan Dữ liệu")
        c.setFont("TimesNewRoman", 12)
        y_position = 680
        total_revenue = filtered_df['Total Purchase Amount'].sum()
        total_revenue = 0 if pd.isna(total_revenue) else total_revenue
        transaction_count = len(filtered_df)
        c.drawString(100, y_position, f"Tổng doanh thu: {total_revenue:,.0f} VND")
        y_position -= 20
        c.drawString(100, y_position, f"Số giao dịch: {transaction_count:,}")
        y_position -= 20
        top_category = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().idxmax() if not filtered_df.empty else "Không có dữ liệu"
        c.drawString(100, y_position, f"Top danh mục: {top_category}")
        y_position -= 20

        # 2. Phân tích Doanh thu theo Danh mục
        c.setFont("TimesNewRoman-Bold", 14)
        y_position -= 20
        c.drawString(100, y_position, "2. Doanh thu theo Danh mục Sản phẩm")
        y_position -= 20
        revenue_by_category = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().reset_index()
        data = [["Danh mục", "Doanh thu (VND)"]]
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
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, y_position, "3. Top 5 Khách hàng Chi tiêu Nhiều nhất")
        y_position -= 20
        top_spenders = filtered_df.groupby('Customer ID')['Total Purchase Amount'].sum().nlargest(5).reset_index()
        data = [["Customer ID", "Tổng Chi tiêu (VND)"]]
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
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, y_position, "4. Phân khúc Khách hàng")
        y_position -= 20
        avg_spending = customer_segments.groupby('Cluster')['Total Purchase Amount'].mean().reset_index()
        data = [["Nhóm (Cluster)", "Chi tiêu Trung bình (VND)"]]
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
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, y_position, "5. Tỷ lệ Hoàn trả theo Danh mục")
        y_position -= 20
        return_rate = filtered_df.groupby('Product Category')['Returns'].mean().reset_index()
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
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, y_position, "6. Gợi ý Hành động")
        y_position -= 20
        c.setFont("TimesNewRoman", 12)
        low_transaction_day = filtered_df.groupby('Day of Week')['Customer ID'].count().idxmin()
        c.drawString(100, y_position, f"- Tăng khuyến mãi vào {low_transaction_day} (ngày ít giao dịch nhất).")
        y_position -= 20
        top_category = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().idxmax()
        c.drawString(100, y_position, f"- Tập trung quảng bá {top_category} (danh mục doanh thu cao nhất).")
        y_position -= 20

        # 7. Dự đoán Doanh thu
        c.setFont("TimesNewRoman-Bold", 14)
        c.drawString(100, y_position, "7. Dự đoán Doanh thu 3 Tháng Tới")
        y_position -= 20
        future_months = np.arange(len(monthly_revenue), len(monthly_revenue) + 3).reshape(-1, 1)
        future_pred = revenue_model.predict(future_months)
        data = [["Tháng", "Doanh thu Dự đoán (VND)"]]
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