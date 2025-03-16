import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# Đọc và làm sạch dữ liệu
df = pd.read_csv('customer_data.csv', parse_dates=['Purchase Date'], dtype={
    'Customer ID': 'int32', 'Product Price': 'int32', 'Quantity': 'int32',
    'Total Purchase Amount': 'int32', 'Age': 'int32', 'Churn': 'int8',
    'Year': 'int16', 'Month': 'int8'
})
df['Returns'] = df['Returns'].fillna(0).astype('float32')
df['Total Purchase Amount'] = df['Product Price'] * df['Quantity']

# Kiểm tra và xử lý lỗi logic
df = df[(df['Product Price'] > 0) & (df['Quantity'] > 0)]  # Loại bỏ giá trị không hợp lệ
df = df.drop_duplicates(subset=['Customer ID', 'Purchase Date', 'Product Category'])  # Xóa trùng
df['Gender'] = df['Gender'].str.lower().str.strip()
df['Payment Method'] = df['Payment Method'].str.title().str.strip()

# Phân khúc khách hàng
customer_features = df.groupby('Customer ID').agg({
    'Total Purchase Amount': 'sum', 'Purchase Date': 'count', 'Returns': 'mean', 'Age': 'mean'
}).rename(columns={'Purchase Date': 'Transaction Count'})
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features)
kmeans = KMeans(n_clusters=3, random_state=42)
customer_features['Cluster'] = kmeans.fit_predict(scaled_features)
customer_features.to_csv('customer_segments.csv', index=True)

# Dự đoán churn
features = customer_features.reset_index().merge(df[['Customer ID', 'Churn']].drop_duplicates(), on='Customer ID')
X = features[['Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']]
y = features['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
churn_model = LogisticRegression()
churn_model.fit(X_train, y_train)
y_pred = churn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác dự đoán churn: {accuracy:.2f}")
joblib.dump(churn_model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Dự đoán doanh thu
monthly_revenue = df.groupby(df['Purchase Date'].dt.to_period('M'))['Total Purchase Amount'].sum().reset_index()
monthly_revenue['Month_Num'] = np.arange(len(monthly_revenue))
X_rev = monthly_revenue[['Month_Num']]
y_rev = monthly_revenue['Total Purchase Amount']
revenue_model = LinearRegression()
revenue_model.fit(X_rev, y_rev)
joblib.dump(revenue_model, 'revenue_model.pkl')
print(f"Đã huấn luyện mô hình dự đoán doanh thu và lưu vào 'revenue_model.pkl'")

# Vẽ biểu đồ phân tích cơ bản
# Doanh thu theo danh mục
category_revenue = df.groupby('Product Category')['Total Purchase Amount'].sum()
plt.figure(figsize=(8, 5))
category_revenue.plot(kind='bar', color='skyblue', title='Doanh thu theo danh mục sản phẩm')
plt.xlabel('Danh mục sản phẩm')
plt.ylabel('Tổng doanh thu')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('category_revenue.png')

# Giao dịch theo ngày
purchases_by_day = df.groupby('Day of Week').size()
plt.figure(figsize=(8, 5))
purchases_by_day.plot(kind='bar', color='lightgreen', title='Số lượng giao dịch theo ngày trong tuần')
plt.xlabel('Ngày trong tuần')
plt.ylabel('Số giao dịch')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('purchases_by_day.png')

# Top 5 khách hàng
top_spenders = df.groupby('Customer ID')['Total Purchase Amount'].sum().nlargest(5)
plt.figure(figsize=(8, 5))
top_spenders.plot(kind='bar', color='salmon', title='Top 5 khách hàng chi tiêu nhiều nhất')
plt.xlabel('Customer ID')
plt.ylabel('Tổng chi tiêu')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('top_spenders.png')

# Lưu dữ liệu đã xử lý
df.to_csv('cleaned_customer_data.csv', index=False)
print("Đã xử lý dữ liệu và lưu các file cần thiết.")