import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression

# Đọc và làm sạch dữ liệu
df = pd.read_csv('cleaned_customer_data.csv', parse_dates=['Purchase Date'], dtype={
    'Customer ID': 'int32', 'Product Price': 'int32', 'Quantity': 'int32',
    'Total Purchase Amount': 'int32', 'Age': 'int32',
    'Year': 'int16', 'Month': 'int8'
})

# Xử lý giá trị thiếu và chuẩn hóa dữ liệu
df['Returns'] = df['Returns'].fillna(0).astype('float32')
df['Total Purchase Amount'] = df['Product Price'] * df['Quantity']

# Kiểm tra và xử lý lỗi logic
df = df[(df['Product Price'] > 0) & (df['Quantity'] > 0) & (df['Total Purchase Amount'] > 0)]
df = df.drop_duplicates(subset=['Customer ID', 'Purchase Date', 'Product Category'])
df['Gender'] = df['Gender'].str.lower().str.strip()
df['Payment Method'] = df['Payment Method'].str.title().str.strip()

# Chuẩn bị dữ liệu huấn luyện cho revenue_model
monthly_revenue = df.groupby(df['Purchase Date'].dt.to_period('M'))['Total Purchase Amount'].sum().reset_index()
monthly_revenue['Month_Num'] = np.arange(len(monthly_revenue))

# Chuẩn hóa dữ liệu cho revenue_model
revenue_scaler = StandardScaler()
scaled_month_num = revenue_scaler.fit_transform(monthly_revenue[['Month_Num']])

# Huấn luyện mô hình revenue_model
revenue_model = RandomForestRegressor(random_state=42)
revenue_model.fit(scaled_month_num, monthly_revenue['Total Purchase Amount'])

# Lưu mô hình và scaler cho revenue_model
joblib.dump(revenue_model, 'revenue_model.pkl')
joblib.dump(revenue_scaler, 'revenue_scaler.pkl')
print("Đã lưu revenue_model tại 'revenue_model.pkl' và revenue_scaler tại 'revenue_scaler.pkl'")

# Phân khúc khách hàng
customer_features = df.groupby('Customer ID').agg({
    'Customer Name': 'first',  # Lấy tên khách hàng
    'Gender': 'first',  # Lấy giới tính
    'Total Purchase Amount': 'sum',
    'Purchase Date': 'count',  # Số giao dịch
    'Returns': 'mean',
    'Age': 'mean'
}).rename(columns={'Purchase Date': 'Transaction Count'})

# Chuẩn hóa dữ liệu và phân cụm
churn_scaler = StandardScaler()
scaled_features = churn_scaler.fit_transform(customer_features[['Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']])
kmeans = KMeans(n_clusters=3, random_state=42)
customer_features['Cluster'] = kmeans.fit_predict(scaled_features)

# Lưu file customer_segments.csv
customer_features.to_csv('customer_segments.csv', index=True)
print("Đã lưu file customer_segments.csv với các cột: Customer ID, Customer Name, Gender, Total Purchase Amount, Transaction Count, Returns, Age, Cluster")

# ### Trực quan hóa và phân tích bổ sung

# 1. Trực quan hóa phân khúc khách hàng (Customer Segmentation)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_features, x='Total Purchase Amount', y='Transaction Count', hue='Cluster', size='Returns', palette='Set2')
plt.title('Phân khúc khách hàng dựa trên Tổng chi tiêu và Số giao dịch')
plt.xlabel('Tổng chi tiêu')
plt.ylabel('Số giao dịch')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('customer_segmentation.png')
print("Đã lưu biểu đồ phân khúc khách hàng tại 'customer_segmentation.png'")

# 2. Phân tích doanh thu theo độ tuổi và giới tính
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Gender', y='Total Purchase Amount', hue='Gender', palette='Set3')
plt.title('Phân bố doanh thu theo Giới tính')
plt.xlabel('Giới tính')
plt.ylabel('Tổng chi tiêu')
plt.tight_layout()
plt.savefig('revenue_by_gender.png')
print("Đã lưu biểu đồ doanh thu theo giới tính tại 'revenue_by_gender.png'")

# Phân tích theo độ tuổi
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Total Purchase Amount', hue='Gender', size='Quantity', palette='Set1')
plt.title('Doanh thu theo Độ tuổi và Giới tính')
plt.xlabel('Độ tuổi')
plt.ylabel('Tổng chi tiêu')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('revenue_by_age_gender.png')
print("Đã lưu biểu đồ doanh thu theo độ tuổi và giới tính tại 'revenue_by_age_gender.png'")

# 3. Phân tích phương thức thanh toán phổ biến
payment_methods = df['Payment Method'].value_counts()
plt.figure(figsize=(8, 5))
payment_methods.plot(kind='bar', color='teal', alpha=0.7)
plt.title('Phân bố phương thức thanh toán')
plt.xlabel('Phương thức thanh toán')
plt.ylabel('Số lượng giao dịch')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('payment_methods.png')
print("Đã lưu biểu đồ phân bố phương thức thanh toán tại 'payment_methods.png'")

# 4. Doanh thu theo danh mục
category_revenue = df.groupby('Product Category')['Total Purchase Amount'].sum()
plt.figure(figsize=(8, 5))
category_revenue.plot(kind='bar', color='skyblue', title='Doanh thu theo danh mục sản phẩm')
plt.xlabel('Danh mục sản phẩm')
plt.ylabel('Tổng doanh thu')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('category_revenue.png')
print("Đã lưu biểu đồ doanh thu theo danh mục tại 'category_revenue.png'")

# 5. Giao dịch theo ngày
purchases_by_day = df.groupby('Day of Week').size()
plt.figure(figsize=(8, 5))
purchases_by_day.plot(kind='bar', color='lightgreen', title='Số lượng giao dịch theo ngày trong tuần')
plt.xlabel('Ngày trong tuần')
plt.ylabel('Số giao dịch')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('purchases_by_day.png')
print("Đã lưu biểu đồ giao dịch theo ngày tại 'purchases_by_day.png'")

# 6. Top 5 khách hàng
top_spenders = df.groupby('Customer ID')['Total Purchase Amount'].sum().nlargest(5)
plt.figure(figsize=(8, 5))
top_spenders.plot(kind='bar', color='salmon', title='Top 5 khách hàng chi tiêu nhiều nhất')
plt.xlabel('Customer ID')
plt.ylabel('Tổng chi tiêu')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('top_spenders.png')
print("Đã lưu biểu đồ top 5 khách hàng tại 'top_spenders.png'")

# Lưu dữ liệu đã xử lý
df.to_csv('cleaned_customer_data.csv', index=False)
print("Đã hoàn thành phân tích, trực quan hóa và lưu các file cần thiết.")

# ### Tối ưu hóa dự đoán churn
# Chuẩn bị dữ liệu
features = customer_features.reset_index().merge(df[['Customer ID', 'Churn']].drop_duplicates(), on='Customer ID')
X = features[['Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']]
y = features['Churn']

# Cân bằng dữ liệu với SMOTE và giữ tên cột
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Tối ưu LogisticRegression
param_grid_lr = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='f1')
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_

# Tối ưu RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='f1')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

# Đánh giá và chọn mô hình
y_pred_lr = best_lr.predict(X_test)
y_pred_rf = best_rf.predict(X_test)
if classification_report(y_test, y_pred_lr, output_dict=True)['1']['f1-score'] > classification_report(y_test, y_pred_rf, output_dict=True)['1']['f1-score']:
    churn_model = best_lr
    print("Chọn LogisticRegression làm mô hình dự đoán churn.")
else:
    churn_model = best_rf
    print("Chọn RandomForest làm mô hình dự đoán churn.")

# Lưu mô hình và scaler
joblib.dump(churn_model, 'churn_model.pkl')
joblib.dump(churn_scaler, 'scaler.pkl')