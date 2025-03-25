import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE  # Thư viện xử lý mất cân bằng dữ liệu

# Đọc và làm sạch dữ liệu
df = pd.read_csv('cleaned_customer_data.csv', parse_dates=['Purchase Date'], dtype={
    'Customer ID': 'int32', 'Product Price': 'int32', 'Quantity': 'int32',
    'Total Purchase Amount': 'int32', 'Age': 'int32', 'Churn': 'int8',
    'Year': 'int16', 'Month': 'int8'
})
df['Returns'] = df['Returns'].fillna(0).astype('float32')
df['Total Purchase Amount'] = df['Product Price'] * df['Quantity']

# Kiểm tra và xử lý lỗi logic
df = df[(df['Product Price'] > 0) & (df['Quantity'] > 0)]
df = df.drop_duplicates(subset=['Customer ID', 'Purchase Date', 'Product Category'])
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

# ### Tối ưu hóa dự đoán churn
# Chuẩn bị dữ liệu
features = customer_features.reset_index().merge(df[['Customer ID', 'Churn']].drop_duplicates(), on='Customer ID')
X = features[['Total Purchase Amount', 'Transaction Count', 'Returns', 'Age']]
y = features['Churn']

# Trực quan hóa phân bố dữ liệu trước khi cân bằng
plt.figure(figsize=(8, 5))
sns.countplot(x=y, palette='Set2')
plt.title('Phân bố dữ liệu Churn trước khi cân bằng')
plt.xlabel('Churn')
plt.ylabel('Số lượng')
plt.tight_layout()
plt.savefig('churn_distribution_before_balancing.png')
print("Đã lưu biểu đồ phân bố dữ liệu trước khi cân bằng tại 'churn_distribution_before_balancing.png'")

# Xử lý mất cân bằng dữ liệu với SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Trực quan hóa phân bố dữ liệu sau khi cân bằng
plt.figure(figsize=(8, 5))
sns.countplot(x=y_resampled, palette='Set2')
plt.title('Phân bố dữ liệu Churn sau khi cân bằng với SMOTE')
plt.xlabel('Churn')
plt.ylabel('Số lượng')
plt.tight_layout()
plt.savefig('churn_distribution_after_balancing.png')
print("Đã lưu biểu đồ phân bố dữ liệu sau khi cân bằng tại 'churn_distribution_after_balancing.png'")

# Chia tập train/test với dữ liệu đã cân bằng
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Tối ưu LogisticRegression với GridSearchCV
param_grid_lr = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='f1')
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
print(f"Tham số tốt nhất cho LogisticRegression: {grid_lr.best_params_}")

# Thử nghiệm với RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='f1')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print(f"Tham số tốt nhất cho RandomForest: {grid_rf.best_params_}")

# Đánh giá hai mô hình
y_pred_lr = best_lr.predict(X_test)
y_pred_rf = best_rf.predict(X_test)

print("\nBáo cáo phân loại LogisticRegression:")
print(classification_report(y_test, y_pred_lr))
print("\nBáo cáo phân loại RandomForest:")
print(classification_report(y_test, y_pred_rf))

# Lưu mô hình tốt nhất (chọn dựa trên F1-score của class 1)
if classification_report(y_test, y_pred_lr, output_dict=True)['1']['f1-score'] > classification_report(y_test, y_pred_rf, output_dict=True)['1']['f1-score']:
    churn_model = best_lr
    y_pred = y_pred_lr
    print("Chọn LogisticRegression làm mô hình dự đoán churn.")
else:
    churn_model = best_rf
    y_pred = y_pred_rf
    print("Chọn RandomForest làm mô hình dự đoán churn.")
joblib.dump(churn_model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# ### Trực quan hóa bổ sung và đánh giá kết quả

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

# 2. Đánh giá độ chính xác mô hình dự đoán Churn (Confusion Matrix)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Ma trận nhầm lẫn của mô hình dự đoán Churn')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.tight_layout()
plt.savefig('churn_confusion_matrix.png')

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
