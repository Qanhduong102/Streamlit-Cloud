import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc file CSV với các tham số tối ưu
df = pd.read_csv('customer_data.csv', parse_dates=['Purchase Date'], dtype={
    'Customer ID': 'int32', 
    'Product Price': 'int32', 
    'Quantity': 'int32', 
    'Total Purchase Amount': 'int32', 
    'Age': 'int32', 
    'Churn': 'int8', 
    'Year': 'int16', 
    'Month': 'int8'
})

# Điền giá trị thiếu trong cột 'Returns' bằng 0
df['Returns'] = df['Returns'].fillna(0).astype('float32')

# Tính toán Calculated_Total
df['Calculated_Total'] = df['Product Price'] * df['Quantity']

# Cập nhật Total Purchase Amount để khớp với Calculated_Total
df['Total Purchase Amount'] = df['Calculated_Total']

# Tính toán lại các cột bổ sung để kiểm tra
df = df.assign(
    Mismatch=lambda x: x['Total Purchase Amount'] != x['Calculated_Total'],
    Difference=lambda x: x['Total Purchase Amount'] - x['Calculated_Total']
)

# Hàm hiển thị dữ liệu với định dạng đẹp hơn
def display_df(title, dataframe, columns=None, n_rows=5):
    print(f"\n{title}:")
    if columns:
        print(dataframe[columns].head(n_rows))
    else:
        print(dataframe.head(n_rows))
    print(f"Kích thước: {len(dataframe)} dòng")

# Hiển thị thông tin cơ bản
display_df("5 dòng đầu tiên của DataFrame sau khi sửa", df)
print("\nThông tin cơ bản của DataFrame:")
print(df.info(memory_usage='deep'))
print("\nSố lượng giá trị thiếu trong mỗi cột:")
print(df.isnull().sum())

# Thống kê số lượng dòng không khớp
mismatch_count = df['Mismatch'].sum()
total_rows = len(df)
mismatch_pct = mismatch_count / total_rows * 100
print(f"\nSố dòng không khớp: {mismatch_count}/{total_rows} ({mismatch_pct:.2f}%)")

# Hiển thị các dòng không khớp và khớp (nếu có)
mismatch_cols = ['Customer ID', 'Product Price', 'Quantity', 'Total Purchase Amount', 'Calculated_Total', 'Difference']
display_df("5 dòng đầu tiên có tổng giá trị không khớp", df[df['Mismatch']], mismatch_cols)
display_df("5 dòng đầu tiên có tổng giá trị khớp", df[~df['Mismatch']], mismatch_cols)

# Thống kê chênh lệch chi tiết
print("\nThống kê chênh lệch (Total Purchase Amount - Calculated Total):")
diff_stats = df['Difference'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2)
print(diff_stats)

# Phân tích bổ sung: Chênh lệch theo Product Category và Payment Method
print("\nChênh lệch trung bình theo Product Category:")
print(df.groupby('Product Category')['Difference'].mean().round(2))
print("\nChênh lệch trung bình theo Payment Method:")
print(df.groupby('Payment Method')['Difference'].mean().round(2))

# Phân tích bổ sung với biểu đồ và lưu thành file ảnh
# 1. Doanh thu theo danh mục sản phẩm
print("\nDoanh thu theo danh mục sản phẩm:")
category_revenue = df.groupby('Product Category')['Total Purchase Amount'].sum()
print(category_revenue)
plt.figure(figsize=(8, 5))
category_revenue.plot(kind='bar', color='skyblue', title='Doanh thu theo danh mục sản phẩm')
plt.xlabel('Danh mục sản phẩm')
plt.ylabel('Tổng doanh thu')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('category_revenue.png')  # Lưu thành file ảnh
print("Đã lưu biểu đồ doanh thu theo danh mục vào 'category_revenue.png'")

# 2. Số lượng giao dịch theo ngày trong tuần
print("\nSố lượng giao dịch theo ngày trong tuần:")
purchases_by_day = df.groupby('Day of Week').size()
print(purchases_by_day)
plt.figure(figsize=(8, 5))
purchases_by_day.plot(kind='bar', color='lightgreen', title='Số lượng giao dịch theo ngày trong tuần')
plt.xlabel('Ngày trong tuần')
plt.ylabel('Số giao dịch')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('purchases_by_day.png')  # Lưu thành file ảnh
print("Đã lưu biểu đồ số lượng giao dịch theo ngày vào 'purchases_by_day.png'")

# 3. Top 5 khách hàng chi tiêu nhiều nhất
print("\nTop 5 khách hàng chi tiêu nhiều nhất:")
top_spenders = df.groupby('Customer ID')['Total Purchase Amount'].sum().nlargest(5)
print(top_spenders)
plt.figure(figsize=(8, 5))
top_spenders.plot(kind='bar', color='salmon', title='Top 5 khách hàng chi tiêu nhiều nhất')
plt.xlabel('Customer ID')
plt.ylabel('Tổng chi tiêu')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('top_spenders.png')  # Lưu thành file ảnh
print("Đã lưu biểu đồ top 5 khách hàng vào 'top_spenders.png'")

# Lưu file dữ liệu đã xử lý
df.to_csv('cleaned_customer_data.csv', index=False)
print("\nĐã lưu dữ liệu đã xử lý vào 'cleaned_customer_data.csv'")