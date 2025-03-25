import pandas as pd

# Đọc file customer_data.csv
df = pd.read_csv('customer_data.csv')

# 1. Loại bỏ cột Customer Age (trùng với Age)
df = df.drop(columns=['Customer Age'])

# 2. Chuẩn hóa dữ liệu
df['Gender'] = df['Gender'].str.lower()  # Chuyển Gender thành chữ thường
df['Payment Method'] = df['Payment Method'].str.capitalize()  # Chuẩn hóa Payment Method (Paypal, Credit, etc.)

# 3. Tính lại Total Purchase Amount (nếu cần)
df['Total Purchase Amount'] = df['Product Price'] * df['Quantity']

# 4. Trích xuất Year, Month, Day of Week từ Purchase Date
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])  # Chuyển sang định dạng datetime
df['Year'] = df['Purchase Date'].dt.year
df['Month'] = df['Purchase Date'].dt.month
df['Day of Week'] = df['Purchase Date'].dt.day_name()

desired_columns = [
    'Customer ID', 'Purchase Date', 'Product Category', 'Product Price', 'Quantity',
    'Total Purchase Amount', 'Payment Method', 'Returns', 'Customer Name', 'Age',
    'Gender', 'Churn', 'Year', 'Month', 'Day of Week'
]
df = df[desired_columns]

# 6. Lưu vào file mới
df.to_csv('cleaned_customer_data.csv', index=False)
print("Đã tạo file cleaned_customer_data.csv")