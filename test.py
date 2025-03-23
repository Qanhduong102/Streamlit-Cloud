import pickle

# Lưu mô hình vào file
with open('revenue_model.pkl', 'wb') as f:
    pickle.dump(revenue_model, f)

print("Mô hình đã được lưu với scikit-learn 1.6.1")