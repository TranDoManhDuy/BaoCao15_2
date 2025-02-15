link data: https://www.kaggle.com/datasets/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2/data
Cách train: 
1. tải data ở link.tạo thư mục data và giải nén file vừa tại trong vào thư mục data
2. tạo thư mục checkpoints để lưu checkpoint model
3. cho chạy file train.py, khi quá trình train kết thúc checkpoint sẽ được lưu trong thư mục checkpoints

để đổi file train thì với dữ liệu khác trong tệp data thì đổi tên file dữ liệu (line 78 file train.py), đổi giá trị 'drop_last' thành True (line 95 file train.py)
