import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        
    def forward(self, predictions, target):
        # target có dạng batch_size*7*7*30
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5) # bs * 7 * 7 * 30
        # predictions[..., 21:25] có kích thước bs*7*7*4 => lấy ra vị trí + kích thước khung dự đoán ở mỗi ô lưới.
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # tính iou của box thứ 1 so với đáp án
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25]) # tính iou của box thứ 2 so với đáp án
        #iou_b1 và iou_b2 đang giữ kích thước bs * 7 * 7
        
        # hàm unsqueeze(dim) giúp tạo ra thêm chiều mới vào vị trí chỉ định.
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        # => tạo ra tensor 2 * bs * 7 * 7, sử dụng dim = 0 so sánh, tìm iou max ở mỗi cell.
        iou_maxes, best_box = torch.max(ious, dim=0)
        # iou_maxes chứa các chỉ số iou cao nhất
        # best_box sẽ chứa vị trí của ô chứa giá trị đó
        # iou_maxes và best_box sẽ có kích thước bs * 7 * 7
        
        # exists_box chứa 1 danh sách 0, 1. Xác định xem các box nào có đối tượng
        # exists_box có kích thước bs*7*7.
        # thêm 1 chiều thứ 3, tạo thành bs*7*7*1
        exists_box = target[..., 20].unsqueeze(3)
        # ? để làm gì

        best_box = best_box.unsqueeze(-1)
        #FOR BOX
        box_predictions = exists_box * (
            (
                # (bs,7,7,1) * (bs*7*7*4) ???????????????????????????????????????????
                # => (bs, 7, 7, 4)
                
                # lấy ra các dự đoán hộp có Iou cao hơn và lấy kích thước của nó.
                best_box * predictions[..., 26:30]
                + (1 - best_box) * predictions[..., 21:25]
            )
        )
        # exists_box(bs*7*7*1) * reusult(bs*7*7*4)
        # => lấy ra các hộp dự đoán được so sánh với target, và bỏ đi các hộp mà vốn dĩ trong target ko có đối tượng
        
        # lấy ra được các ô trong đó chắc chắn có vật và lấy được cả dự đoán vị trí.
        
        # (bs*7*7*1) * (bs*7*7*4) => (bs * 7 * 7 * 4)
        box_targets = exists_box * target[..., 21:25]
        # lấy ra vị trí của vật trong các ô có tồn tại vật
        # size tensor = (bs * 7 * 7 * 4)
        
        # 2:4 (vi tri 2 va 3 trong (x, y, w, h))
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # tính toán ra các mất mát.
        box_loss = self.mse(
            # dạng ma trận (bs * 7 * 7, 4)
            torch.flatten(box_predictions, end_dim=2),
            torch.flatten(box_targets, end_dim=2)
        ) # trả về một giá trị duy nhất là sai số của từng thành phần trong box_predictions và box_target

        # (bs * 7 * 7 * 1)
        pred_box = (
            best_box * predictions[..., 25:26] +
            (1 - best_box) * predictions[..., 20:21]
        ) # lấy ra sai số về xác suất xuất hiện của các dự đoán trong từng ô.
        
        # lấy ra mất mát về xác suất xuất hiện trong từng ô dự đoán là có đối tượng
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )
        # exists_box (bs, 7, 7, 1) * (bs * 7 * 7 * 1) => 1 tổng
        
        # lấy ra mất mát về xác suất xuất hiện trong từng ô dự đoán là không có đối tượng
        # (1 - exists_box) * predictions[..., 25:26] => (bs * 7 * 7 * 1) * (bs * 7 * 7 * 1) => (bs, 7 * 7)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )
        
        #FOR CLASS LOSS
        # (bs,7,7,1) * (bs,7,7,20) => (bs,7,7,20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2), # (bs * 7 * 7, 20)
            torch.flatten(exists_box * target[..., :20], end_dim=-2) # (bs * 7 * 7, 20)
        )
        
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        
        return loss