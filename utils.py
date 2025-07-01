# Bên trong class Moment của file utils.py

def contrastive_loss(self, x, labels, is_memory=False, des=None, relation_2_cluster=None):
    '''
    x (B, H)
    '''
    # --- THAY ĐỔI BẮT ĐẦU TỪ ĐÂY ---
    # 1. Chuyển đổi kiểu dữ liệu của các tensor đầu vào sang float32
    x = x.float()
    if des is not None:
        des = des.float()

    if is_memory:
        # 2. Chuyển đổi kiểu dữ liệu của các tensor lấy từ bộ nhớ sang float32
        ct_x = self.mem_features.to(self.config.device).float()
        ct_x_des = self.mem_features_des.to(self.config.device).float()
        ct_y = self.mem_labels
    else:
        idx = list(range(len(self.features)))
        if len(idx) > self.sample_k:
            sample_id = random.sample(idx, self.sample_k)
        else:
            sample_id = idx
        # 3. Chuyển đổi kiểu dữ liệu của các tensor lấy từ bộ nhớ sang float32
        ct_x = self.features[sample_id].to(self.config.device).float()
        ct_x_des = self.features_des[sample_id].to(self.config.device).float()
        ct_y = self.labels[sample_id]

    # --- KẾT THÚC THAY ĐỔI ---

    # Bây giờ tất cả các tensor đều là float32, các phép toán sẽ hoạt động chính xác
    # l2 normalize
    x = F.normalize(x, p=2, dim=1)
    ct_x = F.normalize(ct_x, p=2, dim=1)
    
    t1 = torch.mm(x, ct_x.T) + 1 # 0 <= cos + 1 <= 2
    
    if des is not None:
        des = F.normalize(des, p=2, dim=1)
        ct_x_des = F.normalize(ct_x_des, p=2, dim=1)
        t2 = torch.mm(des, ct_x_des.T)
    else:
        # Nếu des là None, cần khởi tạo t2 để tránh lỗi
        t2 = torch.zeros_like(t1)

    zeros = (torch.zeros_like(t1)).to(self.config.device)
    
    pos = torch.ones_like(t1)
    neg = torch.ones_like(t1)

    if relation_2_cluster is not None:
        labels_clusters = torch.tensor([relation_2_cluster[label.item()] for label in labels], device=self.config.device)
        ct_y_clusters = torch.tensor([relation_2_cluster[label.item()] for label in ct_y], device=self.config.device)
        relation_match = (labels_clusters.unsqueeze(1) == ct_y_clusters.unsqueeze(0)).float()
        neg = relation_match * (1.0 + 0.2* t2) + (1.0 - relation_match) * 1.0

    dot_product_tempered_pos = torch.where(pos > 0, pos * t1 / self.temperature, zeros)
    dot_product_tempered_neg = torch.where(neg > 0, neg * t1 / self.temperature, zeros)
    
    exp_dot_tempered_pos = (
        torch.exp(dot_product_tempered_pos - torch.max(dot_product_tempered_pos, dim=1, keepdim=True)[0].detach()) + 1e-5
    )
    exp_dot_tempered_neg = (
        torch.exp(dot_product_tempered_neg - torch.max(dot_product_tempered_pos, dim=1, keepdim=True)[0].detach()) + 1e-5
    )
    mask_combined_pos = (labels.unsqueeze(1).repeat(1, ct_y.shape[0]) == ct_y).to(self.config.device)
    mask_combined_neg = ~mask_combined_pos
    cardinality_per_samples = torch.sum(mask_combined_pos, dim=1)
    
    # Tránh chia cho 0 nếu một mẫu không có positive pair nào trong batch
    cardinality_per_samples = torch.where(cardinality_per_samples == 0, 1, cardinality_per_samples)

    sum_temp = torch.sum(exp_dot_tempered_pos * mask_combined_pos, dim=1, keepdim=True) \
        + torch.sum(exp_dot_tempered_neg * mask_combined_neg, dim=1, keepdim=True)
    log_prob = -torch.log(exp_dot_tempered_pos / (sum_temp + 1e-8)) # Thêm epsilon để tránh log(0)
    supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined_pos, dim=1) / cardinality_per_samples
    supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

    return supervised_contrastive_loss
