import torch

from pytorch_tabnet.TabNet import train_dataset
from pytorch_tabnet.tab_model import TabNetClassifier

# 示例数据维度为 input_dim
input_dim = 10
output_dim = 1  # 假设是一个回归问题

# 创建一个 TabNet 模型
model = TabNetClassifier(input_dim=input_dim, output_dim=output_dim)

# 示例：使用 DataLoader 加载你的训练数据
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model, 'tabnet_model.pth')
