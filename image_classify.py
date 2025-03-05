import warnings
warnings.filterwarnings('ignore')  # 忽略所有警告信息

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 数据处理和增强
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),  # 先调整到较大尺寸
    transforms.RandomCrop(224),     # 随机裁剪到224
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAutocontrast(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 数据集准备
train_dir = '/home/disk2/dachuang1-23/pythonProject1/exam/train_data'
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)

# 划分训练集、验证集、测试集
train_size = int(0.8 * len(train_dataset))
val_size = int(0.1 * len(train_dataset))
test_size = len(train_dataset) - train_size - val_size
train_data, val_data, test_data = random_split(train_dataset, [train_size, val_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f'训练数据集大小: {len(train_dataset)}')
print(f'训练集类别数: {len(train_dataset.classes)}')  # 6个类别
print(f'类别标签: {train_dataset.classes}')

# 在加载训练数据集后，保存类别映射
train_classes = train_dataset.classes

# 在文件开头添加模型保存路径
MODEL_DIR = '/home/disk2/dachuang1-23/pythonProject1/models/'
# 确保模型保存目录存在
os.makedirs(MODEL_DIR, exist_ok=True)

# 模型设计：使用预训练的ResNet18模型进行微调
def get_model(model_name='vit'):
    """获取预训练模型并进行微调设置"""
    if model_name == 'vit':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        num_ftrs = model.heads[0].in_features
        model.heads = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 6)
        )
        # 确保新添加的层可以计算梯度
        for param in model.heads.parameters():
            param.requires_grad = True
    elif model_name == 'convnext':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 6)
        )
        # 确保新添加的层可以计算梯度
        for param in model.classifier[2].parameters():
            param.requires_grad = True
    elif model_name == 'regnet':
        model = models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2), # 防止过拟合
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 6) # 6个类别
        )
        # 确保新添加的fc层可以计算梯度
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 6)
        )
    elif model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 6)
        )
    elif model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 6)
        )
    
    # 修改解冻策略
    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
        
    # 然后根据不同模型解冻特定层
    if model_name == 'vit':
        # 解冻最后几个transformer块和heads
        for name, param in model.named_parameters():
            if any(x in name for x in ['encoder.layers.11', 'encoder.layers.10', 'heads']):
                param.requires_grad = True
    elif model_name == 'convnext':
        # 解冻最后几个stage和classifier
        for name, param in model.named_parameters():
            if any(x in name for x in ['stages.3', 'stages.2', 'classifier']):
                param.requires_grad = True
    elif model_name == 'regnet':
        # 解冻最后几个stage和fc层
        for name, param in model.named_parameters():
            if 'stage4' in name or 'stage3' in name or 'fc' in name:
                param.requires_grad = True
    elif model_name == 'resnet50' or model_name == 'resnet18':
        for name, param in model.named_parameters():
            if 'layer4' in name or 'layer3' in name or 'fc' in name:
                param.requires_grad = True
    elif model_name == 'densenet121':
        for name, param in model.named_parameters():
            if 'denseblock4' in name or 'denseblock3' in name:
                param.requires_grad = True
    
    # 验证是否有参数需要梯度
    has_grad = False
    for param in model.parameters():
        if param.requires_grad:
            has_grad = True
            break
    if not has_grad:
        raise ValueError(f"No parameters require gradients for {model_name}")
    
    return model.to(device)

# 训练和验证模型
def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs=40):
    """训练模型并保存最佳状态"""
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    best_accuracy = 0.0
    best_model_path = os.path.join(MODEL_DIR, f'{model_name}_best.pth')  # 修改临时文件路径
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # 使用混合精度训练
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                scheduler.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/len(pbar):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'LR': f'{scheduler.get_last_lr()[0]:.6f}'
                })
        
        # 验证阶段
        val_accuracy = evaluate_model(model, val_loader)
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        
        # 早停检查
        early_stopping(1.0 - val_accuracy)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        # 保存最佳模型到临时文件
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved with accuracy: {best_accuracy:.4f}')
    
    # 加载最佳模型状态
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    # 删除临时文件
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        print(f"Removed temporary file: {best_model_path}")
    
    return model

# 在验证集上评估模型
def evaluate_model(model, val_loader):
    """评估模型，只返回总体准确率"""
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 只计算并返回总体准确率
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


# 添加模型集成训练函数
def train_ensemble():
    """训练多个模型并进行集成"""
    models_config = [
        ('vit', 'vit'),  # Vision Transformer
        ('convnext', 'convnext'),  # ConvNeXt
        ('regnet', 'regnet'),  # RegNet
        ('resnet50', 'resnet50'),  # ResNet50
        ('resnet18', 'resnet18'),  # ResNet18
        ('densenet121', 'densenet121')  # DenseNet121
    ]

    trained_models = []

    for model_name, model_prefix in models_config:
        print(f"\nTraining {model_name}...")
        model = get_model(model_name)

        optimizer = optim.AdamW([
            {'params':
                 (p for n, p in model.named_parameters() if any(x in n for x in ['classifier', 'fc', 'heads'])),
             'lr': 0.001},
            {'params':
                 (p for n, p in model.named_parameters() if
                  not any(x in n for x in ['classifier', 'fc', 'heads']) and p.requires_grad),
             'lr': 0.0001}
        ], weight_decay=0.01)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 传入model_name用于临时文件命名
        model = train_model(model, model_name, train_loader, val_loader, criterion, optimizer)

        # 在测试集上评估准确率
        test_accuracy = evaluate_model(model, test_loader)
        print(f"Testing {model_name}:")
        print(f'Test Accuracy: {test_accuracy:.4f}')

        # 保存最终模型
        save_path = os.path.join(MODEL_DIR, f'{model_prefix}-{test_accuracy:.4f}.pth')
        torch.save(model.state_dict(), save_path)
        trained_models.append((model_name, model))

    return trained_models

# 修改预测函数以支持集成
def ensemble_predict(models, pred_loader):
    """集成多个模型的预测结果"""
    all_predictions = []

    for _, model in models:
        model.eval()
        predictions = []
        with torch.no_grad():
            for images, _ in pred_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                predictions.append(probs)

        predictions = torch.cat(predictions)
        all_predictions.append(predictions)

    # 平均所有模型的预测概率
    ensemble_preds = sum(all_predictions) / len(all_predictions)
    final_preds = torch.argmax(ensemble_preds, dim=1).cpu().numpy()

    return final_preds


# 将预测结果写入 CSV 文件
def save_predictions(pred_labels, pred_dataset,
                     filename='/home/disk2/dachuang1-23/pythonProject1/exam/pred_result.csv'):
    # 获取所有图片文件名和对应的预测标签
    image_paths = [path for path, _ in pred_dataset.samples]
    image_names = [os.path.basename(path) for path in image_paths]

    # 使用训练数据集的类别映射
    classes = ['0', '1', '2', '3', '4', '5']
    pred_classes = [classes[label] for label in pred_labels]

    # 创建包含图片名称和预测标签的DataFrame
    pred_df = pd.DataFrame({
        'Image Name': image_names,
        'Label': pred_classes
    })

    # 自然排序
    def get_number(filename):
        # 从文件名中提取数字部分
        return int(''.join(filter(str.isdigit, filename)))

    # 按照图片编号排序
    pred_df['sort_key'] = pred_df['Image Name'].apply(get_number)
    pred_df = pred_df.sort_values('sort_key')
    pred_df = pred_df.drop('sort_key', axis=1)

    # 保存到CSV文件
    pred_df.to_csv(filename, index=False)
    print(f"预测结果已保存到: {filename}")


# 修改加载模型的代码
def load_model(model_name, save_path):
    model = get_model(model_name)
    # 添加 map_location 参数，根据当前设备加载模型
    model.load_state_dict(torch.load(save_path,
                                     weights_only=True,
                                     map_location=device))  # 使用全局定义的 device
    return model


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def main():
    """主函数，用于控制程序流程"""
    while True:
        print("\n请选择操作：")
        print("1. 训练集成模型并预测")
        print("2. 仅使用已有模型预测")
        print("3. 退出")
        
        choice = input("请输入选项（1-3）：")
        
        if choice == '1':
            print("开始训练集成模型...")
            trained_models = train_ensemble()
            print("开始预测新数据...")
            
            # 加载预测数据
            pred_data_dir = '/home/disk2/dachuang1-23/pythonProject1/exam/pred_data'
            pred_dataset = datasets.ImageFolder(root=pred_data_dir, transform=transform_test)
            pred_loader = DataLoader(pred_dataset, batch_size=32, shuffle=False)
            
            # 使用集成模型预测
            pred_labels = ensemble_predict(trained_models, pred_loader)
            save_predictions(pred_labels, pred_dataset)
            
        elif choice == '2':
            print("使用已有模型进行预测...")
            # 查找所有保存的模型文件
            model_files = {
                'resnet50': None,
                'resnet18': None,
                'densenet121': None,
                'vit': None,
                'convnext': None,
                'regnet': None
            }
            
            print("\n查找模型文件...")
            # 查找每个模型最新的权重文件
            for filename in os.listdir(MODEL_DIR):
                for model_name in model_files.keys():
                    if filename.startswith(model_name) and filename.endswith('.pth'):
                        print(f"找到 {model_name} 的权重文件: {filename}")
                        if model_files[model_name] is None or filename > model_files[model_name]:
                            model_files[model_name] = filename
                            print(f"更新 {model_name} 的最新权重文件为: {filename}")
            
            print("\n最终选择的模型文件:")
            for model_name, filename in model_files.items():
                if filename:
                    print(f"{model_name}: {filename}")
                else:
                    print(f"{model_name}: 未找到权重文件")
            
            trained_models = []
            print("\n开始加载模型...")
            for model_name, save_path in model_files.items():
                if save_path and os.path.exists(save_path):
                    try:
                        model = load_model(model_name, save_path)
                        model.eval()
                        trained_models.append((model_name, model))
                        print(f"成功加载模型: {save_path}")
                    except Exception as e:
                        print(f"加载模型 {save_path} 失败: {str(e)}")
            
            if trained_models:
                print(f"\n成功加载 {len(trained_models)} 个模型，开始集成预测...")
                pred_data_dir = '/home/disk2/dachuang1-23/pythonProject1/exam/pred_data'
                pred_dataset = datasets.ImageFolder(root=pred_data_dir, transform=transform_test)
                pred_loader = DataLoader(pred_dataset, batch_size=32, shuffle=False)
                
                pred_labels = ensemble_predict(trained_models, pred_loader)
                save_predictions(pred_labels, pred_dataset)
            else:
                print("未找到已保存的模型文件")
                
        elif choice == '3':
            print("程序结束")
            break
        else:
            print("无效的选择，请重新输入")


if __name__ == "__main__":
    main()
