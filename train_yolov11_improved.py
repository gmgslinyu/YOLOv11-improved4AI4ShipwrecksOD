from ultralytics import YOLO
import torch
import torch.nn as nn


class UnderWaterEnhancementModule(nn.Module):
    """优化的水下目标检测增强模块"""

    def __init__(self, channels):
        super().__init__()

        # 1. 特征增强 - 使用深度可分离卷积
        self.conv = nn.Sequential(
            # 深度卷积
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.Hardswish(),  # 使用Hardswish替代SiLU
            # 点卷积
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Hardswish()
        )

        # 2. 通道注意力 - 使用更高效的压缩比
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),  # 更大的压缩比(1/8)
            nn.Hardswish(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        # 3. 自适应系数
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        identity = x

        # 特征增强
        feat = self.conv(x)

        # 通道注意力
        att = self.att(feat)
        feat = feat * att

        # 自适应残差连接
        return identity + feat * self.alpha.sigmoid()


def main():
    # 设置随机种子
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')
    # 加载模型
    model = YOLO("ultralytics/cfg/models/11/yolo11.yaml").load("dataset4OD/yolo11n.pt")

    # 应用增强模块 - 只在深层特征上应用
    for m in model.model.model:
        if isinstance(m, nn.Conv2d) and m.out_channels >= 128:  # 只在通道数>=128的层上应用
            m.add_module('auto_enhance', UnderWaterEnhancementModule(m.out_channels))

    # 训练参数配置
    training_args = {
        'data': 'dataset4OD/dataset.yaml',
        'epochs': 300,
        'imgsz': 640,
        'batch': 16,
        'mosaic': 0.0,  # 禁用mosaic
        'close_mosaic': 0,  # 禁用mosaic关闭
        'patience': 0,  # 设置为0禁用早停
        'project': 'runs/train',
        'name': 'underwater_yolov11_improved',
        'exist_ok': True,
        'pretrained': True,
        'device': device
    }

    # 训练
    try:
        results = model.train(**training_args)
        print("\n训练完成!")
        print(f"最终结果:")
        for k, v in results.results_dict.items():
            print(f"{k}: {v}")

    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        print(f"误类型: {type(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == '__main__':
    main()