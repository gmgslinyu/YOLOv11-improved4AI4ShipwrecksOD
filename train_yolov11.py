from ultralytics import YOLO
import torch

def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')

    # 加载预训练模型
    model = YOLO("ultralytics/cfg/models/11/yolo11.yaml").load("dataset4OD/yolo11n.pt")

    # 训练参数配置
    training_args = {
        'data': 'dataset4OD/dataset.yaml',
        'epochs': 300,
        'imgsz': 640,
        'batch': 16,
        'mosaic': 0.0,  # 禁用mosaic
        'close_mosaic': 0,  # 禁用mosaic关闭
        'patience': 0,  # 设置为0禁用早停，默认值是50

        # 其他设置
        'project': 'runs/train',
        'name': 'underwater_yolov11',
        'exist_ok': True,
        'pretrained': True,
        'device': device
    }

    # 开始训练
    try:
        results = model.train(**training_args)
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        print(f"错误类型: {type(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    main() 