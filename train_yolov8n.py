from ultralytics import YOLO
import torch

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')

    # 加载YOLOv8n模型
    model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml").load("dataset4OD/yolov8n.pt")  # 修改预训练模型路径

    # 训练参数配置
    training_args = {
        'data': 'dataset4OD/dataset.yaml',  # 数据集配置文件
        'epochs': 300,             # 训练轮数
        'batch': 16,              # 批次大小
        'imgsz': 640,             # 图像大小
        'mosaic': 0.0,            # 禁用mosaic
        'close_mosaic': 0,        # 禁用mosaic关闭
        'patience': 0,            # 设置为0禁用早停，默认值是50
                                 # 如果连续50个epoch没有改善就会停止训练

        # 保存设置
        'project': 'runs/train',
        'name': 'underwater_yolov8',
        'exist_ok': True,
        'save_period': 50         # 每50轮保存一次
    }

    # 开始训练
    try:
        results = model.train(**training_args)
        print("\n训练完成!")
        print(f"最终结果:")
        for k, v in results.results_dict.items():
            print(f"{k}: {v}")
        
    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        print(f"错误类型: {type(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    main() 