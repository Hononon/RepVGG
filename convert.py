import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from model import create_RepVGG_A2, repvgg_model_convert


def convert(args):
    model = create_RepVGG_A2(deploy=False, num_classes=5)  # 创建模型
    model.load_state_dict(torch.load(args.load), strict=False)  # 加载权重
    a = torch.rand(1, 3, 224, 224)
    model.eval()
    y_1 = model(a)
    repvgg_model_convert(model, save_path=args.save)  # 转换权重
    y_2 = model(a)
    print(((y_1 - y_2) ** 2).sum())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RepVGG(plus) Conversion')
    parser.add_argument('--load', default='save_model/A2-trained.pth', help='path to the weights file') #训练后的模型路径
    parser.add_argument('--save', default='save_model/A2-trained-converted.pth', help='path to the weights file') #转换后模型路径
    opt = parser.parse_args()
    convert(opt)
