import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import create_RepVGG_A2 as create_model
import time
import argparse


def main(args):
    start_time = time.time()
    device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # load image
    img_path = args.image_path
    assert os.path.exists(img_path), "file:'{}'dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N,C,H,W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r")as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=5, deploy=False).to(device)
    # load model weights
    weights_path = args.weights_path
    assert os.path.exists(weights_path), "file:'{}'dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)  # 不严格匹配权重，没有就不匹配

    # prediction
    model.eval()
    with torch.no_grad():

        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()  # 返回指定维度最大值的序号

    print_res = "class:{} prob:{:.6}".format(class_indict[str(predict_cla)],
                                             predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class:{:10} prob:{:.6}".format(class_indict[str(i)],
                                              predict[i].numpy()))
    print(f"consume time: {time.time() - start_time}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='2.jpg') #待预测目标路径
    parser.add_argument('--weights_path', type=str, default='save_model/A2-trained.pth') #使用转换后模型路径
    opt = parser.parse_args()

    main(opt)
