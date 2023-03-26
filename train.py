import os
import wandb
import argparse

import torch
import torch.optim as optim
from torchvision import transforms

from model import create_RepVGG_A2 as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate, create_lr_scheduler, get_params_groups


def main(args):
    # 如果显卡可以，则用gpu进行训练
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("use {} device".format(device))
    print(torch.cuda.get_device_name(0))

    print(args)

    # if os.path.exists("./weights") is False:
    #     os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "linear" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "linear" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.SGD(pg, lr=args.lr, weight_decay=args.wd, momentum=0.9)
    lr_s = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                               warmup=True, warmup_epochs=1)
    loss_function = torch.nn.CrossEntropyLoss()

    epoch = args.epochs
    max_acc = 0

    wandb.init(project=args.project, name=args.model_name)

    for i in range(epoch):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                loss_fc=loss_function,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                cur_epoch=i,
                                                lr_scheduler=lr_s)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     loss_fc=loss_function,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     cur_epoch=i)

        wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'epoch': i})
        wandb.log({'val_loss': val_loss, 'val_acc': val_acc, 'epoch': i})

        # 保存最好的模型权重
        if val_acc > max_acc:
            folder = 'save_model'
            if not os.path.exists(folder):
                os.mkdir('save_model')
            max_acc = val_acc
            print('save best model')
            torch.save(model.state_dict(), args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RepVGG config')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)  # 学习率
    parser.add_argument('--wd', type=float, default=1e-5)  # 权重衰减

    # 数据集下载地质https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,default="flower_photos") #数据集路径

    #预训练模型下载地址：https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq
    parser.add_argument('--weights', type=str, default="pretrained_model/A2-pretrain.pth",help='initial weights path')  # 预训练模型路径 空为不加载预训练权重
    
    parser.add_argument('--freeze-layers', type=bool, default=False)  # 只训练全连接层
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--project', type=str, default='RepVGG') #wandb中的project
    parser.add_argument('--model_name', type=str, default='RepVGG-A2') #wandb中的name
    parser.add_argument('--save', type=str, default='save_model/A2-trained.pth') #训练后模型路径
    opt = parser.parse_args()

    main(opt)

