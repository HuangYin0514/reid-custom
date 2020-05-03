import torchreid
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


state_dict = {
    'name': 'pcb_p6',
    'is_pretrained': False,
    'load_path_url': '/home/hy/vscode/reid-custom/log/tensorboard/log/resnet50/model/model.pth.tar-60',
    'batch_size_train': 4,
    'batch_size_test': 16,
    'height': 384,
    'width': 128,
    'max_epoch': 2,
    'eval_freq': 1,
    'print_freq': 1
}


datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=state_dict['height'],
    width=state_dict['width'],
    batch_size_train=state_dict['batch_size_train'],
    batch_size_test=state_dict['batch_size_test'],
    transforms=['random_flip', 'random_crop']
)

model = torchreid.models.build_model(
    name=state_dict['name'],
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)


torchreid.utils.Load_trained_parameters(
    is_pretrained=state_dict['is_pretrained'],
    model=model,
    load_path_url=state_dict['load_path_url'],
    device=device,
)


model = model.to(device)


optimizer = torchreid.optim.build_optimizer(
    model, optim='sgd', lr=0.1, staged_lr=True,
    new_layers=['parts_avgpool', 'dropout', 'conv5', 'classifier'], base_lr_mult=0.01,
    weight_decay=0.0005, momentum=0.9
)


scheduler = torchreid.optim.build_lr_scheduler(
    optimizer, lr_scheduler='multi_step', stepsize=[41, ],
    gamma=0.1
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)


engine.run(
    save_dir='log/'+state_dict['name'],
    max_epoch=state_dict['max_epoch'],
    eval_freq=state_dict['eval_freq'],
    print_freq=state_dict['print_freq'],
    # test_only=True,
    # visrank=True,
)
