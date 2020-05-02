import torchreid
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=4,
    batch_size_test=16,
    transforms=['random_flip', 'random_crop']
)

model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

load_path = '/home/hy/vscode/reid-custom/log/pcb_p4/model/model.pth.tar-8'
torchreid.utils.Load_trained_parameters(load_path).load_trained_model_weights(model)

model = model.to(device)

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    save_dir='log/resnet50',
    max_epoch=1,
    eval_freq=1,
    print_freq=1,
    # test_only=True
)
