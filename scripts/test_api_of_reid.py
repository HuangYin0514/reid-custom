# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torchreid
import torch

# %%

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

# %%
model = torchreid.models.build_model(
    name='pcb_p4',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=False
)

model = model.to(device)

# %%
input = torch.randn(4,3,256,128)
# model(input).shape

# %%
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
    save_dir='log/pcb_p4',
    max_epoch=30,
    eval_freq=20,
    print_freq=20,
    test_only=False
)
