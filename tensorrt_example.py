import argparse
import os

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models

import time


import numpy as np
import tensorrt as trt
import common
import torchvision.transforms as transforms

TRT_LOGGER = trt.Logger()

# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (
    0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401,
                      0.2564384629170883, 0.27615047132568404)



def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


def ONNX_build_engine(onnx_file_path, trt_file):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    batch_size = 64  
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, G_LOGGER) as parser:
        builder.max_batch_size = batch_size
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, common.GiB(1))
        config.set_flag(trt.BuilderFlag.FP16)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(
            onnx_file_path))

        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 32, 32),
                          (1, 3, 32, 32), (batch_size, 3, 32, 32))
        config.add_optimization_profile(profile)
        engine = builder.build_serialized_network(network, config)
        print("Completed creating Engine")
        with open(trt_file, "wb") as f:
            f.write(engine)
        return engine

#%%


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', action='store_true',
                        default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=32,
                        help='batch size for dataloader')
    args = parser.parse_args()
    print(args)

    cifar100_test_loader = get_test_dataloader(
        CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD,
        num_workers=1,
        batch_size=args.b)


    device = "cuda" if args.gpu else "cpu"
    net = models.resnet101(pretrained=True)
    net.fc=torch.nn.Linear(in_features=2048, out_features=100, bias=True)
    net = net.to(device)
    # # print(net)
    net.eval()
#%%
    t1 = time.time()
    for n_iter, (image, label) in enumerate(cifar100_test_loader):
        pred = net(image.to(device))
        # print(pred.shape)
    t2 = time.time()
    print(t2-t1)

#%% save onnx 
    input = torch.rand([1, 3, 32, 32]).to(device)
    onnx_file = "resnet101.onnx"

    if  os.path.exists(onnx_file):
        os.remove(onnx_file)
    torch.onnx.export(net, input, onnx_file,
                      input_names=['input'],  # the model's input names
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}},
                      # opset_version=12,
                      )
    print("onnx file generated!")


# %%generate tensorrt engine file
    trt_file = "resnet101.trt"

    ONNX_build_engine(onnx_file, trt_file)
    print("trt file generated!")


# %%　deserialize
    trt_file = "resnet101.trt"
    runtime = trt.Runtime(TRT_LOGGER)
    with open(trt_file, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        print("Completed creating Engine")
    context = engine.create_execution_context()
    context.set_binding_shape(0, (16, 3, 32, 32))

    inputs, outputs, bindings, stream = common.allocate_buffers(engine, 32)


# %%inference
    t1 = time.time()
    label_ls = []
    pred_ls = []
    for n_iter, (image, label) in enumerate(cifar100_test_loader):

        inputs[0].host = image.numpy()

        trt_outputs = common.do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=32)
        label_ls.extend(label.numpy())
        pred_ls.extend(np.array(trt_outputs[0]).reshape(
            [-1, 1000]).argmax(1).tolist())
    t2 = time.time()
    print(t2-t1)



#%%

    from torch2trt import torch2trt
    inputs = torch.rand([1, 3, 32, 32]).to(device)
    model_trt = torch2trt(net, [inputs], fp16_mode=True)

    t1 = time.time()
    label_ls = []
    pred_ls = []
    for n_iter, (image, label) in enumerate(cifar100_test_loader):

        output_trt = model_trt(image.to(device))

    t2 = time.time()
    print(t2-t1)

