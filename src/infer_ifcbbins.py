import json
import os
import argparse
import datetime as dt

from tqdm import tqdm
import numpy as np
import onnxruntime as ort
import ifcb

# set import paths to project root
if __name__ == '__main__':
    import sys, pathlib
    PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
    #print('PROJECT_ROOT:',PROJECT_ROOT)
    if sys.path[0] != str(PROJECT_ROOT): sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2
    from src.datasets_torch import IfcbBinsDataset
    TORCH_MODE = True
except ImportError:
    from datasets import IfcbBinImageTransformer, MyDataLoader, IfcbBinDataset
    TORCH_MODE = False

# if torch not being imported/installed, do:
# pip install onnxruntime-gpu[cuda,cudnn]
ort.preload_dlls(directory="")
# see https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls

def argparse_init(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='Perform onnx-model inference on ifcb bins, without torch')

    ## Run Vars ##
    parser.add_argument('MODEL', help='Path to a previously-trained model file')
    parser.add_argument('BINS', nargs='+', help='Bin(s) to be classified. Can be a directory, bin-path, or list-file thereof')
    parser.add_argument('--batch', '-b', type=int, help='Specify inference batchsize (for dynamically-batched MODEL only)')
    parser.add_argument('--classes', help="Path to row-delimited classlist file. Required for output-csv's headers")
    parser.add_argument('--outdir', default='./outputs/{RUN_DATE}', help='Default is "./outputs/{RUN_DATE}')
    parser.add_argument('--outfile', default='{BIN_ID}.csv', help='Default is "{BIN_ID}.csv"')
    parser.add_argument('--force-notorch', action='store_true', help='Forces inference without torch dataloaders')

    return parser


def argparse_runtime_args(args):
    # Record Timestamp
    args.cmd_timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    args.run_date_str, args.run_time_str = args.cmd_timestamp.split('T')

    # Record GPUs
    args.gpus = [int(gpu) for gpu in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')]

    # read in classes from file, one class per line
    if args.classes and os.path.isfile(args.classes):
        with open(args.classes) as f:
            args.classes = f.read().strip().splitlines()

    bins = []
    for bin_thing in args.BINS:
        if os.path.isdir(bin_thing):
            dd = ifcb.DataDirectory(bin_thing, blacklist=('bad','skip','beads','temp','data_temp'))
            bins.extend( [binobj.fileset.basepath for binobj in dd] )
        elif bin_thing.endswith('.txt') or bin_thing.endswith('.list'):  # TODO TEST: textfile bin run
            with open(bin_thing, 'r') as f:
                bin_list_from_file = f.read().splitlines()
            bins.extend(bin_list_from_file)
        else:
            bins.append(bin_thing)
    args.BINS = bins

def softmax(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def pad_batch(batch:np.ndarray, target_batch_size:int):
    """
    Pads the input batch to target_batch_size by repeating the last element.

    Args:
        batch (np.ndarray): Input batch of shape (N, C, H, W)
        target_batch_size (int): Expected batch size by the model

    Returns:
        np.ndarray: Padded batch of shape (target_batch_size, C, H, W)
    """
    current_size = batch.shape[0]
    if current_size == target_batch_size:
        return batch
    elif current_size > target_batch_size:
        raise ValueError(f"Batch size {current_size} exceeds target size {target_batch_size}")

    pad_size = target_batch_size - current_size
    pad_shape = (pad_size,) + batch.shape[1:]
    pad = np.zeros(pad_shape, dtype=batch.dtype)
    padded_batch = np.concatenate([batch, pad], axis=0)
    return padded_batch


def write_output(args, bin_id, pids, score_matrix):
    outpath = os.path.join(args.outdir, args.outfile)
    outpath = outpath.format(RUN_DATE=args.run_date_str, BIN_ID=bin_id)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, 'w') as f:
        if args.classes:
            f.write(','.join(['pid']+args.classes)+'\n')
        for pid, score_row in zip(pids, score_matrix):
            str_row = ','.join(map(str,[pid]+score_row.tolist()))
            f.write(str_row+'\n')
    #print(f'{bin_id} Scores written to {outpath}')


def main(args):
    global TORCH_MODE

    if args.force_notorch or TORCH_MODE is False:
        from datasets import IfcbBinImageTransformer, MyDataLoader, IfcbBinDataset
        TORCH_MODE = False
    print('TORCH_MODE:', TORCH_MODE)

    # load model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    ort_session = ort.InferenceSession(args.MODEL, sess_options=sess_options, providers=providers)

    input0 = ort_session.get_inputs()[0]
    model_batch = input0.shape[0]
    img_size = input0.shape[-1]
    input_type = input0.type  # str, eg "tensor(float16)"

    if TORCH_MODE:
        input_type = getattr(torch, input_type[7:-1])
    else:
        input_type = getattr(np, input_type[7:-1]) if input_type!="tensor(float)" else np.float32

    dynamic_batching = True
    if isinstance(model_batch, str):  # dynamic
        assert args.batch is not None, 'Must specify inference batch size for dynamically batched MODEL'
        inference_batchsize = args.batch
    else:
        assert args.batch is None or model_batch==args.batch, 'MODEL is statically batched, inference batch size cannot be adjusted'
        dynamic_batching = False
        inference_batchsize = model_batch

    # initialize dataset
    if TORCH_MODE:
        transforms = [v2.Resize((img_size,img_size)), v2.ToImage(), v2.ToDtype(input_type, scale=True)]
        #if img_norm:
        #    norm = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #    transforms.insert(2, norm)
        transformer = v2.Compose(transforms)
    else:
        transformer = IfcbBinImageTransformer(img_size, dtype=input_type)

    pbar = tqdm(args.BINS, desc=f'batchsize={inference_batchsize}', unit='bins')
    for bin_accessor in pbar:
        img_pids = []
        score_matrix = None

        if TORCH_MODE:
            root_dir = os.path.dirname(bin_accessor)
            bin_id = os.path.basename(bin_accessor)
            dataset = IfcbBinsDataset(bin_dirs=[root_dir], bin_whitelist=[bin_id],
                transform=transformer, with_sources=True, shuffle=False, use_len=False)
            dataloader = DataLoader(dataset, batch_size=inference_batchsize,
                                    num_workers=0, drop_last=False)
            bin_pid = list(dataset.iter_binfilesets())[0].pid.pid
        else:
            dataset = IfcbBinDataset(bin_accessor)
            dataloader = MyDataLoader(dataset, inference_batchsize, transformer)
            bin_pid = dataset.pid

        # do inference
        for batch_tuple in dataloader:
            batch,batch_pids = batch_tuple[0], batch_tuple[1]
            if TORCH_MODE:
                batch = batch.numpy()
            size_of_batch = batch.shape[0]

            if dynamic_batching or size_of_batch == inference_batchsize:
                outputs = ort_session.run(None, {input0.name:batch})
            else:
                #print('Last batch difference:', inference_batchsize-size_of_batch)
                batch = pad_batch(batch, inference_batchsize)
                outputs = ort_session.run(None, {input0.name:batch})
                outputs = [output[:size_of_batch] for output in outputs]

            batch_score_matrix = outputs[0]
            batch_score_matrix = softmax(batch_score_matrix, axis=1)
            if score_matrix is None:
                score_matrix = batch_score_matrix
            else:
                score_matrix = np.concatenate([score_matrix, batch_score_matrix], axis=0)
            img_pids.extend(batch_pids)

        # write a score matrix csv for each bin
        write_output(args, bin_pid, img_pids, score_matrix)


if __name__ == '__main__':
    parser = argparse_init()
    args = parser.parse_args()
    argparse_runtime_args(args)
    main(args)








