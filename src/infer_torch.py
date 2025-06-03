import json
import os
import argparse
import datetime as dt

from tqdm import tqdm
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# set import paths to project root
if __name__ == '__main__':
    import sys, pathlib
    PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
    if sys.path[0] != str(PROJECT_ROOT): sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets_torch import IfcbBinsDataset, DEFAULT_BLACKLIST

# if torch not being imported/installed, do:
# pip install onnxruntime-gpu[cuda,cudnn]
#onnxruntime.preload_dlls(directory="")
# see https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls

def argparse_init(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description='Train, Run, and perform other tasks related to ifcb and general image classification!')

    ## Run Vars ##
    parser.add_argument('MODEL', help='Path to a previously-trained model file')
    parser.add_argument('SRC', help='Resource(s) to be classified. Accepts a bin, an image, a text-file, or a directory. Directories are accessed recursively')
    parser.add_argument('--batch', type=int, help='Specify inference batchsize (for dynamically-batched MODEL only)')
    parser.add_argument('--type', dest='src_type', default='bin', choices=['bin','img'], help='File type to perform classification on. Defaults is "bin"')
    parser.add_argument('--outdir', default='./output/{RUN_DATE}', help='Default is "./output/{RUN_DATE}')
    parser.add_argument('--outfile', action='append',
        help='''Name/pattern of the output classification file.
                If TYPE==bin, files are created on a per-bin basis. OUTFILE must include "{BIN_ID}", which will be replaced with the a bin's id.
                A few patters are recognized: {BIN_ID}, {BIN_YEAR}, {BIN_DATE}, {INPUT_SUBDIRS}.
                A few output file formats are recognized: .json, .mat, and .h5 (hdf).
                Default for TYPE==bin is "D{BIN_YEAR}/D{BIN_DATE}/{BIN_ID}_class.h5"; Default for TYPE==img is "img_results.json".
             ''')
    parser.add_argument('--filter', nargs='+', metavar=('IN|OUT','KEYWORD'),
        help='Explicitly include (IN) or exclude (OUT) bins or image-files by KEYWORDs. KEYWORD may also be a text file containing KEYWORDs, line-deliminated.')

    return parser


def argparse_runtime_args(args):
    # Record Timestamp
    args.cmd_timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    run_date_str, run_time_str = args.cmd_timestamp.split('T')

    # Record GPUs
    args.gpus = [int(gpu) for gpu in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')]

    # ARG CORRECTIONS AND CHECKS
    if os.path.isdir(args.SRC) and not args.SRC.endswith(os.sep): args.SRC = args.SRC+os.sep

    # Format Output Directory
    args.outdir = args.outdir.format(RUN_DATE=run_date_str)

    # set OUTFILE defaults
    if not args.outfile:
        if args.src_type == 'bin': args.outfile=['D{BIN_YEAR}/D{BIN_DATE}/{BIN_ID}_class.h5']
        if args.src_type == 'img': args.outfile = ['img_results.json']

    # assert correct filter arguments
    if args.filter:
        if not args.filter[0] in ['IN', 'OUT']:
            argparse.ArgumentTypeError('IN|OUT must be either "IN" or "OUT"')
        if len(args.filter) < 2:
            argparse.ArgumentTypeError('Must be at least one KEYWORD')


def make_dataset(args, resize, dtype=torch.float32):

    # dataset filter if any
    filter_mode, filter_keywords = None,[]
    if args.filter:
        filter_mode = args.filter[0]
        for keyword in args.filter[1:]:
            if os.path.isfile(keyword):
                with open(keyword) as f:
                    filter_keywords.extend(f.read().splitlines())
            else:
                filter_keywords.append(keyword)

    # create dataset
    root_dir = args.SRC
    whitelist = None
    blacklist = None
    if args.src_type == 'bin':
        # Formatting Dataset
        if os.path.isdir(args.SRC):
            if filter_mode=='IN':
                whitelist = filter_keywords
            elif filter_mode=='OUT':
                blacklist = filter_keywords
        elif os.path.isfile(args.SRC) and args.SRC.endswith('.txt'):  # TODO TEST: textfile bin run
            with open(args.SRC,'r') as f:
                bins = f.read().splitlines()
            root_dir = os.path.commonpath(bins)
            whitelist = bins
        else: # single bin # TODO TEST: single bin run
            root_dir = os.path.dirname(args.SRC)
            bin_id = os.path.basename(args.SRC)
            whitelist = [bin_id]

        transforms = [v2.Resize((resize,resize)), v2.ToImage(), v2.ToDtype(dtype, scale=True)]
        if 'img_norm' in args and args.img_norm:
            norm = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.insert(2,norm)
            raise NotImplemented
        transform = v2.Compose(transforms)

        dataset = IfcbBinsDataset(
                    bin_dirs=[root_dir],
                    bin_whitelist=whitelist,
                    bin_blacklist=blacklist,
                    transform=transform,
                    with_sources=True,
                    shuffle=False,
                    use_len=True)

        return dataset

'''
            if not args.clobber:
                output_files = [os.path.join(args.outdir, ofile) for ofile in args.outfile]
                outfile_dict = dict(BIN_ID=bin_obj.pid,
                                    BIN_YEAR=bin_obj.year,
                                    BIN_DATE=bin_obj.yearday,
                                    INPUT_SUBDIRS=bin_obj.namespace)
                output_files = [ofile.format(**outfile_dict).replace(2*os.sep,os.sep) for ofile in output_files]
                if all([ os.path.isfile(ofile) for ofile in output_files ]):
                    print('{} result-file(s) already exist - skipping this bin'.format(bin_obj))
                    continue

'''

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


def main(args):

    # load model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    ort_session = ort.InferenceSession(args.MODEL, sess_options=sess_options, providers=providers)

    input0 = ort_session.get_inputs()[0]
    model_batch = input0.shape[0]
    img_size = input0.shape[-1]
    input_type = input0.type  # str, eg "tensor(float16)"
    input_type = getattr(torch, input_type[7:-1])

    if isinstance(model_batch, str):  # dynamic
        assert args.batch is not None, 'Must specify inference batch size for dynamically batched MODEL'
        dynamic_batching = True
        inference_batchsize = args.batch
    else:
        assert args.batch is None or model_batch==args.batch, 'MODEL is statically batched, inference batch size cannot be adjusted'
        dynamic_batching = False
        inference_batchsize = model_batch

    # initialize dataset
    bin_dataset = make_dataset(args, resize=img_size, dtype=input_type)
    #print(len(bin_dataset))
    image_loader = DataLoader(bin_dataset, batch_size=inference_batchsize,
                              num_workers=0, drop_last=False)

    # do inference
    pbar = tqdm(image_loader, desc=f'batchsize={inference_batchsize}')
    score_matrix = None
    for batch_tuple in pbar:
        batch, pids = batch_tuple[0], batch_tuple[1]
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

    score_matrix  #TODO to something with score_matrix
    output_classes = np.argmax(score_matrix, axis=1)
    output_scores = np.max(score_matrix, axis=1)



if __name__ == '__main__':
    parser = argparse_init()
    args = parser.parse_args()
    argparse_runtime_args(args)
    main(args)








