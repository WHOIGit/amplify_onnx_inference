# amplify_onnx_inference

## Container Use
```
podman build . -t onnx:latest
podman run -it --rm -e CUDA_VISIBLE_DEVICES=1 \
       --device nvidia.com/gpu=all \
       -v $(pwd)/models:/app/models \ 
       -v $(pwd)/inputs/:/app/inputs \
       -v $(pwd)/outputs:/app/outputs \
       onnx:latest models/PathToYourModel.onnx inputs/PathToBinDirectory 
# optional flags: 
#     --batch N           # required for models without pre-set input sizes
#     --classes LISTFILE  # to add class name headers to output score-matrix csv
#     --outdir DIRPATH    # directory to write files to. Default is './outputs'
#     --outfile FILENAME  # filename of output csv. Default is "{RUN_DATE}/{BIN_ID}.csv"
#     --force-notorch     # use non-torch datasets and dataloaders. If torch is not installed, this flag is automaticly set 
```
