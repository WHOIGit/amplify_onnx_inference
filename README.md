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
#     --classes LISTFILE
#     --force-notorch 
```
