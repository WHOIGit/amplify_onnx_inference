import onnx

def convert_onnx_batchsize(model_path: str, output_path: str, batch=None):
    """
    Automatically convert ONNX model batch size:
      - If input model is static and batch=None -> make it dynamic
      - If input model is dynamic and batch=int -> make it static
      - If both are static -> change batch size
      - If both are dynamic -> no change unless you specify a batch size

    Parameters:
        model_path (str): Path to input ONNX model.
        output_path (str): Path to save converted ONNX model.
        batch (int or None):
            - None = convert to dynamic batch
            - int = convert to fixed batch size
    """
    print()
    model = onnx.load(model_path)

    def is_dynamic_dim(dim):
        return (dim.HasField("dim_param") and dim.dim_param != "") or \
               (not dim.HasField("dim_value")) or \
               (dim.HasField("dim_value") and dim.dim_value == 0)

    def update_batch_dim(tensor, batch, axis=0):
        dim = tensor.type.tensor_type.shape.dim[axis]

        # Detect original type
        original_is_dynamic = is_dynamic_dim(dim)
        original = dim.dim_param if original_is_dynamic else dim.dim_value
        print(f"Original batch dim for '{tensor.name}': {original} ({'dynamic' if original_is_dynamic else 'static'})")

        # Determine conversion action
        if batch is None:
            if original_is_dynamic: return # skip
            # Convert to dynamic
            dim.dim_value = 0
            dim.dim_param = "batch_size"
        else:
            # Convert to static
            dim.dim_param = ""
            dim.dim_value = int(batch)

    for tensor in model.graph.input:
        update_batch_dim(tensor, batch)
    for tensor in model.graph.output:
        update_batch_dim(tensor, batch)

    onnx.save(model, output_path)
    print(f"converted model saved to: {output_path}")