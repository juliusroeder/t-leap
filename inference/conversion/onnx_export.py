import os.path

from models.tleap import TLEAP
import torch.onnx
import onnx
import onnxoptimizer
from onnxconverter_common import float16


def convert_to_onnx(model, input_data, target_folder, model_name="test", batch_size=None, onnx_opset_version=18, half_precision=True, optimize=True):

    model.eval()

    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    if batch_size:
        model_name += "_BS" + str(batch_size)
    else:
        model_name += "_dynamic_BS"

    target_onnx = os.path.join(target_folder, model_name + ".onnx")
    if optimize:
        target_optimized = os.path.join(target_folder, model_name + ".opt.onnx")

    if half_precision:
        target_half = os.path.join(target_folder, model_name + "_fp16.opt.onnx")

    torch.onnx.export(model,
                      input_data,
                      target_onnx,
                      export_params=True,
                      do_constant_folding=True,
                      opset_version=onnx_opset_version,
                      input_names=["image"],
                      output_names=["keypoint"],
                      dynamic_axes={"image": {0: 'batch_size'},
                                    "keypoint": {0: 'batch_size'}}
                      )

    onnx_model = onnx.load(target_onnx)
    onnx.checker.check_model(onnx_model, full_check=True)

    # Optimizer
    if optimize:
        optimized_onnx_model = onnxoptimizer.optimize(onnx_model)
        onnx.checker.check_model(optimized_onnx_model, full_check=True)
        onnx.save_model(optimized_onnx_model, target_optimized)

    # Half precision
    if half_precision:
        if optimize:
            model_fp16 = float16.convert_float_to_float16(optimized_onnx_model)
        else:
            model_fp16 = float16.convert_float_to_float16(onnx_model)
        onnx.checker.check_model(model_fp16, full_check=True)
        onnx.save(model_fp16, target_half)


if __name__ == "__main__":
    model = TLEAP(in_channels=3, out_channels=10,  depth=4, seq_length=2) # seq_length=1
    x = torch.randn(32, 2, 3, 200, 200, requires_grad=True) #[batch, seq_length, channels, height, width]

    convert_to_onnx(model, x, target_folder="../onnx_models", model_name="tleap_seq_length2")