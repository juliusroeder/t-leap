import torch # import first! Otherwise execution providers arent found!
import onnxruntime


def run_inference(dtype=torch.float):

    print(onnxruntime.get_available_providers())

    session = onnxruntime.InferenceSession("../onnx_models/tleap_seq_length2_dynamic_BS_fp16.opt.onnx", providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
    io_binding = session.io_binding()

    x = torch.randn(32, 2, 3, 200, 200, requires_grad=False, device="cpu", dtype=dtype) # dim 1 --> seq length
    io_binding.bind_cpu_input('image', x.numpy())
    io_binding.bind_output('keypoint')

    for i in range(10):
        session.run_with_iobinding(io_binding) #first compile

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(100):
        session.run_with_iobinding(io_binding) #first compile
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end)/1000)

if __name__ == "__main__":
    run_inference(dtype=torch.half)