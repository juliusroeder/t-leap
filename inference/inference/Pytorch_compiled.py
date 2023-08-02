from models.tleap import TLEAP
import torch


torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True #should make sure the GPU has a high enough compute capability

model = TLEAP(in_channels=3, out_channels=10, depth=4, seq_length=2).to("cuda:0")
model.eval()
x = torch.randn(32, 2, 3, 200, 200, requires_grad=False, device="cuda:0") #[batch, seq_length, channels, height, width]

for i in range(10):
    torch_out = model(x) #first compile

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for i in range(100):
    torch_out = model(x)
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end)/1000)

import torch._dynamo
torch._dynamo.reset()

opt_model = torch.compile(model, mode="max-autotune", fullgraph=True).to("cuda:0")
opt_model.eval()

for i in range(10):
    torch_out = opt_model(x) #first compile

start.record()
for i in range(100):
    with torch.cuda.amp.autocast():
        torch_out = opt_model(x)
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end)/1000)


