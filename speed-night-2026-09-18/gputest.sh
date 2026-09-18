docker run --rm --gpus all --entrypoint python3 ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 -c "
import torch,time
a=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); b=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16)
for _ in range(3): c=a@b
torch.cuda.synchronize(); t=time.time(); n=0
while time.time()-t<4: c=a@b; n+=1
torch.cuda.synchronize(); dt=time.time()-t
x=torch.randn(64*1024*1024,device='cuda',dtype=torch.bfloat16); torch.cuda.synchronize(); t2=time.time()
for _ in range(20): y=x*1.0001
torch.cuda.synchronize(); print(f'{n*2*4096**3/dt/1e12:.1f} TFLOPS, membw {20*2*x.numel()*2/(time.time()-t2)/1e9:.0f} GB/s')
" 2>&1 | grep TFLOPS
