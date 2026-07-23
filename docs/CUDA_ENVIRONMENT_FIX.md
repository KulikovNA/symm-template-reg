# CUDA environment diagnosis

The exact cause of the earlier `torch.cuda.is_available() == False` result was
execution-context isolation, not a broken `fracs` environment.

Inside the ordinary restricted Codex process, `/dev/nvidia0`,
`/dev/nvidiactl`, and `/dev/nvidia-uvm` were absent and `nvidia-smi` could not
contact the driver. The same repository command, executed with GPU device
access, used exactly:

- `/home/nikita/anaconda3/envs/fracs/bin/python`;
- Python 3.10.19;
- PyTorch 2.9.1+cu130;
- torchvision 0.24.1+cu130;
- CUDA build 13.0;
- NVIDIA driver 580.159.03.

It then saw `/dev/nvidia*`, loaded `libcuda.so.1`, initialized CUDA, and used an
NVIDIA GeForce RTX 3080 Ti Laptop GPU (compute capability 8.6). No Python,
PyTorch, torchvision, CUDA, or driver version was changed.

The complete evidence is in:

`/home/nikita/disser/fragment-template-registration-lab/work_dirs/cuda_diagnostics_20260716_133159/`

Pure CUDA tensor creation, matrix multiplication, backward, AdamW, float16
autocast, and GradScaler passed. The full model uses `amp_dtype="auto"`; on
this GPU it resolves to BF16. Full `SymmTemplateReg` forward, symmetry-aware
loss, backward, and optimizer step passed with finite values and gradients.
Peak diagnostic memory was 243.50 MiB allocated and 402.00 MiB reserved.

The first diagnostic deliberately exposed an additional numerical fact:
float16 worked for the minimal tensor smoke but overflowed some full-model
gradients. This is not a driver or installation failure. BF16 has the larger
exponent range needed by the current baseline and passed without changing the
architecture.

Run locally with:

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg
python tools/diagnose_cuda.py \
  --output-root /home/nikita/disser/fragment-template-registration-lab/work_dirs
```

If this command works in the terminal but a restricted automation process does
not expose `/dev/nvidia*`, grant that process GPU device access. Reinstalling
PyTorch cannot repair a missing device namespace and should not be attempted.
