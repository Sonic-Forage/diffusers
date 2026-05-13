# Sonic-Forage Diffusers v0.38.0 Integration Map

Source release: <https://github.com/huggingface/diffusers/releases/tag/v0.38.0>  
Fork: <https://github.com/Sonic-Forage/diffusers>  
Local clone: `/opt/data/workspace/github-forks/diffusers`  
Release tag: `v0.38.0` / commit `275869dcae4ebcfee6a80253fdabc56033335020`

## What v0.38.0 adds that matters for Sonic-Forage / MindForge

### 1. ACE-Step 1.5 — highest-value audio/music target

- Diffusers files: `src/diffusers/pipelines/ace_step/pipeline_ace_step.py`, `src/diffusers/pipelines/ace_step/modeling_ace_step.py`
- Model family: `ACE-Step/Ace-Step1.5`
- Modality: text/lyrics to variable-length stereo music/audio at 48 kHz.
- Architecture note: Qwen3-style text encoder + `AutoencoderOobleck` waveform VAE + `AceStepTransformer1DModel` DiT using flow matching.
- Sonic-Forage use: autonomous radio stingers, scene beds, intro/outro sketches, lyric-conditioned mini songs.
- Best first path: use Diffusers directly inside a Modal GPU endpoint, then port native vLLM-Omni once prompts/latency/cost are validated.

### 2. LongCat-AudioDiT — second audio target

- Diffusers files: `src/diffusers/pipelines/longcat_audio_dit/pipeline_longcat_audio_dit.py`
- Model: `ruixiangma/LongCat-AudioDiT-1B-Diffusers`
- Modality: text-to-audio diffusion.
- Sonic-Forage use: SFX, ambience, transitions, non-vocal radio drops.
- Best first path: DiffusersAdapter or direct Diffusers endpoint, because audio post-processing may differ from vLLM-Omni's current image/video post processors.

### 3. Flux.2 Klein Inpaint + Flux.2 small decoder — practical visual tool

- Diffusers file: `src/diffusers/pipelines/flux2/pipeline_flux2_klein_inpaint.py`
- Also relevant: `black-forest-labs/FLUX.2-small-decoder` for faster Flux.2 decode.
- Sonic-Forage use: fast image cleanup/inpainting for posters, album art, thumbnails, character cards, QR proof props.
- vLLM-Omni status: local fork already has `Flux2KleinPipeline` and `Flux2Pipeline`; missing/next item is native `Flux2KleinInpaintPipeline` parity and small-decoder toggles.

### 4. Ernie-Image and Nucleus-MoE — new image generation lanes

- Ernie file: `src/diffusers/pipelines/ernie_image/pipeline_ernie_image.py`
- Nucleus file: `src/diffusers/pipelines/nucleusmoe_image/pipeline_nucleusmoe_image.py`
- Sonic-Forage use: comparison bakeoff against Flux/Qwen/GLM for poster art and stylized scene cards.
- vLLM-Omni status: local vLLM-Omni fork already registers `ErnieImagePipeline`; `NucleusMoEImagePipeline` appears not yet native and is a clean adapter candidate.

### 5. LTX-2 and HunyuanVideo 1.5 modular pipelines — video research lane

- LTX files: `src/diffusers/pipelines/ltx2/`
- Hunyuan files: `src/diffusers/pipelines/hunyuan_video1_5/`
- Sonic-Forage use: short-loop video beds, idents, show bumpers, image-to-video animation of generated art.
- vLLM-Omni status: local vLLM-Omni fork already registers `LTX2*`, `LTX23*`, `HunyuanVideo15Pipeline`, and `HunyuanVideo15ImageToVideoPipeline`, so this is closer to production than the audio additions.

### 6. LLaDA2 — weird/cool text diffusion lane

- Diffusers file: `src/diffusers/pipelines/llada2/pipeline_llada2.py`
- Modality: discrete diffusion language modeling with iterative unmasking.
- Sonic-Forage use: experimental script mutation, surreal taglines, alternate takes, text glitch FX.
- Best path: research sandbox first; not a drop-in replacement for vLLM text generation.

## Core library improvements to exploit

- Flash Attention 4 backend: try where compatible on new GPUs after driver/kernel checks.
- FlashPack loading: useful for faster cold starts/loading in Modal and RunPod workflows.
- Group offloading + TorchAO: useful for fitting bigger image/video/audio DiTs on cost-conscious GPUs.
- `ring_anything` CP backend: relevant to long sequence/video/audio parallelism research.
- Pipeline profiling utilities: use to get stage timing and memory for proof receipts before scaling.

## vLLM-Omni bridge strategy

The local vLLM-Omni fork already contains a generic `DiffusersAdapterPipeline`:

- File: `/opt/data/workspace/github-forks/vllm-omni/vllm_omni/diffusion/models/diffusers_adapter/pipeline_diffusers_adapter.py`
- Advantage: can serve almost any Diffusers pipeline through vLLM-Omni with near-zero per-model code.
- Limitation: no CFG parallel, no sequence parallel, no TeaCache/Cache-DiT, no step-wise execution/continuous batching, no native quantization.

Recommended pattern:

1. **Probe with direct Diffusers** in Modal for each model: one prompt, one output, log stage timing and peak VRAM.
2. **Serve through `DiffusersAdapterPipeline`** for quick MindForge API access when quality is good enough.
3. **Native vLLM-Omni port** only for winners that need throughput, batching, Cache-DiT/TeaCache, sequence/tensor parallelism, or custom API polish.

## Priority order

1. ACE-Step 1.5 direct Modal endpoint for Sonic-Forage radio/music assets.
2. LongCat-AudioDiT direct Modal endpoint for SFX/ambience.
3. Flux.2 Klein Inpaint parity/native port in vLLM-Omni.
4. Nucleus-MoE image adapter bakeoff.
5. LTX-2 / HunyuanVideo 1.5 production smoke tests using existing vLLM-Omni native support.
6. LLaDA2 text diffusion sandbox.

## Smoke-test snippets

### ACE-Step 1.5 direct Diffusers prototype

```python
import torch
from diffusers import AceStepPipeline

pipe = AceStepPipeline.from_pretrained("ACE-Step/Ace-Step1.5", torch_dtype=torch.bfloat16)
pipe.to("cuda")
out = pipe(
    prompt="old-timey radio jungle breaks, calm dream whisper, dusty tape, PLUR kandi rave signal",
    lyrics="Mind Expander on the air tonight",
    audio_duration=30,
    num_inference_steps=30,
)
out.audios[0].save("sonic_forage_ace_step_test.wav")
```

### LongCat-AudioDiT direct Diffusers prototype

```python
import torch
from diffusers import LongCatAudioDiTPipeline

pipe = LongCatAudioDiTPipeline.from_pretrained(
    "ruixiangma/LongCat-AudioDiT-1B-Diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
out = pipe("old-timey radio static turning into a glittery cyberpunk rave ambience", num_inference_steps=30)
out.audios[0].save("sonic_forage_longcat_test.wav")
```

### Flux.2 Klein Inpaint prototype

```python
import torch
from diffusers import Flux2KleinInpaintPipeline

pipe = Flux2KleinInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
# provide image + mask_image according to pipeline docs/signature
```

## Notes for safe productionization

- Keep all model tokens and commercial/license gates in Modal secrets or `/opt/data/.env`; do not commit them.
- Validate generated voice/music assets before upload or broadcast.
- Use bounded batch sizes and stage receipts: prompt, model, seed, runtime, GPU, output path, SHA-256.
- For public demos, use generated voices/music only where licenses permit.
