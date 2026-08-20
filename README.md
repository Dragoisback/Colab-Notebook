# Colab-Notebook — MOSS-TTS v1.5 Standalone Foundation Model

**High-Fidelity 48 kHz Stereo TTS and Zero-Shot Voice Cloning — Kaggle T4 x2 GPU Edition**  
Created by **AIQUEST Academy** · Powered by `OpenMOSS-Team/MOSS-TTS-v1.5`

[![AIQUEST Academy](https://img.shields.io/badge/AIQUESTAcademy-blueviolet?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@aiquestacademy)
[![Kaggle T4 x2](https://img.shields.io/badge/Kaggle-T4%20GPU%20x2-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]()

## Notebook
- `moss-tts-v1-5-standalone-foundation-model.ipynb` — main notebook (Kaggle + Colab compatible, robust to T4 fp16 / OOM / disk-full / HF 403)

## Quick Start (Kaggle)

1. **Settings → Accelerator → GPU T4 x2** (must be x2, not x1)  
2. **Internet → ON** (right panel)  
3. Run cells top-to-bottom (Cell 1 → 2 → 3 → 4)  
4. Wait for `✅ Models loaded successfully` in Cell 3  
5. Click the **Gradio public link** (`https://…gradio.live`) in Cell 4

> If Cell 2 reinstalls torch (rare, only if your Kaggle image has torch <2.4), do **Kernel → Restart** then **Run All** again — second run skips the heavy install and is fast.

## Fixes in This Fork

This fork fixes ~20 Kaggle-specific crashes (see `KAGGLE_TROUBLESHOOTING.md` for full list), including the critical `ImportError: _center from numpy._core.umath` (numpy 2.2 vs scipy mismatch):

- **Disk full** → cache redirected to `/kaggle/tmp/hf_cache` (70 GB scratch) + `mkdir -p` + disk-free prints
- **psutil missing**, **ffmpeg missing** (apt + sudo fallback), **torch reinstall loop** (smart skip if torch≥2.4)
- **Transformers 5.0 `dtype` vs `torch_dtype`**, **audio_tokenizer .to() crashes**, **single-GPU fallback**
- **T4 fp16 NaN** (`probability tensor contains nan`) → patched `sample_token` with float32 softmax + fallback `argmax`
- **CUDA OOM** → explicit `empty_cache`, actionable OOM message, per-GPU memory report
- **torchaudio.save codec fail** → `soundfile` fallback, **Gradio launch** → `0.0.0.0:7860` + `share` fallbacks + `allowed_paths`
- **Cell 4 `import torch` missing**, **reference_audio `""` handling**, **clear_btn defaults**, **duration control UI**

## Troubleshooting

See **`KAGGLE_TROUBLESHOOTING.md`** for every error + fix, plus “Still Seeing Errors?” checklist.

## Credits

- Model: [OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5)
- Original notebook: AIQUEST Academy (YouTube: [@aiquestacademy](https://www.youtube.com/@aiquestacademy))
- Kaggle T4 x2 robustness patches: this repo
