# Kaggle Troubleshooting — MOSS-TTS v1.5 Notebook

This file explains the common errors you hit on Kaggle (especially **T4 x2**) and how the fixed notebook (`moss-tts-v1-5-standalone-foundation-model.ipynb`) solves them.

## Quick Start Checklist (Kaggle)

1. **Settings (right panel) -> Accelerator -> GPU T4 x2** (NOT T4 x1, NOT P100)
2. **Internet -> ON** (required for Hugging Face download)
3. Run cells **top to bottom**: Cell 1 → Cell 2 → Cell 3 → Cell 4
4. Wait for `✅ Models loaded successfully` in Cell 3 before launching Gradio
5. Use the **Gradio public link** ( `*.gradio.live` ) printed in Cell 4

If you reinstalled torch in Cell 2 (you see "Installing pinned PyTorch stack"), do **Kernel -> Restart** then **Run All** again — second run will skip the heavy reinstall and go straight to loading.

---

## Errors Fixed in This Version

### 1) `ModuleNotFoundError: No module named 'psutil'`
**Cause:** `psutil` is not preinstalled on some Kaggle images, but Cell 1 did `import psutil` unconditionally.  
**Fix:** Cell 1 now wraps `psutil` in `try/except` and auto-installs it via `pip install psutil` with fallback. RAM reporting is skipped gracefully if install fails.

### 2) `OSError: [Errno 28] No space left on device` during model download
**Cause:** Default HF cache is `/root/.cache/huggingface` on the 20 GB root overlay. MOSS-TTSv1.5 is ~15 GB, so the download fills root and crashes.  
**Fix:**
- Cell 1 and Cell 3 now create and use `/kaggle/tmp/hf_cache` (70 GB scratch) with fallback to `/tmp/hf_cache` on Colab.
- Sets **all** cache env vars: `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HUGGINGFACE_HUB_CACHE`.
- Creates directories with `os.makedirs(..., exist_ok=True)` before any `from_pretrained`.
- Prints free disk with `shutil.disk_usage` so you can see if space is low.

### 3) `apt-get install` silently fails (ffmpeg not found)
**Cause:** Original cell did `!apt-get update -qq && apt-get install -y ffmpeg ... > /dev/null 2>&1` — hid all errors, and didn't try `sudo` (Kaggle sometimes needs it). Then `torchaudio`/`torchcodec` later failed because ffmpeg was missing.  
**Fix:** Cell 2 now:
- Checks `DEBIAN_FRONTEND=noninteractive` install, tries with and without `sudo`, logs to `/tmp/apt.log` instead of `/dev/null`.
- Verifies with `shutil.which("ffmpeg")` and `ffmpeg -version` and warns if missing.

### 4) 10-minute `pip install torch==2.9.1+cu128` every run + kernel restart needed
**Cause:** Cell 2 forced a full reinstall of torch 2.9.1 even when Kaggle already has torch 2.4/2.5 which works fine with fp16. The reinstall is 2 GB and invalidates the current CUDA runtime until you restart the kernel — users get `RuntimeError: CUDA driver mismatch` or `ImportError` in Cell 3.  
**Fix:** Cell 2 now does **smart logic**:
```python
need_torch_reinstall = torch_version < 2.4 or not cuda_available
```
- If current torch >=2.4 and CUDA works → **skip** heavy reinstall, only ensure `torchcodec` is there.
- If reinstall *is* needed, prints "Kernel -> Restart recommended" and re-run will be fast (second run skips).

### 5) `torch.backends.cuda.enable_*` AttributeError
**Cause:** API moved in torch 2.4+ (`enable_cudnn_sdp` etc. deprecated) — call crashes on newer torch.  
**Fix:** Cell 1 wraps all `torch.backends.cuda.enable_*` calls in `try/except AttributeError`.

### 6) `TypeError: AutoModel.from_pretrained() got unexpected keyword argument 'torch_dtype'`
**Cause:** Transformers 5.0 renamed `torch_dtype` → `dtype`. Original code used `torch_dtype=dtype` which raises TypeError on 5.0, and `dtype=` fails on 4.x.  
**Fix:** Cell 3 inspects `inspect.signature(AutoModel.from_pretrained)` and picks the correct kwarg name, with fallback.

### 7) `AttributeError: 'AutoProcessor' has no attribute 'audio_tokenizer' / .to() failed`
**Cause:** `processor.audio_tokenizer` may be a custom codec object without `.to()` or already on correct device; unconditional `.to().float()` crashed.  
**Fix:** Cell 3 checks `hasattr(processor, "audio_tokenizer")` and `hasattr(tok, "to")`, tries `to(device).float()` and falls back gracefully with warning, never crashing.

### 8) Single GPU fallback crash (`device_map="auto"` on 1 GPU)
**Cause:** `device_map="auto"` shards aggressively; on T4 x1 it can OOM or place layers incorrectly.  
**Fix:** Cell 3 detects `torch.cuda.device_count()`:
- `>=2` → `device_map="auto"` (sharded)
- `1` → `device_map="cuda:0"` (single GPU, no sharding)
- `0` → `cpu`

### 9) `NameError: name 'torch' is not defined` in Cell 4
**Cause:** Cell 4 used `torch.Tensor`, `torch.cuda`, `torch.no_grad` but never did `import torch` — relied on Cell 3's import still being in memory. After a kernel restart (common after torch reinstall) Cell 4 would crash.  
**Fix:** Cell 4 now starts with explicit `import torch`, `import torchaudio`, `import gradio`, etc.

### 10) `RuntimeError: probability tensor contains either 'inf', 'nan' or element < 0` + CUDA assertion on T4
**Cause:** T4 is fp16-only (no bf16). Softmax + `multinomial` in fp16 can produce NaNs/Infs, especially with high `audio_temp`. The original monkeypatch looped `sys.modules` *before* the model was fully imported, so sometimes `sample_token` was never patched. Also `apply_top_k` etc. missing in fallback.  
**Fix:** Cell 3 now:
- Searches `sys.modules` **after** model load *plus* tries `importlib.import_module` for candidates.
- Checks that `utils_module` actually has `apply_top_k` etc.; if missing, injects fallback implementations.
- Patches all found `sample_token` with `nan_to_num`, `-inf` row fix, float32 softmax, zero-sum guard, and `multinomial`→`argmax` fallback.
- Logs `✅ Patched sample_token` or warns if patch missed.

### 11) `CUDA out of memory` (OOM) with `max_new_tokens=32768` on 16GB T4
**Cause:** 32768 tokens × 32 RVQ levels is huge VRAM; long prompts orContinuation mode with reference audio easily OOMs.  
**Fix:**
- Cell 4 now catches `RuntimeError: out of memory`, does `torch.cuda.empty_cache()` + `gc.collect()` and returns actionable message: try shorter text, split sentences, ensure T4 x2, lower `max_new_tokens`.
- Cell 3 prints per-GPU `memory_allocated/reserved` after load so you see headroom.
- Generates with `torch.no_grad()` and explicit `empty_cache` before/after.

### 12) `torchaudio.save` codec error / `FileNotFoundError: /kaggle/tmp/...`
**Cause:** `torchaudio.save` may fail if ffmpeg/libsndfile mismatch, and `/kaggle/working` or `/kaggle/tmp` doesn't exist.  
**Fix:**
- Cell 3 creates `/kaggle/working` and `/kaggle/tmp/hf_cache` up front.
- Cell 4 saves to `/kaggle/working` if exists else `tempfile.gettempdir()`, and on `torchaudio.save` failure falls back to `soundfile.write` (transposes `[channels, samples]` → `[samples, channels]`).

### 13) `TypeError: to_device` crash when batch contains `None`
**Fix:** `to_device` now handles `None` and tuple/list/dict recursion safely.

### 14) `Gradio launch failed: share=True` or `Cannot find empty port` on Kaggle
**Cause:** Original `demo.launch(share=True, show_error=True)` without `server_name`/`server_port` sometimes binds to `127.0.0.1` which Kaggle's proxy can't reach, or fails when `share` tunnel can't start.  
**Fix:** Cell 4 now does:
```python
demo.launch(share=True, server_name="0.0.0.0", server_port=7860, allowed_paths=["/kaggle/tmp","/tmp","/kaggle/working"])
```
with **two fallbacks**: if `share=True` fails → retry `share=False` on 7860, then bare `demo.launch(show_error=True)`. Also `queue(max_size=3)` before launch.

### 15) `reference_audio == ""` treated as valid path + missing file error
**Fix:** `generate_speech` normalizes `""` → `None` and checks `os.path.exists` before calling `processor.build_user_message`, returning a clear error if file missing.

### 16) `clear_btn` returning wrong defaults (text_temp 1.2 vs slider 1.5)
**Fix:** Clear lambda now returns `1.5` for `text_temp` (matching slider `value=1.5`) and correct 16-tuple order.

### 17) Duration control disabled UI stuck
**Fix:** `update_duration_controls` now correctly returns 3 `gr.update()` objects for `[expected_tokens, duration_hint, duration_control_enabled]` with proper `visible`/`minimum`/`maximum` handling and `interactive` toggles.

### 18) Hugging Face 403 / timeout via `hf-mirror.com`
**Fix:** Explicit `HF_ENDPOINT="https://huggingface.co"` (official) and `HF_HUB_ENABLE_HF_TRANSFER=1` for fast stable downloads. Retry loops (3 attempts) for both processor and model with `time.sleep`.

---

## Still Seeing Errors?

1. **Kernel -> Restart & Clear Outputs**, then **Run All** again (clears fragmented VRAM).
2. Check top of Cell 1: `Number of GPUs Available: 2` — if 1, change accelerator.
3. Ensure **Internet ON** (Cell 1 prints `Hugging Face connectivity: OK`).
4. In Cell 2, if you see `❌ ffmpeg not found`, run manually: `!sudo apt-get install -y ffmpeg libsndfile1-dev && ffmpeg -version`
5. If later OOM persists: edit Cell 4 `max_new_tokens_val = 32768` → `8192` and retry with shorter text.
6. If `probability tensor contains nan`: lower **audio_temp** from 1.7 → 1.2 and **audio_top_p** 0.9 → 0.85.

File issues or want a Colab version? Open an issue with the full traceback (copy the `❌ Error during generation` + `Traceback` from the log box).

