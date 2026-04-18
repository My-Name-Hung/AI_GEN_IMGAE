import os
import gc
import time
import torch

PROJECT_DIR = '/kaggle/working/AI_GEN_IMAGE'
os.environ['PYTHONPATH'] = PROJECT_DIR

import sys
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

os.environ['LORA_OUTPUTS_DIR'] = PROJECT_DIR + '/outputs'
print('LORA_OUTPUTS_DIR=' + os.environ['LORA_OUTPUTS_DIR'])

HF_TOKEN = os.environ.get('HF_TOKEN', '')

t0 = time.time()

try:
    # Reset lora_manager singleton (modules + global variable)
    for mod in list(sys.modules.keys()):
        if 'lora_manager' in mod or mod.startswith('app.'):
            del sys.modules[mod]
    # Reset global singleton so it re-scans with fresh _discovered dict
    import app.services.lora_manager as _lm
    _lm._lora_manager = None
    print('Singleton + module cache reset')

    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from app.services.lora_manager import get_lora_manager

    # Detect device BEFORE loading pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    print('Device: {} | dtype: {}'.format(device, dtype))

    # Load base pipeline
    print('Downloading SDXL Turbo (~6GB)...')
    pipeline = StableDiffusionPipeline.from_pretrained(
        'stabilityai/sdxl-turbo',
        token=HF_TOKEN,
        variant='fp16' if device == 'cuda' else None,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    print('Base model loaded')

    # Move pipeline to device
    pipeline = pipeline.to(device)
    print('Pipeline on {}'.format(device))

    # Load LoRA adapters
    lora_mgr = get_lora_manager()
    discovered = lora_mgr.available_types()
    print('LoRA Manager: {}'.format(discovered))
    if not discovered:
        print('   No LoRA found - using pure SDXL')
        print('   To use LoRA: upload outputs/ folder to Kaggle')
    else:
        for lora_type in discovered:
            ok = lora_mgr.load_adapter(lora_type, pipeline, scale=1.0, stack=False)
            status = 'OK' if ok else 'FAIL'
            print('   {} {}'.format(status, lora_type))
        if discovered:
            lora_mgr.unload_all(pipeline)

    # Apply scheduler + xformers
    if device == 'cuda':
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print('xformers enabled')
        except Exception as e:
            print('(xformers skip: {})'.format(e))
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # Warmup
    print()
    print('Warmup GPU...')
    if device == 'cuda':
        torch.cuda.synchronize()
        print('GPU warmup done')
    print('Base model ready')

    # Cleanup
    del pipeline
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print()
    print('=' * 60)
    print('PRE-DOWNLOAD DONE! Time: {:.1f}s'.format(elapsed))
    print('   Base model cached - Cell 6 will start FAST')
    print('=' * 60)

except Exception as e:
    print('Error: {}'.format(e))
    import traceback
    traceback.print_exc()
