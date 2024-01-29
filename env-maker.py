
try:
    import environmentinator
except:
    import pip
    pip.main('install --user environmentinator'.split())
    import environmentinator

torch = environmentinator.ensure_module('torch', 'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
timm = environmentinator.ensure_module('timm')
transformers = environmentinator.ensure_module('transformers', 'transformers[torch]')
fairscale = environmentinator.ensure_module('fairscale')
pycocoevalcap = environmentinator.ensure_module('pycocoevalcap')

print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
