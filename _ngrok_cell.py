# CELL 6: Start Backend THEN ngrok tunnel

import os
import subprocess
import time
import urllib.request
import json
import sys
from pyngrok import ngrok

PROJECT_DIR = '/kaggle/working/AI_GEN_IMAGE'
os.chdir(PROJECT_DIR)
os.environ['PYTHONPATH'] = PROJECT_DIR

log_path = PROJECT_DIR + '/backend.log'

# Kill old processes
print('Killing old processes...')
subprocess.run(['bash', '-c', 'pkill -f uvicorn 2>/dev/null; pkill -f ngrok 2>/dev/null; sleep 2'], capture_output=True)
time.sleep(2)

# 1. Start backend FIRST and wait for it to be ready
print('Starting FastAPI backend...')
subprocess.run(['bash', '-c', 'pkill -f uvicorn 2>/dev/null; sleep 1'], capture_output=True)

cmd = (
    'cd ' + PROJECT_DIR + ' && '
    'PYTHONPATH=' + PROJECT_DIR + ' '
    'HF_TOKEN=' + str(os.environ.get('HF_TOKEN', '')) + ' '
    'nohup python -m uvicorn app.main:app '
    '--host 0.0.0.0 --port 8000 '
    '--log-level info > ' + log_path + ' 2>&1 &'
)
subprocess.run(['bash', '-c', cmd], capture_output=True)
print('   Backend process started. Log: ' + log_path)
print()

# Wait for backend to actually start (check local first)
print('Waiting for backend on localhost:8000...')
backend_ready = False
for i in range(30):
    time.sleep(5)
    try:
        resp = urllib.request.urlopen('http://localhost:8000/health', timeout=5)
        data = json.loads(resp.read())
        print('   Backend local ready after ~' + str((i + 1) * 5) + 's')
        print('   GPU: ' + str(data.get('gpu_available')))
        backend_ready = True
        break
    except Exception:
        pass
    if i % 4 == 3:
        print('   Still waiting... (' + str((i + 1) * 5) + 's)')

if not backend_ready:
    print()
    print('!!! Backend not ready. Checking log:')
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        for line in lines[-30:]:
            sys.stdout.write(line.rstrip() + '\n')
            sys.stdout.flush()
    except Exception:
        sys.stdout.write('(log empty)\n')
    sys.stdout.flush()
    raise Exception('Backend failed to start. See log above.')

print()

# 2. Now start ngrok tunnel
print('Opening ngrok tunnel...')
try:
    ngrok.kill()
except Exception:
    pass
time.sleep(1)

http_tunnel = ngrok.connect(8000, bind_tls=True)
public_url = http_tunnel.public_url
api_url = public_url + '/api'

print()
print('=' * 65)
print('BACKEND PUBLIC URL:')
print('   ' + public_url)
print('   API Base: ' + api_url)
print('=' * 65)
print()

# 3. Final health check via public URL
print('Final health check via ngrok...')
try:
    resp = urllib.request.urlopen(public_url + '/health', timeout=10)
    data = json.loads(resp.read())
    print('SERVER READY!')
    print('   Backend: ' + str(data.get('backend')))
    print('   GPU: ' + str(data.get('gpu_available')))
except Exception as e:
    print('Warning: health check via ngrok failed (may be ok): ' + str(e)[:80])

print()
print('Backend log (last 30 lines):')
print('-' * 60)
try:
    with open(log_path, 'r') as f:
        lines = f.readlines()
    for line in lines[-30:]:
        sys.stdout.write(line.rstrip() + '\n')
        sys.stdout.flush()
except Exception:
    sys.stdout.write('(log empty)\n')
print('-' * 60)
