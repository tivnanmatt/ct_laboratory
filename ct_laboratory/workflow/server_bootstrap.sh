#!/bin/bash
# Turn a fresh RunPod pod into a recon server (no network volume needed), run from the client:
#   server_bootstrap.sh <pod_id> [alias]   -> prints an ssh alias on success (exit 3/4/5 on a bad host)
# Steps: wait for ssh, health gate (CUDA on every GPU, CPU + launch latency), python deps,
# ct_laboratory from GitHub main (built for 4090 + 5090), store root /root/recon_assets.
set -u
POD=$1; ALIAS=${2:-staticct-srv}
CTLAB=$(cd "$(dirname "$0")/../.." && pwd)        # this ct_laboratory checkout (client); the server clones GitHub main
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
SSHO="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $KEY"
t0=$(date +%s); el() { echo $(( $(date +%s) - t0 )); }
until o=$(timeout 20 runpodctl ssh info $POD 2>/dev/null) && IP=$(echo "$o" | grep -o '"ip": "[^"]*' | cut -d'"' -f4) && PORT=$(echo "$o" | grep -o '"port": [0-9]*' | grep -o '[0-9]*$') && [ -n "$PORT" ] \
      && timeout 8 ssh $SSHO -p $PORT root@$IP true 2>/dev/null; do
  [ $(el) -gt ${SSH_WAIT:-300} ] && { echo "POD NOT READY after $(el) s"; exit 4; }; sleep 5; done
SSH="ssh $SSHO -p $PORT root@$IP"
echo "[t+$(el)s] ssh ready $IP:$PORT: $($SSH 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c | tr -s " "' | tr '\n' ';')"
scp -q $SSHO -P $PORT "$(dirname "$0")/pod_check.py" root@$IP:/root/pod_check.py
$SSH 'python /root/pod_check.py' || { echo "HOST REJECTED"; exit 3; }
$SSH 'cat > /root/setup.sh <<"EOF"
set -e
pip install -q --break-system-packages ninja cupy-cuda12x numpy==2.2.6 scipy==1.15.2 matplotlib==3.9.2 PyYAML==6.0.2 pandas==2.3.2 pydicom==3.0.1 xraydb==4.5.8 spekpy==2.5.4 scikit-image==0.25.2 scikit-learn==1.7.2 tqdm==4.66.4 imageio tifffile 2>&1 | grep -vi warn | tail -1
[ -d /root/ct_laboratory ] || git clone -q https://github.com/tivnanmatt/ct_laboratory.git /root/ct_laboratory
cd /root/ct_laboratory && git remote set-url --push origin PUSH_DISABLED && git pull -q --ff-only
TORCH_CUDA_ARCH_LIST="8.9;12.0+PTX" CUDA_HOME=/usr/local/cuda MAX_JOBS=32 PATH=/usr/local/cuda/bin:$PATH python setup.py build_ext --inplace > /root/build.log 2>&1
echo /root/ct_laboratory > $(python -c "import site;print(site.getsitepackages()[0])")/ct_laboratory.pth
mkdir -p /root/recon_assets
cd / && python -c "import ct_laboratory.reconstruction" && echo SETUP_DONE
EOF
nohup bash /root/setup.sh > /root/setup.log 2>&1 < /dev/null &'
until $SSH 'grep -qE "SETUP_DONE|rror" /root/setup.log || ! pgrep -f setup.sh >/dev/null'; do sleep 5; done
$SSH 'grep -q SETUP_DONE /root/setup.log' || { $SSH 'tail -5 /root/setup.log /root/build.log'; echo "SETUP FAILED"; exit 5; }
echo "[t+$(el)s] server ready: $($SSH 'cd /root/ct_laboratory && git log --oneline -1')"
python3 - "$ALIAS" "$IP" "$PORT" "$KEY" <<'EOF'
import os, re, sys
alias, ip, port, key = sys.argv[1:]
p = os.path.expanduser('~/.ssh/config'); s = open(p).read() if os.path.exists(p) else ''
s = re.sub(rf'\n# sct server {alias}.*?LogLevel ERROR\n', '\n', s, flags=re.S)
s += f'''
# sct server {alias} (RunPod pod, self-terminates)
Host {alias}
    HostName {ip}
    Port {port}
    User root
    IdentityFile {key}
    IdentitiesOnly yes
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
'''
open(p, 'w').write(s)
EOF
echo "SERVER_ALIAS $ALIAS  (ctlab run ... --remote $ALIAS)"
