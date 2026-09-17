#!/bin/bash

echo "Killing existing uvicorn and cloudflared processes..."
pkill -f uvicorn || true
pkill -f cloudflared || true
sleep 2

echo "Activating conda environment and starting Uvicorn..."
# Source conda to allow activation from a bash script
source /home/unknown/miniconda3/etc/profile.d/conda.sh
conda activate oil_spill

# Start Uvicorn in the background using nohup
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &

echo "Starting cloudflared tunnel..."
# Start cloudflared in the background using nohup
nohup cloudflared tunnel --url http://localhost:8000 > tunnel.log 2>&1 &

echo "Waiting for Cloudflare tunnel URL..."
PUBLIC_URL=""
while [ -z "$PUBLIC_URL" ]; do
    sleep 1
    # Cloudflared outputs the URL in its logs, we extract it using grep
    if [ -f tunnel.log ]; then
        PUBLIC_URL=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' tunnel.log | head -n 1)
    fi
done

# Save and print the URL
echo "$PUBLIC_URL" > PUBLIC_URL.txt
echo ""
echo "====================================================="
echo "Pipeline successfully started!"
echo "Public URL: $PUBLIC_URL"
echo "Saved to PUBLIC_URL.txt"
echo "====================================================="
