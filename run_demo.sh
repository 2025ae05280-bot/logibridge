#!/bin/bash

echo "============================================="
echo "🚀 Initializing Edge AI Pipeline Demo Stack..."
echo "============================================="

# 1. Clean shutdown of any legacy processes and wipe the broker data cache
docker compose --profile dev down -v

# 2. Boot only the MQTT message brokers first to let them settle
docker compose --profile dev up -d logiedge_broker logiedge_ops_broker
sleep 3

# 3. Spin up the multi-threaded ML inference engine module
docker compose --profile dev up -d inference_engine
sleep 3

# 4. Fire up the data simulation metrics profile stream
SIM_ANOMALY=combined docker compose --profile dev up -d telemetry_simulator

echo "============================================="
echo "📊 Containers active. Building monitor grid..."
echo "============================================="

# 5. Initialize a fresh tmux window session canvas named 'demopanel'
tmux new-session -d -s demopanel -n 'ML-Pipeline'

# 6. Left Pane: Bind it to follow the live inference processing metrics
tmux send-keys -t demopanel:0 'docker compose --profile dev logs -f inference_engine' C-m

# 7. Right Pane: Split the column screen layout down the center
tmux split-window -h -t demopanel:0

# 8. Right Pane: Launch your real-time color-coded prediction dashboard
tmux send-keys -t demopanel:0.1 "python3 -c \"
import paho.mqtt.client as mqtt, json
def on_message(c, u, m):
    try:
        data = json.loads(m.payload.decode())
        if 'latency_ms' in data or 'confidence' in data or 'final_class' in data:
            final_class = data.get('final_class', 0)
            labels = ['NORMAL', 'WARNING', 'CRITICAL']
            status_label = labels[final_class] if final_class < len(labels) else 'UNKNOWN'
            color = '\\033[92m' if final_class == 0 else ('\\033[93m' if final_class == 1 else '\\033[91m')
            reset = '\\033[0m'
            print(f'📈 [Model Output] Topic: {m.topic} | TS: {data.get(\\\"ts\\\")} | Status: {color}{status_label}{reset} | Confidence: {data.get(\\\"confidence\\\", 0)*100:.2f}% | Latency: {data.get(\\\"latency_ms\\\", 0):.2f}ms')
    except Exception: pass
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = lambda c,u,f,r,p: c.subscribe('#')
client.on_message = on_message
client.connect('127.0.0.1', 1883)
print('📢 Dynamic Inference Streaming Dashboard Active...'); client.loop_forever()
\"" C-m

# 9. Attach to the newly constructed split-screen presentation array
tmux attach-session -t demopanel
