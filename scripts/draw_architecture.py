from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scenario_architecture" / "system_architecture.png"
FONT = "C:/Windows/Fonts/arial.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def box(draw, xy, title, lines, fill):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=16, fill=fill, outline="#263238", width=3)
    draw.text((x1 + 18, y1 + 14), title, fill="#101820", font=font(24))
    y = y1 + 52
    for line in lines:
        draw.text((x1 + 18, y), line, fill="#263238", font=font(18))
        y += 28


def arrow(draw, start, end, label):
    draw.line((start, end), fill="#455A64", width=4)
    draw.polygon([(end[0], end[1]), (end[0] - 14, end[1] - 8), (end[0] - 14, end[1] + 8)], fill="#455A64")
    draw.text(((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 - 24), label, fill="#37474F", font=font(16))


def main():
    image = Image.new("RGB", (2000, 1150), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 25), "LogiEdge System Architecture", fill="#101820", font=font(36))
    draw.text((40, 70), "Built lab path: sensor streams -> local inference -> durable custody -> optional uplink", fill="#455A64", font=font(20))

    box(draw, (40, 140, 390, 520), "Truck sensors", ["Temperature: 1 Hz", "Vibration RMS: 0.5 Hz", "Door: OPEN/CLOSE", "Payload: ts, truck_id,", "seq, value"], "#D7EEF7")
    box(draw, (470, 140, 860, 520), "Local MQTT broker", ["Mosquitto :1883", "Persistence enabled", "sensors/{temperature,", "vibration,door}", "QoS 1 sensor delivery"], "#D5F5D0")
    box(draw, (940, 110, 1510, 570), "Inference service", ["WindowFeatureExtractor", "MA(5) + 30 s window / 10 s step", "Six fused features + training stats", "TFLite FP32 or full INT8 via shim", "Softmax probabilities", "Debounce + ±3 C safety interlock", "MQTT inference QoS 1 / alerts QoS 2"], "#FBE5E5")
    box(draw, (1580, 140, 1950, 350), "Local custody", ["SQLite WAL", "alerts table", "continuous windows table", "synced flag", "health/watchdog state"], "#FFF2CC")
    box(draw, (1580, 410, 1950, 620), "PSI monitor", ["Reference: 300 clean", "Rolling 100 confidence scores", "PSI every 60 simulated s", "Alert threshold: 0.25", "Recovery target: < 0.10"], "#E7DDF5")
    box(draw, (940, 720, 1510, 980), "Optional uplink", ["SyncWorker every 15 s", "Unsent alert IDs", "ops/trucks/{id}/alerts", "Ops broker / operations centre", "Cellular gaps do not stop local ML"], "#DDEBF7")
    box(draw, (1580, 720, 1950, 980), "Operations centre", ["Ops MQTT broker", "Alert de-duplication by id", "Backend/dashboard", "Production TLS/ACL", "shown as deployment design"], "#E2F0D9")

    arrow(draw, (390, 330), (470, 330), "MQTT")
    arrow(draw, (860, 330), (940, 330), "subscribe")
    arrow(draw, (1510, 260), (1580, 260), "alerts + windows")
    arrow(draw, (1510, 470), (1580, 490), "confidence")
    arrow(draw, (1240, 570), (1240, 720), "unsynced alerts")
    arrow(draw, (1510, 850), (1580, 850), "MQTT QoS 1")
    image.save(OUT)


if __name__ == "__main__":
    main()
