"""
bot.py – Slack Bolt ChatOps backend for FSD Training Pipeline.

Slash commands:
  /train          Start model training (with real-time epoch notifications)
  /train-stop     Stop a running training job
  /inference      Run inference on synthetic samples
  /sim-start      Start the PID controller simulator
  /sim-stop       Stop the running simulator
  /status         Show backend system health (CPU / MEM / DISK / GPU)
  /metrics        Show the latest training metrics summary
"""

import os
import sys
import json
import threading
import subprocess
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

from monitor import collect_snapshot, format_slack_message

# ──────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────
load_dotenv()
app = App(token=os.getenv("SLACK_BOT_TOKEN"))

# Global state – track running processes
_processes: dict = {}           # key → subprocess.Popen
_training_channel: str = ""     # channel to post epoch updates


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _is_alive(key: str) -> bool:
    return key in _processes and _processes[key].poll() is None


def _stream_training_output(proc, channel_id: str):
    """
    Read stdout of the training process line-by-line.
    Lines matching the epoch pattern are forwarded to the Slack channel
    so the team sees live progress.
    """
    from slack_sdk import WebClient
    client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
    thread_ts = None

    try:
        for line in iter(proc.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue

            # Forward epoch summary lines to Slack
            if line.startswith("[Epoch") or "best model" in line.lower():
                text = f"📊 `{line}`"
                try:
                    resp = client.chat_postMessage(
                        channel=channel_id,
                        text=text,
                        thread_ts=thread_ts,
                    )
                    if thread_ts is None:
                        thread_ts = resp["ts"]
                except Exception:
                    pass

        # Process finished – send summary
        rc = proc.wait()
        if rc == 0:
            client.chat_postMessage(
                channel=channel_id,
                text="✅ *Training completed successfully!*",
                thread_ts=thread_ts,
            )
        else:
            stderr_out = proc.stderr.read() if proc.stderr else ""
            client.chat_postMessage(
                channel=channel_id,
                text=f"❌ *Training exited with code {rc}*\n```\n{stderr_out[:1500]}\n```",
                thread_ts=thread_ts,
            )
    except Exception as exc:
        try:
            client.chat_postMessage(
                channel=channel_id,
                text=f"❌ *Training monitor error*: {exc}",
                thread_ts=thread_ts,
            )
        except Exception:
            pass


# ──────────────────────────────────────────────
# /train – Start training with real-time updates
# ──────────────────────────────────────────────
@app.command("/train")
def handle_train(ack, respond, command):
    ack()
    user = command["user_id"]
    channel = command["channel_id"]

    if _is_alive("training"):
        respond(f"<@{user}> ⚠️ Training is already running (PID {_processes['training'].pid}). Use `/train-stop` first.")
        return

    respond(f"<@{user}> 🚗💨 Starting E2E FSD training on this MacBook…")

    try:
        proc = subprocess.Popen(
            [sys.executable, "train_fsd.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered for real-time streaming
        )
        _processes["training"] = proc

        # Spawn a background thread that streams epoch output to Slack
        t = threading.Thread(
            target=_stream_training_output,
            args=(proc, channel),
            daemon=True,
        )
        t.start()

        respond(f"✅ Training started (PID {proc.pid}). Epoch updates will appear in this channel.")
    except Exception as exc:
        respond(f"❌ Failed to start training: {exc}")


# ──────────────────────────────────────────────
# /train-stop – Stop the running training job
# ──────────────────────────────────────────────
@app.command("/train-stop")
def handle_train_stop(ack, respond, command):
    ack()
    user = command["user_id"]

    if not _is_alive("training"):
        respond(f"<@{user}> No training job is currently running.")
        return

    proc = _processes["training"]
    respond(f"<@{user}> 🛑 Stopping training (PID {proc.pid})…")
    try:
        proc.terminate()
        proc.wait(timeout=10)
        respond("Training stopped and resources cleaned up.")
    except Exception as exc:
        proc.kill()
        respond(f"⚠️ Force-killed training: {exc}")


# ──────────────────────────────────────────────
# /inference – Run inference script
# ──────────────────────────────────────────────
@app.command("/inference")
def handle_inference(ack, respond, command):
    ack()
    user = command["user_id"]
    respond(f"<@{user}> 🚀 Running inference…")

    try:
        proc = subprocess.Popen(
            [sys.executable, "inference_fsd.py", "--samples", "5"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(timeout=60)
        if proc.returncode == 0:
            respond(f"✅ Inference output:\n```\n{stdout}\n```")
        else:
            respond(f"❌ Inference failed (code {proc.returncode}):\n```\n{stderr[:1500]}\n```")
    except subprocess.TimeoutExpired:
        proc.kill()
        respond("⏰ Inference timed out (>60s) and was killed.")
    except Exception as exc:
        respond(f"❌ Failed to run inference: {exc}")


# ──────────────────────────────────────────────
# /sim-start – Start the PID controller simulator
# ──────────────────────────────────────────────
@app.command("/sim-start")
def handle_sim_start(ack, respond, command):
    ack()
    user = command["user_id"]

    if _is_alive("simulator"):
        respond(f"<@{user}> ⚠️ Simulator is already running (PID {_processes['simulator'].pid}). Use `/sim-stop` first.")
        return

    respond(f"<@{user}> 🎮 Starting PID controller simulator…")
    try:
        proc = subprocess.Popen(
            [sys.executable, "run_controller.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _processes["simulator"] = proc
        respond(f"✅ Simulator started (PID {proc.pid}).")
    except Exception as exc:
        respond(f"❌ Failed to start simulator: {exc}")


# ──────────────────────────────────────────────
# /sim-stop – Stop the running simulator
# ──────────────────────────────────────────────
@app.command("/sim-stop")
def handle_sim_stop(ack, respond, command):
    ack()
    user = command["user_id"]

    if not _is_alive("simulator"):
        respond(f"<@{user}> No simulator is currently running.")
        return

    proc = _processes["simulator"]
    respond(f"<@{user}> 🛑 Stopping simulator (PID {proc.pid})…")
    try:
        proc.terminate()
        proc.wait(timeout=10)
        respond("Simulator stopped.")
    except Exception as exc:
        proc.kill()
        respond(f"⚠️ Force-killed simulator: {exc}")


# ──────────────────────────────────────────────
# /status – Backend system health dashboard
# ──────────────────────────────────────────────
@app.command("/status")
def handle_status(ack, respond, command):
    ack()
    try:
        snap = collect_snapshot()
        msg = format_slack_message(snap)

        # Append managed-process status
        proc_lines = ["\n*🤖 Managed Processes*"]
        for name, label in [("training", "Training"), ("simulator", "Simulator")]:
            if _is_alive(name):
                proc_lines.append(f"  • {label}: 🟢 Running (PID {_processes[name].pid})")
            else:
                proc_lines.append(f"  • {label}: 🔴 Stopped")

        respond(msg + "\n".join(proc_lines))
    except Exception as exc:
        respond(f"❌ Failed to collect system metrics: {exc}")


# ──────────────────────────────────────────────
# /metrics – Show latest training metrics summary
# ──────────────────────────────────────────────
@app.command("/metrics")
def handle_metrics(ack, respond, command):
    ack()

    # Check if training log file exists
    log_path = os.path.join("models", "training_log.json")
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                history = json.load(f)
            if not history:
                respond("📊 Training log is empty. Run `/train` first.")
                return

            latest = history[-1]
            best = min(history, key=lambda r: r.get("val_loss", float("inf")))

            lines = [
                "*📊 Training Metrics Summary*",
                f"Total epochs logged: {len(history)}",
                "",
                "*Latest epoch:*",
                f"  Epoch {latest.get('epoch', '?')} | Train Loss: `{latest.get('train_loss', 0):.4f}` | Val Loss: `{latest.get('val_loss', 0):.4f}`",
            ]
            if "mae" in latest:
                lines.append(f"  MAE: `{latest['mae']:.4f}` | R²: `{latest.get('r2', 0):.4f}`")

            lines += [
                "",
                "*Best epoch:*",
                f"  Epoch {best.get('epoch', '?')} | Val Loss: `{best.get('val_loss', 0):.4f}`",
            ]
            respond("\n".join(lines))
        except Exception as exc:
            respond(f"❌ Could not read training log: {exc}")
    else:
        respond("📊 No training log found. Run `/train` to generate metrics.")


# ──────────────────────────────────────────────
# @mention help
# ──────────────────────────────────────────────
@app.event("app_mention")
def handle_mention(body, say):
    say(
        "👋 *FSD ChatOps Bot* — available commands:\n"
        "`/train`        Start model training (live epoch updates)\n"
        "`/train-stop`   Stop running training\n"
        "`/inference`    Run inference on synthetic samples\n"
        "`/sim-start`    Start PID controller simulator\n"
        "`/sim-stop`     Stop the simulator\n"
        "`/status`       System health dashboard (CPU/MEM/GPU)\n"
        "`/metrics`      Latest training metrics summary"
    )


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")

    if not bot_token or not app_token:
        raise SystemExit(
            "❗ Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN in your .env file."
        )

    print("⚡️ FSD ChatOps Slack Bot is running!")
    print("   Commands: /train /train-stop /inference /sim-start /sim-stop /status /metrics")
    handler = SocketModeHandler(app, app_token)
    handler.start()
