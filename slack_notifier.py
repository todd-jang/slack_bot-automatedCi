"""
slack_notifier.py – Send real-time per-epoch training metrics to Slack.

Usage in train_fsd.py:
    from slack_notifier import SlackTrainingNotifier
    notifier = SlackTrainingNotifier(channel="#ml-training")
    # inside training loop:
    notifier.on_epoch_end(epoch, num_epochs, train_loss, val_loss, extra_metrics={...})
    # at the end:
    notifier.on_training_complete(best_val_loss, model_path)
"""

import os
import json
import time
from typing import Optional, Dict, Any

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    _HAS_SLACK = True
except ImportError:
    _HAS_SLACK = False


class SlackTrainingNotifier:
    """Sends structured training updates to a Slack channel."""

    def __init__(
        self,
        channel: str = "#ml-training",
        token: Optional[str] = None,
        enabled: bool = True,
    ):
        self.channel = channel
        self.enabled = enabled and _HAS_SLACK
        self.token = token or os.getenv("SLACK_BOT_TOKEN", "")
        self._client: Optional[Any] = None
        self._thread_ts: Optional[str] = None  # thread for all updates
        self._start_time: float = 0.0
        self._history: list = []  # store epoch metrics locally

        if self.enabled and self.token:
            self._client = WebClient(token=self.token)

    # ── helpers ──────────────────────────────────────────
    def _post(self, text: str, blocks: Optional[list] = None) -> Optional[str]:
        """Post a message; return ts (timestamp ID)."""
        if not self._client:
            print(f"[SlackNotifier] (no client) {text}")
            return None
        try:
            resp = self._client.chat_postMessage(
                channel=self.channel,
                text=text,
                blocks=blocks,
                thread_ts=self._thread_ts,
            )
            return resp["ts"]
        except SlackApiError as e:
            print(f"[SlackNotifier] Slack error: {e.response['error']}")
            return None
        except Exception as e:
            print(f"[SlackNotifier] Error: {e}")
            return None

    @staticmethod
    def _progress_bar(current: int, total: int, width: int = 20) -> str:
        filled = int(width * current / total)
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def _duration_str(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"

    # ── public callbacks ─────────────────────────────────
    def on_training_start(self, total_epochs: int, batch_size: int, lr: float, device: str):
        """Call once before training begins."""
        self._start_time = time.time()
        text = (
            f"🚀 *Training Started*\n"
            f"• Epochs: {total_epochs}\n"
            f"• Batch size: {batch_size}\n"
            f"• Learning rate: {lr}\n"
            f"• Device: {device}"
        )
        ts = self._post(text)
        if ts and not self._thread_ts:
            self._thread_ts = ts  # subsequent updates go into this thread

    def on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        extra_metrics: Optional[Dict[str, float]] = None,
        best: bool = False,
    ):
        """Call at the end of every epoch."""
        elapsed = time.time() - self._start_time
        bar = self._progress_bar(epoch, total_epochs)
        eta = (elapsed / epoch) * (total_epochs - epoch) if epoch > 0 else 0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **(extra_metrics or {}),
        }
        self._history.append(record)

        # Build message
        star = " ⭐ *new best*" if best else ""
        lines = [
            f"📊 *Epoch {epoch}/{total_epochs}*  `{bar}` {epoch*100//total_epochs}%",
            f"   Train Loss: `{train_loss:.4f}` | Val Loss: `{val_loss:.4f}`{star}",
        ]
        if extra_metrics:
            extras = " | ".join(f"{k}: `{v:.4f}`" for k, v in extra_metrics.items())
            lines.append(f"   {extras}")

        lines.append(f"   ⏱️ Elapsed: {self._duration_str(elapsed)} | ETA: {self._duration_str(eta)}")
        self._post("\n".join(lines))

    def on_training_complete(self, best_val_loss: float, model_path: str):
        """Call once after all epochs are done."""
        elapsed = time.time() - self._start_time
        text = (
            f"✅ *Training Complete*\n"
            f"• Best Val Loss: `{best_val_loss:.4f}`\n"
            f"• Model saved: `{model_path}`\n"
            f"• Total time: {self._duration_str(elapsed)}\n"
            f"• Total epochs: {len(self._history)}"
        )
        self._post(text)

    def on_training_error(self, error: str):
        """Call if training crashes."""
        self._post(f"❌ *Training Failed*\n```\n{error}\n```")

    def get_history(self) -> list:
        """Return all epoch records collected so far."""
        return self._history
