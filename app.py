"""Live call subtitles.

Captures whatever your PC is playing (the other person's voice in a Telegram /
Discord / any call) via WASAPI loopback, runs it through a local Whisper model,
and shows live English subtitles in an always-on-top overlay.

Usage:
  python app.py                     # overlay subtitles, auto GPU/CPU
  python app.py --list-devices      # show output devices you can capture
  python app.py --output "Buds"     # capture a specific output device by name
  python app.py --model large-v3    # best translation quality (slower to load)
  python app.py --task transcribe   # show original language instead of English
  python app.py --no-overlay        # console subtitles only
"""

import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np


def _add_nvidia_dll_dirs():
    """Let ctranslate2 find cuBLAS/cuDNN installed from the pip wheels."""
    import site

    candidates = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        candidates.append(os.path.join(sp, "nvidia", "cublas", "bin"))
        candidates.append(os.path.join(sp, "nvidia", "cudnn", "bin"))
    for d in candidates:
        if os.path.isdir(d):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


_add_nvidia_dll_dirs()

import pyaudiowpatch as pyaudio  # noqa: E402

TARGET_SR = 16000  # Whisper's input sample rate


# --------------------------------------------------------------------------- #
# Audio capture (WASAPI loopback)
# --------------------------------------------------------------------------- #

def list_output_devices():
    with pyaudio.PyAudio() as p:
        try:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("WASAPI not available on this system.")
            return
        default_idx = wasapi["defaultOutputDevice"]
        print("Output devices (capture with --output \"<part of name>\"):\n")
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev["hostApi"] != wasapi["index"] or dev["maxOutputChannels"] < 1:
                continue
            mark = "  <- default" if i == default_idx else ""
            print(f"  {dev['name']}{mark}")


def find_loopback_device(p, name_filter=None):
    """Return the loopback device matching the default (or named) output."""
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    target = p.get_device_info_by_index(wasapi["defaultOutputDevice"])

    if name_filter:
        target = None
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if (dev["hostApi"] == wasapi["index"]
                    and dev["maxOutputChannels"] > 0
                    and name_filter.lower() in dev["name"].lower()):
                target = dev
                break
        if target is None:
            raise SystemExit(
                f'No output device matching "{name_filter}". '
                "Run with --list-devices to see the options."
            )

    if target.get("isLoopbackDevice"):
        return target
    for lb in p.get_loopback_device_info_generator():
        if target["name"] in lb["name"]:
            return lb
    raise SystemExit(
        f'No loopback endpoint found for "{target["name"]}". '
        "Try a different --output device."
    )


class LoopbackCapture:
    """Streams system audio, downmixed and resampled to 16 kHz mono float32."""

    def __init__(self, name_filter=None, block_seconds=0.1):
        self.p = pyaudio.PyAudio()
        self.dev = find_loopback_device(self.p, name_filter)
        self.channels = max(1, int(self.dev["maxInputChannels"]))
        self.rate = int(self.dev["defaultSampleRate"])
        self.frames = int(self.rate * block_seconds)
        self.out_queue = queue.Queue()
        self.stream = None

    def _callback(self, in_data, frame_count, time_info, status):
        audio = np.frombuffer(in_data, dtype=np.float32)
        if self.channels > 1:
            audio = audio.reshape(-1, self.channels).mean(axis=1)
        if self.rate != TARGET_SR:
            n_out = int(len(audio) * TARGET_SR / self.rate)
            audio = np.interp(
                np.linspace(0.0, len(audio) - 1, n_out),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        self.out_queue.put(audio)
        return (None, pyaudio.paContinue)

    def start(self):
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.dev["index"],
            frames_per_buffer=self.frames,
            stream_callback=self._callback,
        )

    def stop(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()


# --------------------------------------------------------------------------- #
# Phrase segmentation (energy-based; loopback audio is digitally clean)
# --------------------------------------------------------------------------- #

class Segmenter:
    """Groups the audio stream into phrases, flushed on pauses in speech."""

    def __init__(self, audio_queue, phrase_queue, threshold=0.0015,
                 min_speech=0.5, trail_silence=0.55, max_buffer=8.0):
        self.audio_queue = audio_queue
        self.phrase_queue = phrase_queue
        self.threshold = threshold
        self.min_speech = min_speech
        self.trail_silence = trail_silence
        self.max_buffer = max_buffer
        self.stop_flag = threading.Event()

    def run(self):
        buf = []
        buf_len = 0.0
        speech_len = 0.0
        silence_run = 0.0

        while not self.stop_flag.is_set():
            try:
                block = self.audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            dur = len(block) / TARGET_SR
            rms = float(np.sqrt(np.mean(block ** 2))) if len(block) else 0.0

            buf.append(block)
            buf_len += dur
            if rms > self.threshold:
                speech_len += dur
                silence_run = 0.0
            else:
                silence_run += dur

            # Nothing but silence so far: keep a short rolling tail only.
            if speech_len == 0.0 and buf_len > 1.0:
                buf = buf[-5:]
                buf_len = sum(len(b) for b in buf) / TARGET_SR
                continue

            phrase_done = speech_len >= self.min_speech and silence_run >= self.trail_silence
            if phrase_done or buf_len >= self.max_buffer:
                audio = np.concatenate(buf)
                # Keep the transcriber real-time: drop the oldest backlog.
                while self.phrase_queue.qsize() > 2:
                    try:
                        self.phrase_queue.get_nowait()
                    except queue.Empty:
                        break
                self.phrase_queue.put(audio)
                buf, buf_len, speech_len, silence_run = [], 0.0, 0.0, 0.0


# --------------------------------------------------------------------------- #
# Transcription / translation
# --------------------------------------------------------------------------- #

def load_model(model_name, device):
    from faster_whisper import WhisperModel

    if device in ("auto", "cuda"):
        name = model_name or "large-v3"
        try:
            model = WhisperModel(name, device="cuda", compute_type="float16")
            print(f"* Model '{name}' loaded on GPU (cuda/float16)")
            return model
        except Exception as e:
            if device == "cuda":
                raise
            print(f"* GPU unavailable ({type(e).__name__}: {e}); falling back to CPU")
    name = model_name or "base"
    model = WhisperModel(name, device="cpu", compute_type="int8")
    print(f"* Model '{name}' loaded on CPU (int8)")
    return model


class Transcriber:
    def __init__(self, model, phrase_queue, text_queue, task="translate",
                 language=None, log_path=None):
        self.model = model
        self.phrase_queue = phrase_queue
        self.text_queue = text_queue
        self.task = task
        self.language = language
        self.log_path = log_path
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            try:
                audio = self.phrase_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            t0 = time.time()
            segments, info = self.model.transcribe(
                audio,
                task=self.task,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(threshold=0.35, min_silence_duration_ms=500),
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            if not text:
                continue
            took = time.time() - t0
            line = f"[{info.language}] {text}"
            self.text_queue.put(line)
            print(f"({took:.1f}s) {line}", flush=True)
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now():%H:%M:%S} {line}\n")


# --------------------------------------------------------------------------- #
# Subtitle overlay
# --------------------------------------------------------------------------- #

def run_overlay(text_queue, stop_flags, lines_kept=3):
    import tkinter as tk

    root = tk.Tk()
    root.title("Live Call Subtitles")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.85)
    root.configure(bg="black")

    width = min(1000, root.winfo_screenwidth() - 120)
    label = tk.Label(
        root,
        text="Listening…  (drag to move, right-click to close)",
        font=("Segoe UI", 15, "bold"),
        fg="white",
        bg="black",
        justify="left",
        anchor="w",
        wraplength=width - 30,
        padx=15,
        pady=10,
    )
    label.pack(fill="both", expand=True)

    # Bottom-center start position
    root.update_idletasks()
    x = (root.winfo_screenwidth() - width) // 2
    y = root.winfo_screenheight() - 190
    root.geometry(f"{width}x110+{x}+{y}")

    drag = {"x": 0, "y": 0}

    def press(e):
        drag["x"], drag["y"] = e.x, e.y

    def move(e):
        root.geometry(f"+{root.winfo_x() + e.x - drag['x']}+{root.winfo_y() + e.y - drag['y']}")

    def close(_e=None):
        for f in stop_flags:
            f.set()
        root.destroy()

    root.bind("<Button-1>", press)
    root.bind("<B1-Motion>", move)
    root.bind("<Button-3>", close)
    root.bind("<Escape>", close)

    def stay_on_top():
        # Re-assert z-order (other topmost windows can bury us). Only touch the
        # position if the bar is fully outside the whole multi-monitor desktop.
        root.attributes("-topmost", True)
        root.lift()
        import ctypes
        u32 = ctypes.windll.user32
        vx, vy = u32.GetSystemMetrics(76), u32.GetSystemMetrics(77)
        vw, vh = u32.GetSystemMetrics(78), u32.GetSystemMetrics(79)
        x, y = root.winfo_x(), root.winfo_y()
        w, h = root.winfo_width(), root.winfo_height()
        margin = 40  # keep at least this much of the bar reachable
        lost = (x + w < vx + margin or x > vx + vw - margin
                or y + h < vy + margin or y > vy + vh - margin)
        if lost:
            root.geometry(f"+{(root.winfo_screenwidth() - width) // 2}"
                          f"+{root.winfo_screenheight() - 190}")
        root.after(2000, stay_on_top)

    root.after(2000, stay_on_top)

    lines = deque(maxlen=lines_kept)

    def poll():
        updated = False
        while True:
            try:
                lines.append(text_queue.get_nowait())
                updated = True
            except queue.Empty:
                break
        if updated:
            label.config(text="\n".join(lines))
        root.after(150, poll)

    root.after(150, poll)
    root.mainloop()


# --------------------------------------------------------------------------- #

def _ensure_single_instance():
    import ctypes

    ctypes.windll.kernel32.CreateMutexW(None, False, "LiveCallSubtitlesMutex")
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        print("Live Call Subtitles is already running (right-click its bar to close it).")
        sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description="Live translation subtitles for voice calls.")
    ap.add_argument("--model", default=None,
                    help="Whisper model: tiny/base/small/medium/large-v3 (default: medium on GPU, base on CPU)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--task", default="translate", choices=["translate", "transcribe"],
                    help="translate -> English subtitles; transcribe -> original language")
    ap.add_argument("--language", default=None,
                    help="Force the source language (e.g. ru, es); default auto-detect")
    ap.add_argument("--output", default=None,
                    help="Capture a specific output device (substring of its name)")
    ap.add_argument("--threshold", type=float, default=0.0015,
                    help="Silence threshold (RMS); raise it if music keeps triggering")
    ap.add_argument("--no-overlay", action="store_true", help="Console output only")
    ap.add_argument("--log", default=None, help="Append subtitles to this file")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        list_output_devices()
        return

    _ensure_single_instance()

    print("* Loading model (first run downloads it, this can take a few minutes)…")
    model = load_model(args.model, args.device)

    capture = LoopbackCapture(name_filter=args.output)
    print(f"* Capturing: {capture.dev['name']}  ({capture.rate} Hz, {capture.channels} ch)")

    phrase_queue = queue.Queue()
    text_queue = queue.Queue()

    segmenter = Segmenter(capture.out_queue, phrase_queue, threshold=args.threshold)
    transcriber = Transcriber(model, phrase_queue, text_queue,
                              task=args.task, language=args.language, log_path=args.log)

    threading.Thread(target=segmenter.run, daemon=True).start()
    threading.Thread(target=transcriber.run, daemon=True).start()
    capture.start()
    print("* Live. Join your call — subtitles appear when the other side speaks.\n")

    try:
        if args.no_overlay:
            while True:
                time.sleep(0.5)
        else:
            run_overlay(text_queue, [segmenter.stop_flag, transcriber.stop_flag])
    except KeyboardInterrupt:
        pass
    finally:
        segmenter.stop_flag.set()
        transcriber.stop_flag.set()
        capture.stop()


if __name__ == "__main__":
    main()
