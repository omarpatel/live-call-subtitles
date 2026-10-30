"""Unit tests for the audio pipeline that need no audio device or Whisper model."""

import queue
import threading

import numpy as np
import pytest

import app

BLOCK = int(app.TARGET_SR * 0.1)  # 100 ms blocks, as produced by LoopbackCapture


def speech(seconds, amplitude=0.1):
    t = np.arange(int(app.TARGET_SR * seconds)) / app.TARGET_SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def blocks(signal):
    return [signal[i:i + BLOCK] for i in range(0, len(signal), BLOCK)]


def run_segmenter(audio_blocks, **kwargs):
    audio_q, phrase_q = queue.Queue(), queue.Queue()
    for b in audio_blocks:
        audio_q.put(b)
    seg = app.Segmenter(audio_q, phrase_q, **kwargs)
    worker = threading.Thread(target=seg.run, daemon=True)
    worker.start()
    deadline = threading.Event()
    while not audio_q.empty():
        deadline.wait(0.01)
    deadline.wait(0.5)  # let the worker finish the last block it dequeued
    seg.stop_flag.set()
    worker.join(timeout=2)
    phrases = []
    while not phrase_q.empty():
        phrases.append(phrase_q.get_nowait())
    return phrases


def test_phrase_is_flushed_after_trailing_silence():
    signal = np.concatenate([speech(1.0), np.zeros(int(app.TARGET_SR * 0.7), np.float32)])
    phrases = run_segmenter(blocks(signal))
    assert len(phrases) == 1
    assert len(phrases[0]) >= int(app.TARGET_SR * 1.5)


def test_silence_alone_produces_no_phrase():
    phrases = run_segmenter(blocks(np.zeros(app.TARGET_SR * 3, np.float32)))
    assert phrases == []


def test_long_speech_is_cut_at_max_buffer():
    phrases = run_segmenter(blocks(speech(5.0)), max_buffer=2.0)
    assert len(phrases) >= 2
    assert all(len(p) <= int(app.TARGET_SR * 2.2) for p in phrases)


def test_loopback_callback_downmixes_and_resamples():
    cap = app.LoopbackCapture.__new__(app.LoopbackCapture)
    cap.channels, cap.rate, cap.out_queue = 2, 48000, queue.Queue()

    frames = 4800  # 100 ms at 48 kHz
    stereo = np.empty((frames, 2), np.float32)
    stereo[:, 0], stereo[:, 1] = 0.2, 0.4
    cap._callback(stereo.tobytes(), frames, None, 0)

    mono = cap.out_queue.get_nowait()
    assert mono.dtype == np.float32
    assert len(mono) == BLOCK
    assert mono == pytest.approx(0.3)
