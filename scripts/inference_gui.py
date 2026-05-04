#!/usr/bin/env python3
"""
Tkinter GUI for handwriting synthesis: load checkpoint like scripts/synthesize.py,
sample with the same encode → sample_means → unnormalize path, then animate strokes
on a canvas (no change to model or sampler internals).
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SRC = os.path.join(_PROJECT_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

from PIL import Image, ImageDraw

from handwriting_synthesis.sampling import HandwritingSynthesizer
from handwriting_synthesis.utils.misc import create_strokes_png, get_strokes, split_into_components


def default_checkpoint_path() -> str:
    return os.path.join(_PROJECT_ROOT, "runs", "run_001", "checkpoints", "Epoch_60")


def sample_sequence(
    synthesizer: HandwritingSynthesizer,
    text: str,
    steps: int,
    stochastic: bool,
) -> torch.Tensor:
    c = synthesizer._encode_text(text)
    with torch.no_grad():
        seq = synthesizer.model.sample_means(
            context=c, steps=steps, stochastic=stochastic
        )
    seq = seq.cpu()
    return synthesizer._undo_normalization(seq)


def strokes_image_space(
    seq: torch.Tensor,
    shrink_factor: float,
    pad_x: float = 100.0,
    pad_y: float = 20.0,
) -> list[list[tuple[float, float]]]:
    """Strokes in the same coordinate space as create_strokes_png (before fitting to widget)."""
    x, y, eos = split_into_components(seq)
    if len(x) == 0:
        return []
    x = np.asarray(x, dtype=np.float64) / shrink_factor
    y = np.asarray(y, dtype=np.float64) / shrink_factor
    x_off = x - np.floor(np.nanmin(x)) + pad_x
    y_off = y - np.floor(np.nanmin(y)) + pad_y
    strokes: list[list[tuple[float, float]]] = []
    for stroke in get_strokes(
        torch.tensor(x_off, dtype=torch.float32),
        torch.tensor(y_off, dtype=torch.float32),
        eos,
    ):
        strokes.append([(float(px), float(py)) for px, py in stroke])
    return strokes


def wrap_text_to_lines(text: str, max_chars_per_line: int) -> list[str]:
    """
    Break text into lines for separate synthesis — only at word boundaries.
    A word is never split across two lines; an oversized word occupies its own line.
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    lines_out: list[str] = []
    for paragraph in text.split("\n"):
        p = paragraph.strip()
        if not p:
            continue
        words = p.split()
        chunk: list[str] = []
        cur_len = 0
        for w in words:
            if len(w) > max_chars_per_line:
                if chunk:
                    lines_out.append(" ".join(chunk))
                    chunk = []
                    cur_len = 0
                lines_out.append(w)
                continue
            add = len(w) if not chunk else 1 + len(w)
            if cur_len + add > max_chars_per_line and chunk:
                lines_out.append(" ".join(chunk))
                chunk = [w]
                cur_len = len(w)
            else:
                chunk.append(w)
                cur_len += add
        if chunk:
            lines_out.append(" ".join(chunk))
    # Two trailing spaces per segment (tokenizer / line cue, same idea as training sentinels).
    _suffix = "  "
    if not lines_out:
        return [text + _suffix] if text else []
    return [ln.rstrip() + _suffix for ln in lines_out]


def stack_stroke_blocks_vertical(
    blocks: list[list[list[tuple[float, float]]]],
    gap: float,
) -> list[list[tuple[float, float]]]:
    """Place each block under the previous (normalize each block's y to start at current cursor)."""
    merged: list[list[tuple[float, float]]] = []
    y_cursor = 0.0
    for strokes in blocks:
        if not strokes:
            continue
        ys = [py for s in strokes for _, py in s]
        if not ys:
            continue
        ymin, ymax = min(ys), max(ys)
        for stroke in strokes:
            merged.append([(px, py - ymin + y_cursor) for px, py in stroke])
        y_cursor += (ymax - ymin) + gap
    return merged


def layout_strokes_for_scroll(
    strokes: list[list[tuple[float, float]]],
    margin: float = 40.0,
    min_w: int = 400,
    min_h: int = 300,
) -> tuple[list[list[tuple[float, float]]], int, int]:
    """
    Shift strokes to origin + margin for scrollable canvas (no scale-to-fit).
    Returns strokes and content size (width, height) for scrollregion.
    """
    if not strokes:
        return [], min_w, min_h
    xs = [px for s in strokes for px, _ in s]
    ys = [py for s in strokes for _, py in s]
    if not xs:
        return [], min_w, min_h
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    mapped: list[list[tuple[float, float]]] = []
    for stroke in strokes:
        mapped.append(
            [(px - min_x + margin, py - min_y + margin) for px, py in stroke]
        )
    cw = int(max(max_x - min_x + 2 * margin, min_w))
    ch = int(max(max_y - min_y + 2 * margin, min_h))
    return mapped, cw, ch


class InferenceApp(tk.Tk):
    # Defaults requested (still apply when advanced panel is hidden)
    _DEF_BIAS = "0.9"
    _DEF_STEPS = "2000"
    _DEF_DELAY = "12"
    _DEF_LINE_W = "12"
    _DEF_SHRINK = "5"
    _DEF_TRIALS = "1"
    _DEF_WRAP = "44"
    _DEF_LINE_GAP = "28"
    _DEF_FG = "#0f172a"
    _DEF_BG = "#f8fafc"

    def __init__(self, model_path: str, device: torch.device) -> None:
        super().__init__()
        self.title("Handwriting synthesis")
        self.geometry("1000x720")
        self.minsize(780, 560)
        self._device = device
        self._model_dir = os.path.abspath(model_path)
        self._synth: HandwritingSynthesizer | None = None
        self._loaded_bias: float | None = None
        self._result_q: queue.Queue = queue.Queue()
        self._last_seq: torch.Tensor | None = None
        self._last_seqs: list[torch.Tensor] | None = None
        self._anim_after_id: str | None = None
        self._worker: threading.Thread | None = None
        self._run_id = 0
        self._settings_visible = tk.BooleanVar(value=False)
        self._await_worker_release = False

        self._setup_style()
        self._build_ui()
        self._poll_queue()
        self.after(120, self._initial_load)

    def _setup_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg = "#eef2f6"
        card = "#ffffff"
        accent = "#2563eb"
        fg = "#0f172a"
        self.configure(bg=bg)
        style.configure(".", background=bg, foreground=fg, font=("Segoe UI", 10))
        if sys.platform.startswith("linux"):
            style.configure(".", font=("Ubuntu", 10))
        style.configure("Card.TFrame", background=card, relief="flat")
        style.configure("Header.TFrame", background="#1e293b")
        style.configure("Header.TLabel", background="#1e293b", foreground="#f8fafc", font=("Segoe UI", 11, "bold"))
        style.configure("Title.TLabel", background="#1e293b", foreground="#94a3b8", font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.map(
            "Accent.TButton",
            background=[("active", "#1d4ed8"), ("!disabled", accent)],
            foreground=[("!disabled", "#ffffff")],
        )
        style.configure("TEntry", fieldbackground="#ffffff", padding=4)
        style.configure("TLabelframe", background=card, foreground=fg)
        style.configure("TLabelframe.Label", background=card, foreground="#64748b", font=("Segoe UI", 9))
        style.configure("TCheckbutton", background=card)
        style.configure("TRadiobutton", background=card)

    def _build_ui(self) -> None:
        header = ttk.Frame(self, style="Header.TFrame", padding=(16, 12))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Handwriting synthesis", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Collapsible settings beside the text field · scroll the canvas to pan the drawing",
            style="Title.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        outer = ttk.Frame(self, padding=(16, 12))
        outer.pack(fill=tk.BOTH, expand=True)

        card = ttk.Frame(outer, style="Card.TFrame", padding=14)
        card.pack(fill=tk.BOTH, expand=True)

        top_row = ttk.Frame(card, style="Card.TFrame")
        top_row.pack(fill=tk.BOTH, expand=False, pady=(0, 8))
        top_row.columnconfigure(0, weight=1)

        text_col = ttk.Frame(top_row, style="Card.TFrame")
        text_col.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 8))

        ttk.Label(text_col, text="Text", font=("Segoe UI", 9), foreground="#64748b").pack(anchor=tk.W)
        self.text_box = tk.Text(
            text_col,
            height=5,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            bg="#ffffff",
            fg="#0f172a",
            insertbackground="#2563eb",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
            highlightcolor="#2563eb",
            padx=10,
            pady=8,
        )
        self.text_box.pack(fill=tk.BOTH, expand=True)
        self.text_box.insert("1.0", "The quick brown fox jumps over the lazy dog.")

        self._settings_col = ttk.Frame(top_row, style="Card.TFrame", width=300)
        self._settings_col.grid(row=0, column=1, sticky=tk.NSEW)
        self._settings_col.grid_propagate(False)

        self._settings_toggle_btn = ttk.Button(
            self._settings_col,
            text="▼ Settings",
            command=self._toggle_settings_panel,
        )
        self._settings_toggle_btn.pack(anchor=tk.NE, pady=(0, 6))

        self.settings_inner = ttk.LabelFrame(self._settings_col, text="Sampling & layout", padding=8)
        self._populate_settings_form(self.settings_inner)
        self._settings_col.configure(width=112)

        colors = ttk.Frame(card)
        colors.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(colors, text="Ink color", foreground="#64748b").pack(side=tk.LEFT)
        self.line_color_var = tk.StringVar(value=self._DEF_FG)
        self._line_color_btn = tk.Button(
            colors,
            text="  ",
            width=3,
            relief="flat",
            cursor="hand2",
            command=lambda: self._pick_color(self.line_color_var, self._line_color_btn, True),
        )
        self._apply_btn_color(self._line_color_btn, self.line_color_var.get())
        self._line_color_btn.pack(side=tk.LEFT, padx=(6, 20))

        ttk.Label(colors, text="Background", foreground="#64748b").pack(side=tk.LEFT)
        self.bg_color_var = tk.StringVar(value=self._DEF_BG)
        self._bg_color_btn = tk.Button(
            colors,
            text="  ",
            width=3,
            relief="flat",
            cursor="hand2",
            command=lambda: self._pick_color(self.bg_color_var, self._bg_color_btn, False),
        )
        self._apply_btn_color(self._bg_color_btn, self.bg_color_var.get())
        self._bg_color_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.bg_color_var.trace_add("write", lambda *_: self._sync_canvas_bg())

        btn_row = ttk.Frame(card)
        btn_row.pack(fill=tk.X, pady=(4, 8))
        self.gen_btn = ttk.Button(btn_row, text="Generate", style="Accent.TButton", command=self._on_generate)
        self.gen_btn.pack(side=tk.LEFT, padx=(0, 8), ipadx=12, ipady=4)
        self.stop_btn = ttk.Button(btn_row, text="Stop", command=self._on_stop)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8), ipadx=10, ipady=4)
        ttk.Button(btn_row, text="Clear canvas", command=self._clear_canvas).pack(side=tk.LEFT, padx=(0, 8), ipadx=8, ipady=4)
        ttk.Button(btn_row, text="Save PNG…", command=self._save_png).pack(side=tk.LEFT, ipadx=8, ipady=4)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(btn_row, textvariable=self.status_var, foreground="#64748b").pack(side=tk.LEFT, padx=(16, 0))

        canvas_outer = ttk.Frame(card, style="Card.TFrame")
        canvas_outer.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        canvas_outer.rowconfigure(0, weight=1)
        canvas_outer.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_outer,
            background=self.bg_color_var.get(),
            highlightthickness=0,
            bd=0,
        )
        vsb = ttk.Scrollbar(canvas_outer, orient=tk.VERTICAL, command=self.canvas.yview)
        hsb = ttk.Scrollbar(canvas_outer, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        vsb.grid(row=0, column=1, sticky=tk.NS)
        hsb.grid(row=1, column=0, sticky=tk.EW)

        self._bind_canvas_scroll()

        foot = ttk.Frame(card, style="Card.TFrame")
        foot.pack(fill=tk.X, pady=(8, 0))
        self.anim_mode = tk.StringVar(value="point")
        ttk.Label(foot, text="Animation:", foreground="#64748b").pack(side=tk.LEFT)
        ttk.Radiobutton(foot, text="Point by point", variable=self.anim_mode, value="point").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Radiobutton(foot, text="Stroke by stroke", variable=self.anim_mode, value="stroke").pack(side=tk.LEFT, padx=(8, 0))

    def _populate_settings_form(self, parent: ttk.LabelFrame) -> None:
        self.path_var = tk.StringVar(value=self._model_dir)
        self.bias_var = tk.StringVar(value=self._DEF_BIAS)
        self.steps_var = tk.StringVar(value=self._DEF_STEPS)
        self.delay_var = tk.StringVar(value=self._DEF_DELAY)
        self.width_var = tk.StringVar(value=self._DEF_LINE_W)
        self.shrink_var = tk.StringVar(value=self._DEF_SHRINK)
        self.trials_var = tk.StringVar(value=self._DEF_TRIALS)
        self.wrap_chars_var = tk.StringVar(value=self._DEF_WRAP)
        self.line_gap_var = tk.StringVar(value=self._DEF_LINE_GAP)
        self.stochastic_var = tk.BooleanVar(value=True)

        r = 0
        ttk.Label(parent, text="Checkpoint", foreground="#64748b").grid(row=r, column=0, sticky=tk.W, columnspan=2)
        r += 1
        ttk.Entry(parent, textvariable=self.path_var, width=26).grid(row=r, column=0, columnspan=2, sticky=tk.EW, pady=(0, 4))
        ttk.Button(parent, text="Browse…", command=self._browse_checkpoint).grid(row=r, column=2, padx=(4, 0), pady=(0, 4))
        r += 1

        ttk.Label(parent, text="Bias").grid(row=r, column=0, sticky=tk.W)
        ttk.Entry(parent, textvariable=self.bias_var, width=8).grid(row=r, column=1, sticky=tk.W, padx=(0, 4))
        ttk.Label(parent, text="Steps").grid(row=r, column=2, sticky=tk.W)
        ttk.Entry(parent, textvariable=self.steps_var, width=8).grid(row=r, column=3, sticky=tk.W)
        r += 1
        ttk.Label(parent, text="Delay ms").grid(row=r, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Entry(parent, textvariable=self.delay_var, width=8).grid(row=r, column=1, sticky=tk.W, pady=(6, 0))
        ttk.Label(parent, text="Line width").grid(row=r, column=2, sticky=tk.W, pady=(6, 0))
        ttk.Entry(parent, textvariable=self.width_var, width=8).grid(row=r, column=3, sticky=tk.W, pady=(6, 0))
        r += 1
        ttk.Label(parent, text="Scale").grid(row=r, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Entry(parent, textvariable=self.shrink_var, width=8).grid(row=r, column=1, sticky=tk.W, pady=(6, 0))
        ttk.Label(parent, text="Trials").grid(row=r, column=2, sticky=tk.W, pady=(6, 0))
        ttk.Entry(parent, textvariable=self.trials_var, width=8).grid(row=r, column=3, sticky=tk.W, pady=(6, 0))
        r += 1
        ttk.Label(parent, text="Max chars/line", foreground="#64748b").grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        ttk.Entry(parent, textvariable=self.wrap_chars_var, width=8).grid(row=r, column=2, sticky=tk.W, pady=(6, 0))
        r += 1
        ttk.Label(parent, text="Line gap (px)", foreground="#64748b").grid(row=r, column=0, columnspan=2, sticky=tk.W)
        ttk.Entry(parent, textvariable=self.line_gap_var, width=8).grid(row=r, column=2, sticky=tk.W)
        r += 1
        ttk.Checkbutton(parent, text="Stochastic", variable=self.stochastic_var).grid(
            row=r, column=0, columnspan=4, sticky=tk.W, pady=(8, 0)
        )
        r += 1
        ttk.Label(
            parent,
            text="Wrap breaks at word boundaries only.",
            font=("Segoe UI", 8),
            foreground="#94a3b8",
        ).grid(row=r, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))
        parent.columnconfigure(1, weight=1)

    def _toggle_settings_panel(self) -> None:
        if self._settings_visible.get():
            self.settings_inner.pack_forget()
            self._settings_visible.set(False)
            self._settings_toggle_btn.configure(text="▼ Settings")
            self._settings_col.configure(width=112)
        else:
            self._settings_col.configure(width=300)
            self.settings_inner.pack(fill=tk.BOTH, expand=True)
            self._settings_visible.set(True)
            self._settings_toggle_btn.configure(text="▲ Settings")

    def _bind_canvas_scroll(self) -> None:
        def wheel_y(event: tk.Event) -> str:
            if event.delta:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif getattr(event, "num", None) == 4:
                self.canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                self.canvas.yview_scroll(3, "units")
            return "break"

        def wheel_x(event: tk.Event) -> str:
            if event.delta:
                self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        self.canvas.bind("<MouseWheel>", wheel_y)
        self.canvas.bind("<Button-4>", wheel_y)
        self.canvas.bind("<Button-5>", wheel_y)
        self.canvas.bind("<Shift-MouseWheel>", wheel_x)

    def _set_generate_idle(self, status: str | None = None) -> None:
        self.gen_btn.configure(state=tk.NORMAL)
        if status is not None:
            self.status_var.set(status)

    def _set_generate_busy(self) -> None:
        self.gen_btn.configure(state=tk.DISABLED)

    def _on_stop(self) -> None:
        self._run_id += 1
        self._cancel_animation()
        if self._worker is not None and self._worker.is_alive():
            self._await_worker_release = True
            self.status_var.set("Stopping…")
            self.gen_btn.configure(state=tk.DISABLED)
        else:
            self._set_generate_idle("Stopped.")

    def _pick_color(self, var: tk.StringVar, btn: tk.Button, is_line: bool) -> None:
        c = colorchooser.askcolor(color=var.get(), title="Choose color", parent=self)
        if c and c[1]:
            var.set(c[1])
            self._apply_btn_color(btn, c[1])
            if not is_line:
                self._sync_canvas_bg()

    def _apply_btn_color(self, btn: tk.Button, hex_color: str) -> None:
        try:
            btn.configure(bg=hex_color, activebackground=hex_color)
        except tk.TclError:
            btn.configure(bg="#888888")

    def _sync_canvas_bg(self) -> None:
        try:
            self.canvas.configure(bg=self.bg_color_var.get())
        except tk.TclError:
            pass

    def _browse_checkpoint(self) -> None:
        path = filedialog.askdirectory(
            initialdir=self.path_var.get() or _PROJECT_ROOT,
            title="Checkpoint (model.pt + meta.json)",
        )
        if path:
            self.path_var.set(path)

    def _parse_float(self, var: tk.StringVar, name: str, default: float) -> float:
        try:
            return float(var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid input", f"{name} must be a number.")
            raise ValueError from None

    def _parse_int(self, var: tk.StringVar, name: str, default: int) -> int:
        try:
            return int(var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid input", f"{name} must be an integer.")
            raise ValueError from None

    def _cancel_animation(self) -> None:
        if self._anim_after_id is not None:
            self.after_cancel(self._anim_after_id)
            self._anim_after_id = None

    def _clear_canvas(self) -> None:
        self._cancel_animation()
        self.canvas.delete("all")
        self._sync_canvas_bg()
        self.canvas.configure(scrollregion=(0, 0, 1, 1))

    def _initial_load(self) -> None:
        self.status_var.set("Loading model…")
        try:
            self._ensure_synthesizer()
            self.status_var.set("Loaded. Click Generate.")
        except Exception as e:
            self.status_var.set("Load failed.")
            messagebox.showerror("Load error", str(e))

    def _ensure_synthesizer(self) -> None:
        path = self.path_var.get().strip()
        bias = self._parse_float(self.bias_var, "Bias", 0.0)
        self._load_synthesizer_threadsafe(path, bias)

    def _load_synthesizer_threadsafe(self, path: str, bias: float) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Not a directory: {path}")
        meta = os.path.join(path, "meta.json")
        mdl = os.path.join(path, "model.pt")
        if not os.path.isfile(meta) or not os.path.isfile(mdl):
            raise FileNotFoundError(
                f"Checkpoint folder must contain meta.json and model.pt: {path}"
            )
        need_reload = (
            self._synth is None
            or os.path.abspath(path) != os.path.abspath(self._model_dir)
            or self._loaded_bias != bias
        )
        self._model_dir = path
        if need_reload:
            self._synth = HandwritingSynthesizer.load(path, self._device, bias)
            self._loaded_bias = bias

    def _get_text(self) -> str:
        return self.text_box.get("1.0", "end-1c").strip()

    def _on_generate(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "Wait for the current run to finish, or press Stop.")
            return
        try:
            steps = max(1, self._parse_int(self.steps_var, "Max steps", 2000))
            delay = max(0, self._parse_int(self.delay_var, "Delay (ms)", 12))
            _ = self._parse_float(self.shrink_var, "Scale", 5.0)
            trials = max(1, self._parse_int(self.trials_var, "Trials", 1))
            wrap_c = max(12, self._parse_int(self.wrap_chars_var, "Wrap chars", 44))
            line_gap = max(0.0, float(self.line_gap_var.get().strip()))
        except ValueError:
            messagebox.showerror("Invalid input", "Check numeric fields in the settings panel.")
            return

        text = self._get_text()
        if not text:
            messagebox.showwarning("Empty text", "Enter some text first.")
            return

        lines = wrap_text_to_lines(text, wrap_c)
        if not lines:
            messagebox.showwarning("Empty text", "Enter some text first.")
            return

        self._await_worker_release = False
        self._run_id += 1
        rid = self._run_id
        self._set_generate_busy()
        self._cancel_animation()
        self.canvas.delete("all")
        self._sync_canvas_bg()
        self.canvas.configure(scrollregion=(0, 0, 1, 1))
        self.status_var.set("Generating…")

        ckpt_path = self.path_var.get().strip()
        try:
            bias = float(self.bias_var.get().strip())
        except ValueError:
            self._set_generate_idle("Ready.")
            messagebox.showerror("Invalid input", "Bias must be a number.")
            return
        steps_i = steps
        stochastic = bool(self.stochastic_var.get())

        def work() -> None:
            try:
                self._load_synthesizer_threadsafe(ckpt_path, bias)
                synth = self._synth
                assert synth is not None
                all_seqs: list[torch.Tensor] = []
                for line in lines:
                    if rid != self._run_id:
                        return
                    last: torch.Tensor | None = None
                    for _ in range(trials):
                        if rid != self._run_id:
                            return
                        last = sample_sequence(synth, line, steps_i, stochastic)
                    assert last is not None
                    all_seqs.append(last)
                if rid == self._run_id:
                    self._result_q.put(("sample", all_seqs, delay, line_gap, rid))
            except Exception as e:
                if rid == self._run_id:
                    self._result_q.put(("error", str(e)))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _poll_queue(self) -> None:
        if self._await_worker_release and (
            self._worker is None or not self._worker.is_alive()
        ):
            self._await_worker_release = False
            self._set_generate_idle("Stopped.")

        try:
            while True:
                msg = self._result_q.get_nowait()
                if msg[0] == "error":
                    self._await_worker_release = False
                    self._set_generate_idle("Error.")
                    messagebox.showerror("Generation failed", msg[1])
                elif msg[0] == "sample":
                    _, seqs, delay, line_gap, rid_msg = msg
                    if rid_msg != self._run_id:
                        continue
                    self._await_worker_release = False
                    self._last_seqs = seqs
                    self._last_seq = seqs[-1] if seqs else None
                    self._set_generate_busy()
                    self.status_var.set(
                        "Drawing…" if self.anim_mode.get() == "stroke" else "Typing…"
                    )
                    try:
                        shrink = float(self.shrink_var.get())
                        if shrink <= 0:
                            shrink = 1.0
                    except ValueError:
                        shrink = 5.0
                    blocks = [strokes_image_space(s, shrink) for s in seqs]
                    stacked = stack_stroke_blocks_vertical(blocks, float(line_gap))
                    strokes, cw, ch = layout_strokes_for_scroll(stacked)
                    self.canvas.delete("all")
                    bg = self.bg_color_var.get()
                    self.canvas.create_rectangle(0, 0, cw, ch, fill=bg, outline="", tags="bg")
                    self.canvas.configure(scrollregion=(0, 0, cw, ch))
                    self.canvas.xview_moveto(0)
                    self.canvas.yview_moveto(0)
                    try:
                        lw = max(1, int(float(self.width_var.get())))
                    except ValueError:
                        lw = 12
                    fill = self.line_color_var.get()
                    self._start_animation(strokes, lw, delay, fill, rid_msg)
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)

    def _start_animation(
        self,
        strokes: list[list[tuple[float, float]]],
        line_width: int,
        delay_ms: int,
        fill: str,
        session_rid: int,
    ) -> None:
        self._cancel_animation()
        if not strokes:
            self._set_generate_idle("Nothing to draw.")
            return
        mode = self.anim_mode.get()
        state = {
            "strokes": strokes,
            "si": 0,
            "pi": 0,
            "line_width": line_width,
            "delay": delay_ms,
            "mode": mode,
            "fill": fill,
            "session": session_rid,
        }
        self._anim_step(state)

    def _anim_step(self, state: dict) -> None:
        if state["session"] != self._run_id:
            self._set_generate_idle("Stopped.")
            self._anim_after_id = None
            return

        strokes: list[list[tuple[float, float]]] = state["strokes"]
        si: int = state["si"]
        pi: int = state["pi"]
        lw: int = state["line_width"]
        delay: int = state["delay"]
        mode: str = state["mode"]
        fill: str = state["fill"]

        if si >= len(strokes):
            if state["session"] == self._run_id:
                self._set_generate_idle("Done.")
            self._anim_after_id = None
            return

        stroke = strokes[si]

        if mode == "stroke":
            if len(stroke) >= 2:
                flat: list[float] = []
                for x, y in stroke:
                    flat.extend([x, y])
                self.canvas.create_line(
                    *flat,
                    fill=fill,
                    width=lw,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                    smooth=False,
                )
            state["si"] = si + 1
            self._anim_after_id = self.after(delay, lambda: self._anim_step(state))
            return

        if pi == 0:
            state["pi"] = 1
            if len(stroke) < 2:
                state["si"] = si + 1
                state["pi"] = 0
                self._anim_after_id = self.after(0, lambda: self._anim_step(state))
                return
            self._anim_after_id = self.after(delay, lambda: self._anim_step(state))
            return

        if pi < len(stroke):
            x0, y0 = stroke[pi - 1]
            x1, y1 = stroke[pi]
            self.canvas.create_line(
                x0,
                y0,
                x1,
                y1,
                fill=fill,
                width=lw,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
            )
            state["pi"] = pi + 1
            self._anim_after_id = self.after(delay, lambda: self._anim_step(state))
            return

        state["si"] = si + 1
        state["pi"] = 0
        self._anim_after_id = self.after(0, lambda: self._anim_step(state))

    def _hex_to_rgb(self, h: str) -> tuple[int, int, int]:
        h = h.strip().lstrip("#")
        if len(h) == 6:
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return 15, 23, 42

    def _save_png(self) -> None:
        if self._last_seq is None:
            messagebox.showinfo("Save", "Generate handwriting first.")
            return
        try:
            thickness = max(1, int(float(self.width_var.get())))
        except ValueError:
            thickness = 12
        try:
            shrink = float(self.shrink_var.get())
            if shrink <= 0:
                shrink = 5.0
        except ValueError:
            shrink = 5.0
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All", "*")],
        )
        if not path:
            return
        try:
            line_rgb = self._hex_to_rgb(self.line_color_var.get())
            bg_rgb = self._hex_to_rgb(self.bg_color_var.get())

            if self._last_seqs and len(self._last_seqs) > 1:
                blocks = [strokes_image_space(s, shrink) for s in self._last_seqs]
                try:
                    gap = float(self.line_gap_var.get())
                except ValueError:
                    gap = 28.0
                stacked = stack_stroke_blocks_vertical(blocks, gap)
                xs = [px for s in stacked for px, _ in s]
                ys = [py for s in stacked for _, py in s]
                if not xs:
                    raise ValueError("empty")
                min_x, min_y = min(xs), min(ys)
                max_x, max_y = max(xs), max(ys)
                w = int(max_x - min_x) + 200
                h = int(max_y - min_y) + 80
                w = max(w, 100)
                h = max(h, 80)
                im = Image.new("RGB", (w, h), bg_rgb)
                dr = ImageDraw.Draw(im)
                for stroke in stacked:
                    if len(stroke) < 2:
                        continue
                    adj = [(px - min_x + 50, py - min_y + 40) for px, py in stroke]
                    dr.line(adj, fill=line_rgb, width=thickness)
                im.save(path)
            else:
                im_l = create_strokes_png(
                    self._last_seq,
                    lines=True,
                    shrink_factor=shrink,
                    suppress_errors=False,
                    thickness=thickness,
                )
                if im_l is None:
                    messagebox.showerror("Save", "Could not render image.")
                    return
                arr = np.asarray(im_l, dtype=np.float32)
                ink = (255.0 - arr) / 255.0
                br, bgc, bb_bg = float(bg_rgb[0]), float(bg_rgb[1]), float(bg_rgb[2])
                lr, lg, lb = float(line_rgb[0]), float(line_rgb[1]), float(line_rgb[2])
                rr = br + (lr - br) * ink
                gg = bgc + (lg - bgc) * ink
                bb_ch = bb_bg + (lb - bb_bg) * ink
                rgb = Image.fromarray(
                    np.stack([rr, gg, bb_ch], axis=-1).astype(np.uint8), mode="RGB"
                )
                rgb.save(path)
            self.status_var.set(f"Saved {path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="Animated handwriting synthesis GUI")
    parser.add_argument(
        "model_path",
        nargs="?",
        default=default_checkpoint_path(),
        help="Directory with model.pt and meta.json",
    )
    parser.add_argument("--device", type=str, default="cpu", help="torch device")
    args = parser.parse_args()
    device = torch.device(args.device)
    app = InferenceApp(args.model_path, device)
    app.mainloop()


if __name__ == "__main__":
    main()
