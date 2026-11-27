#!/usr/bin/env python3
"""Regression tests and latency micro-benchmark for bin/proxy.py.

Runs completely offline using a local mock Loom HTTP server (no GPU required).
Usage:
  python3 tools/test_proxy.py
  python3 tools/test_proxy.py --bench-baseline /path/to/old_proxy.py
"""
import argparse
import http.client
import importlib.util
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROXY_PATH = os.path.join(REPO, "bin", "proxy.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def load_proxy_module(env_overrides=None):
    saved = dict(os.environ)
    try:
        # Ensure repo proxy.env is bypassed during unit tests.
        os.environ["LOOM_PROXY_ENV"] = "/nonexistent-test-proxy.env"
        if env_overrides:
            os.environ.update(env_overrides)
        spec = importlib.util.spec_from_file_location("loom_proxy_test", PROXY_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.environ.clear()
        os.environ.update(saved)


class MockLoomState:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.requests = []
            self.mode = "ok_json"
            self.input_tokens = 12000
            self.input_tokens_calls = 0
            self.ckpt_pos = 11500
            self.vision_delay = 0.0
            self.stream_delay = 0.0


def start_mock_loom(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def _send(self, code, payload, ctype="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            j = json.loads(raw) if raw else {}
            with state.lock:
                state.requests.append((self.path, dict(self.headers), j))
                mode = state.mode
                itok = state.input_tokens
                cpos = state.ckpt_pos
                vdelay = state.vision_delay
                sdelay = state.stream_delay

            if self.path == "/v1/chat/completions/input_tokens":
                with state.lock:
                    state.input_tokens_calls += 1
                self._send(200, {"input_tokens": itok})
                return

            if self.path == "/tokenize":
                txt = j.get("content") or ""
                self._send(200, {"tokens": list(range(max(1, len(txt) // 4)))})
                return

            if self.path.startswith("/v1/checkpoints/"):
                self._send(201, {"position": cpos})
                return

            if self.path == "/v1/chat/completions":
                # Seed request from SysCheckpoints
                if self.headers.get("X-Loom-Session", "").startswith("sysp-"):
                    if mode == "seed_503":
                        self._send(503, {"error": "temporary"})
                        return
                    self._send(200, {
                        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok", "reasoning_content": "done"}}],
                        "usage": {"prompt_tokens": itok, "completion_tokens": 2},
                    })
                    return

                # Side vision request from VisionOffload
                if j.get("reasoning_effort") == "none" and j.get("temperature") == 0.0:
                    if vdelay:
                        time.sleep(vdelay)
                    self._send(200, {
                        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "A blue button labeled Save."}}],
                        "usage": {"prompt_tokens": 500, "completion_tokens": 8},
                    })
                    return

                if mode == "err_400":
                    self._send(400, {"error": {"message": "bad request"}})
                    return
                if mode == "err_503":
                    self._send(503, {"error": {"message": "service unavailable"}})
                    return
                if mode == "sse_alias_same_chunk":
                    events = [
                        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"batch_vision","arguments":"{\\"items\\":[{\\"image_ref\\":\\"a"}}]}}]}\n\n',
                        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":".png\\"}]}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10}}\n\n',
                        b"data: [DONE]\n\n",
                    ]
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for ev in events:
                        self.wfile.write(ev)
                        self.wfile.flush()
                    self.close_connection = True
                    return
                if mode == "sse_slow_reply":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(b'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n')
                    self.wfile.flush()
                    if sdelay:
                        time.sleep(sdelay)
                    self.wfile.write(b'data: {"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":200,"completion_tokens":2}}\n\n')
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    self.close_connection = True
                    return

                self._send(200, {
                    "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hello", "reasoning_content": "thought"}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0}},
                })
                return

            self._send(404, {"error": "not found"})

        def do_DELETE(self):
            self._send(204, b"")

    port = free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, port


def start_proxy_proc(proxy_path, bport, extra_env=None):
    pport = free_port()
    env = dict(os.environ)
    env["LOOM_PROXY_ENV"] = "/nonexistent-test-proxy.env"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        [sys.executable, "-u", proxy_path, str(pport), str(bport), "Inkling-2-Mini"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    lines = []

    def reader():
        for ln in proc.stdout:
            lines.append(ln)

    threading.Thread(target=reader, daemon=True).start()
    # Wait for listening socket
    t0 = time.time()
    while time.time() - t0 < 5:
        try:
            with socket.create_connection(("127.0.0.1", pport), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    return proc, pport, lines


def http_post(port, path, obj, timeout=15):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        data = json.dumps(obj).encode()
        c.request("POST", path, data, {"Content-Type": "application/json", "Content-Length": str(len(data))})
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


def stop_servers(proc, srv):
    proc.terminate()
    try:
        proc.wait(timeout=3)
    finally:
        if proc.stdout:
            proc.stdout.close()
        srv.shutdown()
        srv.server_close()


class ProxyRegressionTests(unittest.TestCase):
    def test_live_meter_no_thread_leak_on_error_returns(self):
        state = MockLoomState()
        srv, bport = start_mock_loom(state)
        proc, pport, logs = start_proxy_proc(PROXY_PATH, bport, {"LOOM_LIVE": "1", "LOOM_LIVE_SECS": "0.15"})
        try:
            for mode in ("err_400", "err_503"):
                state.mode = mode
                st, _ = http_post(pport, "/v1/chat/completions", {
                    "model": "local", "stream": True,
                    "messages": [{"role": "user", "content": "mention image_url in plain text"}],
                })
                self.assertIn(st, (400, 503))
            # Wait several LiveMeter tick intervals; zero "prefilling" lines should appear after error returns.
            time.sleep(0.6)
            prefill_ticks = [ln for ln in logs if "prefilling" in ln]
            self.assertEqual(prefill_ticks, [], f"Leaked LiveMeter ticks: {prefill_ticks}")
            # Also confirm LiveMeter._count_prompt was NOT falsely skipped by "image_url" in plain text!
            self.assertGreaterEqual(state.input_tokens_calls, 2)
        finally:
            stop_servers(proc, srv)

    def test_prefix_diff_and_echo_work_without_budget(self):
        state = MockLoomState()
        srv, bport = start_mock_loom(state)
        proc, pport, logs = start_proxy_proc(PROXY_PATH, bport, {"LOOM_TOKEN_BUDGET": "0", "LOOM_PREFIX_DIFF": "1"})
        try:
            msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u1"}]
            st, _ = http_post(pport, "/v1/chat/completions", {"model": "local", "messages": msgs})
            self.assertEqual(st, 200)
            # Second turn drops reasoning_content -> both PREFIX_DIFF and ECHO should fire even with budget=0!
            msgs2 = msgs + [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "u2"}]
            st, _ = http_post(pport, "/v1/chat/completions", {"model": "local", "messages": msgs2})
            self.assertEqual(st, 200)
            joined = "".join(logs)
            self.assertIn("prefix-diff conv", joined)
            self.assertIn("echo conv", joined)
            self.assertIn("reasoning dropped", joined)
        finally:
            stop_servers(proc, srv)

    def test_param_alias_nested_schema_and_same_chunk_finish_reason(self):
        state = MockLoomState()
        state.mode = "sse_alias_same_chunk"
        srv, bport = start_mock_loom(state)
        proc, pport, _ = start_proxy_proc(PROXY_PATH, bport, {"LOOM_LIVE": "0"})
        try:
            tools = [{
                "type": "function",
                "function": {
                    "name": "batch_vision",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"image_url": {"type": "string"}},
                                    "required": ["image_url"],
                                },
                            }
                        },
                    },
                },
            }]
            st, raw = http_post(pport, "/v1/chat/completions", {
                "model": "local", "stream": True, "tools": tools,
                "messages": [{"role": "user", "content": "check"}],
            })
            self.assertEqual(st, 200)
            # Verify backend saw image_ref (not image_url) inside nested schema
            _, _, sent = state.requests[-1]
            self.assertNotIn("image_url", json.dumps(sent["tools"]))
            self.assertIn("image_ref", json.dumps(sent["tools"]))
            # Verify SSE final chunk has NO duplicate tool_calls entry and restored image_url
            events = [ln[5:].strip() for ln in raw.decode().splitlines() if ln.startswith("data:") and "[DONE]" not in ln]
            last = json.loads(events[-1])
            tcs = last["choices"][0]["delta"]["tool_calls"]
            self.assertEqual(len(tcs), 1, f"Duplicate tool_calls entries emitted: {tcs}")
            args = json.loads(tcs[0]["function"]["arguments"])
            self.assertEqual(args, {"items": [{"image_url": "a.png"}]})
        finally:
            stop_servers(proc, srv)

    def test_sys_checkpoints_pinned_no_double_decrement_and_render_keys(self):
        mod = load_proxy_module({"LOOM_TOKEN_BUDGET": "100000", "LOOM_SYS_CHECKPOINT": "1", "LOOM_CHECKPOINT_CAPACITY": "1"})
        ckpt = mod.SysCheckpoints(cap=1, min_tokens=4096)

        # Render keys check: response_format and parallel_tool_calls must produce distinct keys
        j1 = {"messages": [{"role": "system", "content": "S"}], "response_format": {"type": "json_object"}}
        j2 = {"messages": [{"role": "system", "content": "S"}], "response_format": {"type": "text"}}
        self.assertNotEqual(ckpt._key(j1), ckpt._key(j2))

        # Simulate concurrent eviction during _create() + drop()
        ckpt.ck["oldk"] = {"name": "sysp-old-1", "seed": [], "position": 10000}
        mod.BUDGET.adjust_pinned(10000)

        def fake_call(method, path, body=None, hdr=None, timeout=600):
            if path == "/v1/chat/completions/input_tokens":
                return 200, json.dumps({"input_tokens": 8000}).encode()
            if path == "/v1/chat/completions":
                return 200, json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}).encode()
            if method == "DELETE" and path == "/v1/checkpoints/sysp-old-1":
                # Concurrent drop() removes oldk before _create re-acquires self.lock!
                ckpt.drop("oldk", "conv1", "404 gone")
                return 204, b""
            if method == "POST" and path.startswith("/v1/checkpoints/"):
                return 201, json.dumps({"position": 8500}).encode()
            return 404, b""

        ckpt._call = fake_call
        e, perm = ckpt._create("newk", {"messages": [{"role": "system", "content": "S2"}]})
        self.assertIsNotNone(e)
        self.assertFalse(perm)
        self.assertEqual(mod.BUDGET.pinned, 8500, f"BUDGET.pinned drifted: {mod.BUDGET.pinned}")

        # Transient 503 during seed must NOT permanently blacklist the key
        ckpt._call = lambda m, p, b=None, h=None, timeout=600: (
            (200, b'{"input_tokens": 8000}') if p.endswith("input_tokens") else (503, b"busy")
        )
        e2, perm2 = ckpt._create("k503", {"messages": [{"role": "system", "content": "S3"}]})
        self.assertIsNone(e2)
        self.assertFalse(perm2)

    def test_vision_offload_uses_budget_guard(self):
        mod = load_proxy_module({"LOOM_TOKEN_BUDGET": "20000", "LOOM_MAX_SESSIONS": "2", "LOOM_VISION_OFFLOAD": "1"})
        admits = []
        orig_admit = mod.BUDGET.admit
        orig_release = mod.BUDGET.release

        def track_admit(cost, label):
            t = orig_admit(cost, label)
            admits.append((label, cost, t))
            return t

        released = []

        def track_release(t):
            released.append(t)
            return orig_release(t)

        mod.BUDGET.admit = track_admit
        mod.BUDGET.release = track_release
        mod._post_backend = lambda path, payload, timeout=60: {
            "choices": [{"finish_reason": "stop", "message": {"content": "Red square."}}],
            "usage": {"prompt_tokens": 400},
        }
        voff = mod.VisionOffload()
        msgs = [{"role": "user", "content": [{"type": "text", "text": "what?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}]
        n, new = voff.apply(msgs)
        self.assertEqual((n, new), (1, 1))
        self.assertEqual(len(admits), 1)
        self.assertEqual(admits[0][0], "vision-offload")
        self.assertEqual(released, [admits[0][2]])
        self.assertEqual(mod.BUDGET.used, 0)

    def test_decode_priority_does_not_hold_checkpoint_fork(self):
        state = MockLoomState()
        state.input_tokens = 12000
        state.ckpt_pos = 11500
        srv, bport = start_mock_loom(state)
        proc, pport, logs = start_proxy_proc(PROXY_PATH, bport, {
            "LOOM_LIVE": "0",
            "LOOM_TOKEN_BUDGET": "200000",
            "LOOM_MAX_SESSIONS": "4",
            "LOOM_SYS_CHECKPOINT": "1",
            "LOOM_CHECKPOINT_CAPACITY": "2",
            "LOOM_SYS_CKPT_MIN": "4096",
            "LOOM_DECODE_PRIORITY": "1",
            "LOOM_DECODE_PRIORITY_MIN": "8192",
            "LOOM_DECODE_PRIORITY_WAIT": "3",
        })
        try:
            sysm = {"role": "system", "content": "S" * 1000}
            # Two fresh requests trigger checkpoint creation (seen == 2)
            for i in range(2):
                st, _ = http_post(pport, "/v1/chat/completions", {"model": "local", "messages": [sysm, {"role": "user", "content": f"hi {i}"}]})
                self.assertEqual(st, 200)
            # Now start a slow replying chat in background
            state.mode = "sse_slow_reply"
            state.stream_delay = 1.5
            bg_done = threading.Event()

            def bg():
                http_post(pport, "/v1/chat/completions", {"model": "local", "stream": True, "messages": [{"role": "user", "content": "reply slowly"}]})
                bg_done.set()

            threading.Thread(target=bg, daemon=True).start()
            time.sleep(0.25)  # wait for first SSE token so DecodePriority sees 1 replying chat

            # Send a 3rd fresh chat with the checkpointed system prompt (ptok=12000, ckpt=11500 -> fresh=500 < 8192)
            state.mode = "ok_json"
            t0 = time.time()
            st, _ = http_post(pport, "/v1/chat/completions", {"model": "local", "messages": [sysm, {"role": "user", "content": "hi 3"}]})
            dt = time.time() - t0
            self.assertEqual(st, 200)
            self.assertLess(dt, 0.8, f"Checkpoint-forking request was wrongly held by DecodePriority ({dt:.2f}s)! Logs:\n{''.join(logs)}")
            bg_done.wait(timeout=5)
        finally:
            stop_servers(proc, srv)


def bench_proxy(proxy_path, n_iters=40, payload_kb=900):
    state = MockLoomState()
    srv, bport = start_mock_loom(state)
    proc, pport, _ = start_proxy_proc(proxy_path, bport, {"LOOM_LIVE": "0", "LOOM_TOKEN_BUDGET": "450000", "LOOM_MAX_SESSIONS": "4"})
    big_text = "x" * (payload_kb * 1024)
    payload = {
        "model": "local",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": big_text},
        ],
    }
    try:
        # warmup
        for _ in range(5):
            st, _ = http_post(pport, "/v1/chat/completions", payload)
            assert st == 200
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            st, _ = http_post(pport, "/v1/chat/completions", payload)
            dt = (time.perf_counter() - t0) * 1000
            assert st == 200
            times.append(dt)
        times.sort()
        return {
            "median_ms": round(statistics.median(times), 2),
            "p95_ms": round(times[int(0.95 * len(times))], 2),
            "mean_ms": round(statistics.mean(times), 2),
        }
    finally:
        proc.terminate(); proc.wait(timeout=3); srv.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-baseline", help="Optional path to baseline proxy.py for latency A/B benchmark")
    ap.add_argument("--iters", type=int, default=40)
    args, rest = ap.parse_known_args()
    if args.bench_baseline:
        old = bench_proxy(args.bench_baseline, n_iters=args.iters)
        new = bench_proxy(PROXY_PATH, n_iters=args.iters)
        speedup = old["median_ms"] / max(new["median_ms"], 1e-6)
        print(json.dumps({"benchmark": "900KB_chat_relay", "iters": args.iters, "baseline": old, "patched": new, "median_speedup": round(speedup, 2)}))
    else:
        unittest.main(argv=[sys.argv[0]] + rest, verbosity=2)
