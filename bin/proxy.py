#!/usr/bin/env python3
"""Expose the Loom model as `local` for Pi, Hermes, and other clients.

  - POST bodies: "model" is rewritten to the served tag.
  - GET /v1/models: also lists "local".
  - GET /props: llama.cpp-style {"default_generation_settings": {"n_ctx": N}} for hermes-sync-ctx.
  - LOOM_VISION=1 advertises image support and preserves multimodal content parts.
  - everything else (incl. SSE streams) is passed through unchanged, flushed as it arrives.
Logs one line per request to stdout: path, status, TTFT, total time, bytes.

usage: proxy.py LISTEN_PORT BACKEND_PORT SERVED_TAG
"""
import http.client
import os
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN, BACKEND, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
VISION = os.environ.get("LOOM_VISION") == "1"
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "proxy-connection", "upgrade", "te", "trailers"}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# Approved sampling (BetterBench config/qwen-thinking-xtx.json). Loom's own default is temperature 0 = greedy,
# which makes Inkling loop in its reasoning; clients like Pi send no sampling fields, so fill them in here.
# Temperature 0 (greedy) is never passed through.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


LIVE = os.environ.get("LOOM_LIVE", "1") != "0"          # live pp/tg lines while a streamed reply runs
LIVE_SECS = float(os.environ.get("LOOM_LIVE_SECS", "3"))


def _post_backend(path, payload, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=timeout)
    try:
        c.request("POST", path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        return json.loads(c.getresponse().read())
    finally:
        c.close()


class LiveMeter:
    """Every LIVE_SECS while a streamed chat reply runs, log prefill progress, then pp and live tg.
    Prompt size comes from Loom's /v1/chat/completions/input_tokens; generated tokens are counted by re-tokenizing
    the streamed text with /tokenize (a few ms), so tg here is approximate (~); the final line uses Loom's usage."""

    def __init__(self, body):
        self.t0 = time.time(); self.first = None; self.prompt = None; self.gen = 0
        self.pending = []; self.lock = threading.Lock(); self.stop = threading.Event()
        self.last_t = None; self.last_gen = 0
        threading.Thread(target=self._count_prompt, args=(body,), daemon=True).start()
        threading.Thread(target=self._run, daemon=True).start()

    def _count_prompt(self, body):
        try:
            j = json.loads(body)
            if "image_url" in body.decode("utf-8", "ignore"):
                return                      # counting would run the image path; final line still reports it
            j.pop("stream", None); j.pop("stream_options", None)
            self.prompt = int(_post_backend("/v1/chat/completions/input_tokens", j).get("input_tokens") or 0) or None
        except Exception:
            pass

    def text(self, s):
        if s:
            with self.lock:
                if self.first is None:
                    self.first = time.time()
                    p = self.prompt
                    dt = self.first - self.t0
                    log(f"  prefill done: {p if p else '?'} tok in {dt:.1f}s" + (f" = pp {p / dt:,.0f} tok/s (incl. cached prefix)" if p and dt > 0 else ""))
                    self.last_t = self.first
                self.pending.append(s)

    def _tick(self):
        now = time.time()
        if self.first is None:
            p = self.prompt
            log(f"  prefilling {p if p else '?'} tok ... {now - self.t0:.0f}s")
            return
        with self.lock:
            chunk = "".join(self.pending); self.pending = []
        if chunk:
            try:
                self.gen += len(_post_backend("/tokenize", {"model": TAG, "content": chunk}, timeout=10).get("tokens") or [])
            except Exception:
                self.gen += max(1, len(chunk) // 4)
        win = (self.gen - self.last_gen) / max(now - self.last_t, 1e-6)
        avg = self.gen / max(now - self.first, 1e-6)
        log(f"  generating: ~{self.gen} tok  tg ~{win:.0f} tok/s now, ~{avg:.0f} avg  ({now - self.t0:.0f}s)")
        self.last_t, self.last_gen = now, self.gen

    def _run(self):
        while not self.stop.wait(LIVE_SECS):
            try:
                self._tick()
            except Exception:
                pass

    def done(self):
        self.stop.set()


class TokenBudget:
    """Admission guard for Loom's shared KV pool (2026-11-24).

    Loom's elastic pool has no per-request reservation: if the live conversations together outgrow physical memory,
    Loom fails the whole model ("model residency is waiting for capacity") and every stream returns an empty 200.
    So the proxy admits a chat only if (sum of admitted costs) + this cost <= LOOM_TOKEN_BUDGET, where
    cost = prompt tokens + reply reservation (max_tokens, capped by the context). Requests queue FIFO; a request
    always gets in when nothing else is running. Idle cached histories are not counted: Loom reclaims them under
    pressure (verified 2026-11-24 with 5 x 91K chats over 4 cached ones).
    """

    def __init__(self, budget, max_active, timeout):
        self.budget, self.max_active, self.timeout = budget, max_active, timeout
        self.cv = threading.Condition(); self.used = 0; self.active = {}; self.queue = []; self.seq = 0
        self.pinned = 0   # tokens pinned by system-prompt checkpoints (SysCheckpoints)

    def admit(self, cost, label):
        with self.cv:
            self.seq += 1; t = self.seq; self.queue.append((t, cost))
            t0 = time.time(); waited = False
            while True:
                head = self.queue[0][0] == t
                fits = self.used + cost <= self.budget - self.pinned and len(self.active) < self.max_active
                if head and (fits or not self.active):
                    break
                if not waited:
                    waited = True
                    log(f"  queued {label}: cost {cost:,} tok; in use {self.used:,}/{self.budget - self.pinned:,} by {len(self.active)}, "
                        f"{len(self.queue) - 1} ahead/behind")
                left = self.timeout - (time.time() - t0)
                if left <= 0:
                    self.queue = [q for q in self.queue if q[0] != t]; self.cv.notify_all()
                    return None
                self.cv.wait(min(left, 5))
            self.queue.pop(0); self.active[t] = cost; self.used += cost; self.cv.notify_all()
            log(f"  admitted {label}: cost {cost:,} tok; in use {self.used:,}/{self.budget - self.pinned:,} by {len(self.active)}"
                + (f" after {time.time() - t0:.1f}s queued" if waited else ""))
            return t

    def release(self, t):
        with self.cv:
            self.used -= self.active.pop(t, 0); self.cv.notify_all()

    def status(self):
        with self.cv:
            return {"budget_tokens": self.budget, "pinned_tokens": self.pinned, "in_use_tokens": self.used, "active": len(self.active),
                    "max_active": self.max_active, "queued": len(self.queue), "queue_timeout_s": self.timeout}


BUDGET = TokenBudget(int(os.environ.get("LOOM_TOKEN_BUDGET", "0") or 0),
                     int(os.environ.get("LOOM_MAX_SESSIONS", "0") or 0) or 1 << 30,
                     float(os.environ.get("LOOM_QUEUE_TIMEOUT", "1800")))
CARD_MAX_TOKENS = int(os.environ.get("LOOM_MAX_TOKENS", "48000"))


class PrefixDiff:
    """Prefix-reuse diagnostics (2026-11-24). Loom only reuses a conversation's cached state when the client
    re-sends earlier turns exactly as before (and an assistant turn exactly as generated). For each chat request,
    fingerprint every message; if a request continues a known conversation (same first two messages) but an
    earlier message changed, log where and how: which index, role, which fields appeared or vanished, and
    length changes. Never logs content. Off with LOOM_PREFIX_DIFF=0."""

    def __init__(self, keep=32):
        self.keep = keep; self.convs = {}; self.lock = threading.Lock()

    @staticmethod
    def _shape(m):
        out = {}
        for k, v in m.items():
            if v in (None, "", [], {}):
                continue
            out[k] = len(v) if isinstance(v, str) else len(json.dumps(v, sort_keys=True))
        return out

    def check(self, msgs):
        import hashlib
        if not isinstance(msgs, list) or len(msgs) < 2:
            return
        fp = [hashlib.sha1(json.dumps(m, sort_keys=True).encode()).hexdigest()[:12] for m in msgs]
        key = fp[0] + fp[1]
        # Inkling always generates reasoning; an assistant turn sent back without any reasoning field no longer
        # matches what was generated, so Loom cannot continue the resident row past it (measured: 97% -> 7% reuse).
        bare = sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"
                   and not (m.get("reasoning_content") or m.get("reasoning")))
        with self.lock:
            if bare and self.convs.get(key, (None, None, 0))[2:] != (bare,):
                log(f"  prefix-diff conv {key[:8]}: {bare} assistant turn(s) sent back without reasoning - "
                    f"limits Loom's turn-to-turn reuse")
            prev = self.convs.pop(key, None)
            self.convs[key] = (fp, [self._shape(m) if isinstance(m, dict) else {} for m in msgs], bare)
            while len(self.convs) > self.keep:
                self.convs.pop(next(iter(self.convs)))
        if not prev:
            return
        pfp, pshape = prev[0], prev[1]
        i = next((k for k in range(min(len(pfp), len(fp))) if pfp[k] != fp[k]), None)
        if i is None:
            if len(fp) < len(pfp):
                log(f"  prefix-diff conv {key[:8]}: history shrank {len(pfp)} -> {len(fp)} messages (truncated/compacted)")
            return
        a, b = pshape[i], self._shape(msgs[i]) if isinstance(msgs[i], dict) else {}
        gone = sorted(set(a) - set(b)); new = sorted(set(b) - set(a))
        changed = {k: f"{a[k]}->{b[k]}" for k in set(a) & set(b) if a[k] != b[k]}
        role = msgs[i].get("role") if isinstance(msgs[i], dict) else "?"
        log(f"  prefix-diff conv {key[:8]}: message {i}/{len(pfp)} ({role}) changed since the last request; "
            f"fields gone {gone} new {new} length {changed or ('unchanged' if gone or new else 'same, content edited')} - Loom must re-read from here")


PREFIX_DIFF = PrefixDiff() if os.environ.get("LOOM_PREFIX_DIFF", "1") != "0" else None


class SysCheckpoints:
    """Reuse a shared long system prompt across NEW conversations via Loom named checkpoints (2026-11-24).

    With Loom's prefix cache off (it never evicted and blocked memory reclaim), a new conversation re-reads its whole
    system prompt + tool declarations (18K tokens: ~4.5 s). Loom's checkpoints are explicit pins, but a request can only
    fork one if its history *extends* the pinned history exactly. So, the second time a new conversation arrives with
    the same (system prompt, tools, tool_choice, reasoning_effort), the proxy runs a tiny seed conversation:
    [system, "Reply with only the word: ok"] -> a naturally finished reply. It pins that as a checkpoint, then inserts
    the same two seed messages after the system prompt of every new conversation with that key (and on all of that
    conversation's later turns, so turn-to-turn reuse is unaffected), and sends X-Loom-Checkpoint. The model therefore
    sees one extra short exchange at the start of such conversations. At most LOOM_CHECKPOINT_CAPACITY pins (LRU
    replaced); pinned tokens are subtracted from the KV budget. If Loom rejects a fork, the request is retried plain.
    Off unless LOOM_SYS_CHECKPOINT=1. Small prompts (< LOOM_SYS_CKPT_MIN tokens, default 4096) are never pinned.
    """
    SEED_USER = "Reply with only the word: ok"

    def __init__(self, cap, min_tokens):
        from collections import OrderedDict
        self.cap, self.min = cap, min_tokens
        self.lock = threading.Lock()
        self.ck = OrderedDict()   # key -> {"name", "seed", "position"}; LRU order
        self.seen, self.bad, self.creating = {}, set(), set()
        self.injected = OrderedDict()   # conversation id -> key
        self.n = 0

    @staticmethod
    def _key(j):
        import hashlib
        m0 = j["messages"][0]
        return hashlib.sha1(json.dumps([m0, j.get("tools"), j.get("tool_choice"), j.get("reasoning_effort")],
                                       sort_keys=True).encode()).hexdigest()[:16]

    def _call(self, method, path, body=None, hdr=None, timeout=600):
        c = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=timeout)
        try:
            h = {"Content-Type": "application/json", "X-Loom-Model": TAG, **(hdr or {})}
            data = json.dumps(body).encode() if body is not None else b""
            h["Content-Length"] = str(len(data))
            c.request(method, path, data, h)
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def _create(self, k, j):
        sysm = j["messages"][0]
        base = {k2: j[k2] for k2 in ("tools", "tool_choice", "reasoning_effort") if j.get(k2) is not None}
        st, raw = self._call("POST", "/v1/chat/completions/input_tokens", {"model": TAG, "messages": [sysm], **base})
        n_sys = (json.loads(raw).get("input_tokens") or 0) if st == 200 else 0
        if n_sys < self.min:
            log(f"  sys-ckpt {k[:8]}: system prompt {n_sys:,} tok < {self.min:,}, not pinned"); return None
        with self.lock:
            self.n += 1; name = f"sysp-{k[:12]}-{self.n}"
        seed_body = {"model": TAG, "messages": [sysm, {"role": "user", "content": self.SEED_USER}], "max_tokens": 512,
                     "temperature": 1.0, "top_p": 0.95, "top_k": 20, **base}
        st, raw = self._call("POST", "/v1/chat/completions", seed_body, {"X-Loom-Session": name})
        if st != 200:
            log(f"  sys-ckpt {k[:8]}: seed failed {st} {raw[:120]!r}"); return None
        d = json.loads(raw); ch = d["choices"][0]; m = ch.get("message") or {}
        if ch.get("finish_reason") != "stop" or m.get("tool_calls"):
            log(f"  sys-ckpt {k[:8]}: seed did not finish plainly ({ch.get('finish_reason')}), not pinned"); return None
        a = {"role": "assistant", "content": m.get("content")}
        if m.get("reasoning_content"):
            a["reasoning_content"] = m["reasoning_content"]
        with self.lock:   # make room (LRU)
            old = [(kk, self.ck[kk]) for kk in list(self.ck)[: max(0, len(self.ck) - self.cap + 1)]]
        for kk, e in old:
            self._call("DELETE", f"/v1/checkpoints/{e['name']}")
            with self.lock:
                self.ck.pop(kk, None); BUDGET.pinned -= e["position"]
            log(f"  sys-ckpt {kk[:8]}: unpinned {e['name']} (LRU)")
        st, raw = self._call("POST", f"/v1/checkpoints/{name}", None, {"X-Loom-Session": name})
        if st != 201:
            log(f"  sys-ckpt {k[:8]}: pin failed {st} {raw[:120]!r}"); return None
        pos = int(json.loads(raw).get("position") or 0)
        e = {"name": name, "seed": [{"role": "user", "content": self.SEED_USER}, a], "position": pos}
        with self.lock:
            self.ck[k] = e; BUDGET.pinned += pos
        log(f"  sys-ckpt {k[:8]}: pinned {name} at {pos:,} tokens (system {n_sys:,})")
        return e

    def apply(self, j):
        """Insert the seed and return (checkpoint name, key, conversation id), or None to send the request unchanged."""
        import hashlib
        msgs = j.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2 or not isinstance(msgs[0], dict) or msgs[0].get("role") != "system":
            return None
        k = self._key(j)
        conv = hashlib.sha1((k + json.dumps(msgs[1], sort_keys=True)).encode()).hexdigest()[:16]
        fresh = not any(isinstance(m, dict) and m.get("role") == "assistant" for m in msgs)
        with self.lock:
            prev = None if fresh else self.injected.get(conv)   # a fresh request (no assistant turn yet) always
            if prev is not None:   # starts a new fork, even if its first message repeats an earlier one; a conversation we already seeded: same seed on every later turn, no header
                self.injected.move_to_end(conv)                         # (the resident row continues it; a
                j["messages"] = [msgs[0]] + prev[1] + msgs[1:]          # checkpoint fork would discard turn history)
                return None
            if k in self.bad:
                return None
            e = self.ck.get(k); inj = False
            if e is None:
                if not fresh or k in self.creating:
                    return None
                self.seen[k] = self.seen.get(k, 0) + 1
                if self.seen[k] < 2:
                    return None
                self.creating.add(k)
            elif not (fresh or inj):
                return None
        if e is None:
            try:
                e = self._create(k, j)
            except Exception as ex:
                log(f"  sys-ckpt {k[:8]}: error {ex!r}"); e = None
            with self.lock:
                self.creating.discard(k)
                if e is None:
                    self.bad.add(k); return None
        j["messages"] = [msgs[0]] + e["seed"] + msgs[1:]
        if not fresh:   # mid-conversation first sighting: never insert into an existing history
            j["messages"] = msgs
            return None
        with self.lock:
            if k in self.ck:
                self.ck.move_to_end(k)
            self.injected[conv] = (k, e["seed"]); self.injected.move_to_end(conv)
            while len(self.injected) > 1024:
                self.injected.popitem(last=False)
        return e["name"], k, conv

    def drop(self, k, conv, why):
        structural = "does not extend" in why   # 400: this key can never fork -> stop; 404 (pin gone): re-create later
        with self.lock:
            e = self.ck.pop(k, None); self.injected.pop(conv, None)
            if structural:
                self.bad.add(k)
            if e:
                BUDGET.pinned -= e["position"]
        if e:
            try:
                self._call("DELETE", f"/v1/checkpoints/{e['name']}")
            except Exception:
                pass
        log(f"  sys-ckpt {k[:8]}: {'disabled' if structural else 'dropped (will re-pin)'} after fork rejection ({why[:100]})")

    def status(self):
        with self.lock:
            return {"pinned": [{"name": e["name"], "position": e["position"]} for e in self.ck.values()],
                    "conversations": len(self.injected), "skipped_keys": len(self.bad)}


SYSCKPT = (SysCheckpoints(int(os.environ.get("LOOM_CHECKPOINT_CAPACITY", "2") or 2), int(os.environ.get("LOOM_SYS_CKPT_MIN", "4096")))
           if os.environ.get("LOOM_SYS_CHECKPOINT") == "1" and int(os.environ.get("LOOM_CHECKPOINT_CAPACITY", "0") or 0) > 0 else None)


def _estimate_cost(j):
    """Prompt tokens (exact via Loom when the request has no image) + reply reservation."""
    prompt = None
    raw = json.dumps(j.get("messages") or [])
    if "image_url" not in raw and '"input_image"' not in raw:
        try:
            q = dict(j); q.pop("stream", None); q.pop("stream_options", None)
            prompt = int(_post_backend("/v1/chat/completions/input_tokens", q, timeout=120).get("input_tokens") or 0) or None
        except Exception:
            prompt = None
    if prompt is None:   # images or counting failed: text chars / 3 (conservative for this tokenizer) + 3000 per image
        text = sum(len(_text(m.get("content")) or "") for m in j.get("messages") or [] if isinstance(m, dict))
        prompt = text // 3 + 3000 * raw.count("image_url") + 1024
    nctx = int(os.environ.get("LOOM_NCTX") or 0) or 1 << 30
    gen = j.get("max_completion_tokens") or j.get("max_tokens") or CARD_MAX_TOKENS
    try:
        gen = int(gen)
    except Exception:
        gen = CARD_MAX_TOKENS
    gen = max(0, min(gen, nctx - prompt))
    return min(prompt + gen, nctx), prompt


def _apply_sampling(j):
    changed = []
    for k, v in SAMPLING.items():
        if j.get(k) is None:
            j[k] = v
            changed.append(k)
    t = j.get("temperature")
    if not isinstance(t, (int, float)) or t <= 0:
        j["temperature"] = SAMPLING["temperature"]
        changed.append(f"temperature {t!r}->1.0")
    return changed


def _text(content):
    """Loom takes plain-string content only: join text parts, drop images/other parts."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
                out.append(part.get("text") or "")
        return "\n".join(out)
    return str(content)


def normalize_messages(msgs):
    """Normalize message roles and flatten content for text-only packages."""
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        m = dict(m)
        role = m.get("role")
        if role == "developer":
            m["role"] = "system"
        elif role == "function":
            m["role"] = "tool"
        if "content" in m:
            if not VISION:
                m["content"] = _text(m["content"])
            if m["content"] is None and m.get("role") != "assistant":
                m["content"] = ""
        out.append(m)
    if VISION and os.environ.get("LOOM_KEEP_OLD_IMAGES") != "1":
        out = _drop_old_images(out)
    return out


def _is_image_part(p):
    return isinstance(p, dict) and (p.get("type") in ("image_url", "input_image", "image") or "image_url" in p)


def _drop_old_images(msgs):
    """Keep image parts only after the last assistant message (the current step).

    Loom never reuses its prefix cache for a conversation that contains an image (cached_tokens stays at 2048), so a
    screenshot left in the history makes every later turn re-prefill the whole conversation (243K: ~2 min per turn).
    Older images become a short text note; the model already saw them and its reply describing them stays (2026-11-23).
    """
    last_assistant = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
    out = []
    for i, m in enumerate(msgs):
        c = m.get("content")
        if i < last_assistant and isinstance(c, list) and any(_is_image_part(p) for p in c):
            m = dict(m)
            parts = [p if not _is_image_part(p) else {"type": "text", "text": "[image shown earlier]"} for p in c]
            if all(isinstance(p, dict) and p.get("type") == "text" for p in parts):
                m["content"] = "\n".join(p.get("text", "") for p in parts)
            else:
                m["content"] = parts
        out.append(m)
    return out


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def _backend(self, method, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=3600)
        c.request(method, self.path, body=body, headers=headers or {})
        return c, c.getresponse()

    def _send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _n_ctx(self):
        import os
        if os.environ.get("LOOM_NCTX"):
            return int(os.environ["LOOM_NCTX"])
        try:
            c = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=5)
            c.request("GET", "/stats")
            s = json.loads(c.getresponse().read())
            return int((s.get("loaded_config") or {}).get("max_seq") or 0)
        except Exception:
            return 0

    def _flush_history(self):
        try:
            fb = json.dumps({"model": TAG, "messages": [{"role": "user", "content": "."}], "max_tokens": 1,
                             "temperature": 1.0, "stream": False}).encode()
            fc = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=120)
            fc.request("POST", "/v1/chat/completions", fb, {"Content-Type": "application/json", "Content-Length": str(len(fb))})
            fc.getresponse().read(); fc.close()
        except Exception as e:
            log(f"history flush failed: {e}")

    def _relay(self, method):
        t0 = time.time()
        self._ck = None; self._plain_body = None; self._plain_msgs = None
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        model_in = None
        stream = False
        strip_usage = False   # we asked for usage the client did not: drop that chunk again
        if body and "json" in (self.headers.get("Content-Type") or "json"):
            try:
                j = json.loads(body)
                if isinstance(j, dict) and isinstance(j.get("messages"), list):
                    j["messages"] = normalize_messages(j["messages"])
                    _apply_sampling(j)
                plain_msgs = None
                if (SYSCKPT and isinstance(j, dict) and isinstance(j.get("messages"), list)
                        and self.path.split("?", 1)[0].rstrip("/") in ("/v1/chat/completions", "/chat/completions")):
                    plain_msgs = list(j["messages"]); self._plain_msgs = plain_msgs
                    try:
                        self._ck = SYSCKPT.apply(j)
                    except Exception as ex:
                        log(f"  sys-ckpt error: {ex!r}"); self._ck = None
                if isinstance(j, dict) and "model" in j:
                    model_in = j["model"]
                    j["model"] = TAG
                    stream = bool(j.get("stream"))
                    if stream and self.path.startswith("/v1/") and "completions" in self.path:
                        so = j.get("stream_options") or {}
                        if not so.get("include_usage"):
                            so["include_usage"] = True
                            j["stream_options"] = so
                            strip_usage = True
                    body = json.dumps(j).encode()
                    if getattr(self, "_ck", None):
                        jp = dict(j); jp["messages"] = plain_msgs; self._plain_body = json.dumps(jp).encode()
            except Exception:
                pass
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "host"}
        if getattr(self, "_ck", None):
            headers["X-Loom-Checkpoint"] = self._ck[0]
        if body is not None:
            headers["Content-Length"] = str(len(body))
        if (BUDGET.budget and method == "POST" and body
                and self.path.split("?", 1)[0].rstrip("/") in ("/v1/chat/completions", "/chat/completions")):
            try:
                jb = json.loads(body)
            except Exception:
                jb = None
            if isinstance(jb, dict):
                if PREFIX_DIFF:
                    try:
                        PREFIX_DIFF.check(self._plain_msgs if self._plain_msgs is not None else jb.get("messages"))
                    except Exception as e:
                        log(f"  prefix-diff error: {e!r}")
                cost, ptok = _estimate_cost(jb)
                label = f"prompt {ptok:,}" + (" (est)" if "image_url" in body.decode("utf-8", "ignore") else "")
                self._ticket = BUDGET.admit(cost, label)
                if self._ticket is None:
                    self._send_json(503, {"error": {"message": f"Loom busy: waited {BUDGET.timeout:.0f}s for {cost:,} tokens of KV budget",
                                                    "type": "server_error", "code": "budget_timeout"}})
                    log(f"{method} {self.path} 503 budget queue timeout (cost {cost:,})")
                    return
        meter = LiveMeter(body) if (LIVE and stream and method == "POST" and body and "chat/completions" in self.path) else None
        try:
            c, r = self._backend(method, body, headers)
        except Exception as e:
            if meter: meter.done()
            self._send_json(502, {"error": {"message": f"loom backend not reachable: {e}", "type": "server_error"}})
            log(f"{method} {self.path} 502 backend down")
            return
        if getattr(self, "_ck", None) and r.status in (400, 404):
            errc = r.read()
            if b"checkpoint" in errc:   # fork rejected: drop this checkpoint and send the client's own request
                c.close()
                SYSCKPT.drop(self._ck[1], self._ck[2], errc.decode("utf-8", "replace"))
                body = self._plain_body; headers.pop("X-Loom-Checkpoint", None); headers["Content-Length"] = str(len(body))
                self._ck = None
                try:
                    c, r = self._backend(method, body, headers)
                except Exception as e:
                    self._send_json(502, {"error": {"message": f"loom backend not reachable: {e}", "type": "server_error"}})
                    return
            else:
                self.send_response(r.status); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(errc))); self.end_headers(); self.wfile.write(errc); c.close()
                log(f"{method} {self.path} {r.status} {errc[:160]!r}")
                return
        # Loom keeps the previous conversation resident until the next request replaces it. An image turn (rendered
        # with the generic template, so no prefix reuse) or a new long conversation can then need two copies and is
        # refused. Replace the held history with a 1-token throwaway request and retry once (2026-11-23).
        if r.status == 503 and body and method == "POST":
            err503 = r.read()
            if any(k in err503 for k in (b"retained state leaves insufficient completion capacity", b"exceeds the serving memory limit")):
                c.close()
                self._flush_history()
                log(f"freed held history after: {err503[:90]!r} - retrying once")
                try:
                    c, r = self._backend(method, body, headers)
                except Exception as e:
                    self._send_json(502, {"error": {"message": f"loom backend not reachable: {e}", "type": "server_error"}})
                    return
                if r.status == 503:   # still over the memory limit with one copy: shrink the reply reservation once
                    err2 = r.read()
                    try:
                        j3 = json.loads(body)
                    except Exception:
                        j3 = None
                    if b"exceeds the serving memory limit" in err2 and isinstance(j3, dict):
                        for k in ("max_tokens", "max_completion_tokens"):
                            if k in j3:
                                j3[k] = min(int(j3[k]), 2048)
                        if "max_tokens" not in j3 and "max_completion_tokens" not in j3:
                            j3["max_tokens"] = 2048
                        body = json.dumps(j3).encode(); headers["Content-Length"] = str(len(body))
                        c.close(); self._flush_history()
                        log("still over the memory limit - retrying with max_tokens 2048")
                        try:
                            c, r = self._backend(method, body, headers)
                        except Exception as e:
                            self._send_json(502, {"error": {"message": f"loom backend not reachable: {e}", "type": "server_error"}})
                            return
                    else:
                        self.send_response(503); self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err2))); self.end_headers(); self.wfile.write(err2)
                        c.close(); log(f"{method} {self.path} 503 after retry {err2[:120]!r}"); return
            else:
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(err503)))
                self.end_headers()
                self.wfile.write(err503)
                c.close()
                log(f"{method} {self.path} {r.status} {err503[:160]!r}")
                return
        # Loom rejects input + max_tokens > context (clients like Pi ask for ~110K). Retry once with max_tokens clamped.
        if r.status == 400 and body:
            err = r.read()
            m = re.search(rb"(\d+) input tokens", err) if b"OUT_OF_RANGE" in err else None
            nctx = self._n_ctx()
            if m and nctx:
                room = nctx - int(m.group(1)) - 64
                try:
                    j2 = json.loads(body)
                except Exception:
                    j2 = None
                if room > 0 and isinstance(j2, dict):
                    for k in ("max_tokens", "max_completion_tokens"):
                        if k in j2:
                            j2[k] = min(int(j2[k]), room)
                    if "max_tokens" not in j2 and "max_completion_tokens" not in j2:
                        j2["max_tokens"] = room
                    body = json.dumps(j2).encode()
                    headers["Content-Length"] = str(len(body))
                    c.close()
                    log(f"clamped max_tokens to {room} ({m.group(1).decode()} input tokens, {nctx} ctx)")
                    try:
                        c, r = self._backend(method, body, headers)
                    except Exception as e:
                        self._send_json(502, {"error": {"message": f"loom backend not reachable: {e}", "type": "server_error"}})
                        return
                    err = None
            if err is not None:
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
                c.close()
                log(f"{method} {self.path} {r.status} {err[:160]!r}")
                return
        if method == "GET" and self.path.rstrip("/") == "/v1/models" and r.status == 200:
            data = json.loads(r.read())
            ids = [m.get("id") for m in data.get("data", [])]
            if "local" not in ids and data.get("data"):
                alias = dict(data["data"][0]); alias["id"] = "local"
                data["data"].insert(0, alias)
            # Pi's pi-llama-cpp reads meta.n_ctx (llama-server's field) and falls back to 128K without it.
            for m in data.get("data", []):
                n = m.get("context_length") or self._n_ctx()
                if n:
                    m.setdefault("meta", {})["n_ctx"] = n
                m.setdefault("architecture", {})["input_modalities"] = ["text", "image"] if VISION else ["text"]
            c.close()
            self._send_json(200, data)
            return
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        first = None; first_tok = None; total = 0; usage = None
        sse = "text/event-stream" in (r.getheader("Content-Type") or "")
        buf = b""; raw = b"" if not sse else None
        def note_event(ev):   # inspect one SSE event; return False to drop it
            nonlocal first_tok, usage
            for line in ev.split(b"\n"):
                if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                    continue
                try:
                    d = json.loads(line[5:])
                except Exception:
                    continue
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                if first_tok is None and any((c.get("delta") or {}).get(k) for c in ch for k in ("content", "reasoning_content", "reasoning", "tool_calls")):
                    first_tok = time.time()
                if meter:
                    for c_ in ch:
                        dl = c_.get("delta") or {}
                        for k in ("reasoning_content", "reasoning", "content"):
                            if isinstance(dl.get(k), str):
                                meter.text(dl[k])
                        for tc in dl.get("tool_calls") or []:
                            meter.text(((tc.get("function") or {}).get("arguments")) or "")
                if strip_usage and d.get("usage") and not ch:
                    return False
            return True
        while True:
            chunk = r.read1(65536)
            if not chunk:
                break
            first = first or time.time()
            total += len(chunk)
            out = chunk
            if sse:
                buf += chunk
                parts = buf.split(b"\n\n")
                buf = parts.pop()
                out = b"".join(p + b"\n\n" for p in parts if note_event(p))
            elif raw is not None and len(raw) < 4 << 20:
                raw += chunk
            try:
                if out:
                    self.wfile.write(out); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                log(f"{method} {self.path} client disconnected")
                break
        if sse and buf:
            try:
                if note_event(buf):
                    self.wfile.write(buf); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        if raw:
            try:
                usage = json.loads(raw).get("usage") or usage
            except Exception:
                pass
        c.close()
        if meter: meter.done()
        self.close_connection = True
        if self.path.startswith("/v1/"):
            ttft = first_tok or first
            pt = ct = gt = 0
            if usage:
                pt = int(usage.get("prompt_tokens") or 0); gt = int(usage.get("completion_tokens") or 0)
                ct = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            end = time.time()
            speed = ""
            if ttft and pt:
                pp = (pt - ct) / max(ttft - t0, 1e-6); speed += f" pp={pp:,.0f}"
            if ttft and gt and end - ttft > 0.05:
                speed += f" tg={gt / (end - ttft):.1f}"
            log(f"{method} {self.path} {r.status} model={model_in!r} prompt={pt} cached={ct} gen={gt} "
                f"first-token={(ttft - t0):.2f}s total={end - t0:.1f}s{speed} {total}B" if ttft else
                f"{method} {self.path} {r.status} model={model_in!r} prompt={pt} cached={ct} gen={gt} first-token=- total={time.time() - t0:.1f}s {total}B")

    def do_GET(self):
        if self.path.split("?", 1)[0].rstrip("/") == "/props":   # Pi asks /props?model=local&autoload=false
            n = self._n_ctx(); self._send_json(200, {"default_generation_settings": {"n_ctx": n}, "n_ctx": n, "modalities": {"vision": VISION, "audio": False}, "model_alias": "local", "served_tag": TAG})
            return
        if self.path.split("?", 1)[0].rstrip("/") == "/budget":
            st = BUDGET.status()
            if SYSCKPT:
                st["sys_checkpoints"] = SYSCKPT.status()
            self._send_json(200, st)
            return
        self._relay("GET")

    def do_POST(self):
        self._ticket = None; self._ck = None; self._plain_body = None; self._plain_msgs = None
        try:
            self._relay("POST")
        finally:
            if self._ticket is not None:
                BUDGET.release(self._ticket)

    def do_DELETE(self):
        self._relay("DELETE")

    def do_OPTIONS(self):
        self._relay("OPTIONS")


if __name__ == "__main__":
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer(("127.0.0.1", LISTEN), Proxy)
    log(f"proxy :{LISTEN} -> loom :{BACKEND} ({TAG}), model 'local' accepted; KV budget guard: "
        + (f"{BUDGET.budget:,} tokens, max {BUDGET.max_active} active, queue timeout {BUDGET.timeout:.0f}s" if BUDGET.budget else "off")
        + f"; system-prompt checkpoints: {'on, capacity ' + str(SYSCKPT.cap) if SYSCKPT else 'off'}")
    srv.serve_forever()
