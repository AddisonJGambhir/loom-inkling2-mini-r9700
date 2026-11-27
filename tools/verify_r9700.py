#!/usr/bin/env python3
"""Verify the R9700 changes on a running install (standard library only).

Checks (run any subset; default: kernel decode reuse):
  kernel       llm.loom contains decode attention v2, and whether prefill attention uses 4 tiles per workgroup
               (if --stock is given: that it was generated from the stock file by the tools here)
  decode       long-context decode keeps up: tok/s with ~47K tokens of context divided by tok/s at ~6K.
               On an R9700: v2 ~0.87, stock ~0.68 (from measured 8K/30K/60K stock rates). PASS if >= 0.75.
  reuse        agent-style follow-up turns reuse the cached prompt through the proxy, with a tool that has an
               `image_url` parameter and with tool-call arguments re-serialized by the client. PASS if >= 90% cached.
               (Stock proxy + Loom 0.2.0: ~0%.)
  concurrency  4 chats of ~30K tokens at once all answer correctly; reports whether the proxy budget guard is on.
  vision       a screenshot-style tool result with an image (--image PNG), then a follow-up: PASS if the follow-up
               is >= 90% cached (needs LOOM_VISION_OFFLOAD=1 in the proxy).

usage: python3 -I tools/verify_r9700.py [--proxy URL] [--backend URL] [--llm PATH] [--stock PATH] [--image PNG] [checks...]
Defaults: --proxy http://127.0.0.1:1234  --backend http://127.0.0.1:11437  --llm package/program/llm.loom
Exit status 0 when every requested check passes. Each check prints one JSON line; "measured" holds the numbers.
"""
import base64, hashlib, json, os, random, string, subprocess, sys, threading, time, urllib.error, urllib.request

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
PROXY = opt("--proxy", "http://127.0.0.1:1234").rstrip("/")
BACKEND = opt("--backend", "http://127.0.0.1:11437").rstrip("/")
LLM = opt("--llm", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "package/program/llm.loom"))
STOCK = opt("--stock", None)
IMAGE = opt("--image", None)
MODEL = opt("--model", "Inkling-2-Mini")
CHECKS = args or ["kernel", "decode", "reuse"]
W = ("river stone paper cloud engine garden silver market winter orange signal harbor violet "
     "candle forest mirror thunder copper meadow lantern").split()


def post(base, path, body, timeout=1800):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream_chat(base, body, timeout=1800):
    """Streamed chat; returns (message dict, usage, ttft seconds, decode tok/s)."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic(); first = None; msg = {"content": "", "reasoning_content": "", "tool_calls": {}}; usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                    first = time.monotonic()
                msg["content"] += d.get("content") or ""
                msg["reasoning_content"] += d.get("reasoning_content") or d.get("reasoning") or ""
                for tc in d.get("tool_calls") or []:
                    s = msg["tool_calls"].setdefault(tc.get("index", 0), {"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
                    if tc.get("id"):
                        s["id"] = tc["id"]
                    f = tc.get("function") or {}
                    if f.get("name"):
                        s["function"]["name"] = f["name"]
                    s["function"]["arguments"] += f.get("arguments") or ""
    end = time.monotonic()
    msg["tool_calls"] = [msg["tool_calls"][i] for i in sorted(msg["tool_calls"])]
    ttft = (first or end) - t0
    gen = usage.get("completion_tokens") or 0
    tps = (gen - 1) / (end - first) if first and gen > 1 and end > first else None
    return msg, usage, ttft, tps


def cached(u):
    return ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0), (u.get("prompt_tokens") or 1)


def echo(m, reserialize=False):
    """The assistant message as a client sends it back (optionally re-serializing tool arguments compactly, like some agents)."""
    a = {"role": "assistant", "content": m.get("content")}
    if m.get("reasoning_content"):
        a["reasoning_content"] = m["reasoning_content"]
    if m.get("tool_calls"):
        a["tool_calls"] = []
        for t in m["tool_calls"]:
            args_ = t["function"]["arguments"]
            if reserialize:
                try:
                    args_ = json.dumps(json.loads(args_), separators=(", ", ": "))
                except Exception:
                    pass
            a["tool_calls"].append({"id": t["id"], "type": "function", "function": {"name": t["function"]["name"], "arguments": args_}})
    return a


def report(name, ok, **measured):
    print(json.dumps({"check": name, "result": "PASS" if ok else ("SKIP" if ok is None else "FAIL"), "measured": measured}), flush=True)
    return ok is not False


def check_kernel():
    t = open(LLM).read()
    sha = hashlib.sha256(t.encode()).hexdigest()[:16]
    has_v2 = 'kernel.def export("g_ink_attn_split_span_v2")' in t
    s = t.index('kernel.def export("g_ink_attn_split")'); head = t[s:s + 400]
    one_wave = "%lanes = index.constant 32 : index" in head
    f = t.find('export("g_ink_attention_flash2_ts2_rdna4")')
    prefill_4tile = f >= 0 and "%clanes = index.constant 384 : index" in t[f:f + 400]
    flash = t[f:t.find("\n  kernel.return\n}\n", f)] if f >= 0 else ""
    prefill_prefetch = "%pf0_kp_raw" in flash
    fused_norm_q8 = 'kernel.def export("g_ink_rms_norm_plane_q8x")' in t
    moe_f16_act = "%f16_out_lo = buffer.view %w_stage" in t
    regen = None
    if STOCK:
        here = os.path.dirname(os.path.abspath(__file__))
        out = LLM + ".verify-regen"
        try:
            subprocess.run([sys.executable, "-I", os.path.join(here, "gen_attn_split_v2.py"), STOCK, out, "--span"], check=True, capture_output=True)
            if prefill_4tile:
                flags = ["--nsub", "4", "--key-major"] + (["--uniform"] if "%ms_subu" in flash else []) + (["--prefetch"] if prefill_prefetch else [])
                subprocess.run([sys.executable, "-I", os.path.join(here, "gen_flash2_multi.py"), out, out] + flags, check=True, capture_output=True)
            if fused_norm_q8:
                subprocess.run([sys.executable, "-I", os.path.join(here, "gen_fuse_nq.py"), out, out], check=True, capture_output=True)
            if moe_f16_act:
                subprocess.run([sys.executable, "-I", os.path.join(here, "gen_moe_f16_act.py"), out, out], check=True, capture_output=True)
            regen = open(out).read() == t
        finally:
            if os.path.exists(out):
                os.remove(out)
    return report("kernel", has_v2 and one_wave and regen is not False, llm_sha256_16=sha, span_v2=has_v2, single_token_v2=one_wave,
                  prefill_4tile=prefill_4tile, prefill_prefetch=prefill_prefetch, fused_norm_q8=fused_norm_q8,
                  moe_f16_act=moe_f16_act, regenerated_identically=regen)


def decode_tps(n_tokens, seed):
    rng = random.Random(seed)
    doc = " ".join(rng.choice(W) for _ in range(int(n_tokens / 1.6)))
    body = {"model": MODEL, "messages": [{"role": "user", "content": doc + "\n\nIgnore the words above. Write a long story about a lighthouse keeper."}],
            "max_tokens": 500, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "reasoning_effort": "none"}
    _, u, ttft, tps = stream_chat(BACKEND, body)
    return u.get("prompt_tokens"), tps


def check_decode():
    ps, ts = decode_tps(8000, int(time.time()))
    pl, tl = decode_tps(60000, int(time.time()) + 1)
    ratio = (tl / ts) if ts and tl else None
    return report("decode", bool(ratio and ratio >= 0.75), short_ctx=ps, short_tps=round(ts or 0, 1), long_ctx=pl, long_tps=round(tl or 0, 1),
                  ratio=round(ratio or 0, 3), expect="v2 ~0.87, stock ~0.68; PASS >= 0.75")


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the repository.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "vision_analyze", "description": "Look at an image and answer a question about it.",
          "parameters": {"type": "object", "properties": {"image_url": {"type": "string"}, "question": {"type": "string"}}, "required": ["image_url"]}}}]


def check_reuse():
    rng = random.Random(time.time_ns())
    code = "\n".join(f"def helper_{i}(x):\n    return x * {rng.randint(2, 9)}" for i in range(900))
    msgs = [{"role": "system", "content": f"You are a coding agent. [{time.time_ns()}]"},
            {"role": "user", "content": f"Here is src/util.py:\n```\n{code}\n```\nNow read src/a.py with read_file."}]
    body = {"model": MODEL, "tools": TOOLS, "max_tokens": 1500, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "reasoning_effort": "low"}
    for _ in range(4):   # the model occasionally answers instead of calling the tool
        m, u, _, _ = stream_chat(PROXY, dict(body, messages=msgs))
        if m["tool_calls"]:
            break
    else:
        return report("reuse", False, note="model never called a tool in 4 tries")
    msgs2 = msgs + [echo(m, reserialize=True)] + [{"role": "tool", "tool_call_id": t["id"], "content": "x = 1\n"} for t in m["tool_calls"]]
    m2, u2, ttft2, _ = stream_chat(PROXY, dict(body, messages=msgs2, max_tokens=64))
    c, p = cached(u2)
    return report("reuse", c >= 0.9 * p, follow_up_prompt=p, cached=c, pct=round(100 * c / p, 1), ttft_s=round(ttft2, 2),
                  note="tools include image_url; arguments re-serialized by the client")


def check_concurrency():
    try:
        budget = json.loads(urllib.request.urlopen(PROXY + "/budget", timeout=5).read())
    except Exception:
        budget = None
    res = {}
    def one(i):
        rng = random.Random(1000 + i)
        code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))
        words = [rng.choice(W) for _ in range(19000)]
        words.insert(rng.randint(2000, 17000), f". The secret code is {code}. ")
        body = {"model": MODEL, "messages": [{"role": "user", "content": " ".join(words) + "\n\nWhat is the secret code? Answer with the code only."}],
                "max_tokens": 1000, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "reasoning_effort": "none"}
        try:
            m, u, ttft, _ = stream_chat(PROXY, body)
            res[i] = (code in (m["content"] + m["reasoning_content"]), u.get("prompt_tokens"), round(ttft, 1))
        except Exception as e:
            res[i] = (False, repr(e)[:80], None)
    th = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    [t.start() for t in th]; [t.join() for t in th]
    ok = all(r[0] for r in res.values())
    return report("concurrency", ok, chats=[{"found": r[0], "prompt": r[1], "ttft_s": r[2]} for r in res.values()],
                  budget_guard=(budget or {}).get("budget_tokens") or "off/unavailable")


def check_vision():
    if not IMAGE:
        return report("vision", None, note="pass --image some.png (a screenshot-like PNG)")
    img = base64.b64encode(open(IMAGE, "rb").read()).decode()
    msgs = [{"role": "system", "content": f"You are a coding agent. [{time.time_ns()}]\n" + "\n".join(f"// line {i}" for i in range(2000))},
            {"role": "user", "content": "Check the screenshot and describe what it shows."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_v1", "type": "function", "function": {"name": "vision_analyze",
             "arguments": json.dumps({"image_url": "/tmp/shot.png", "question": "What does it show?"})}}]},
            {"role": "tool", "tool_call_id": "call_v1", "content": [{"type": "text", "text": "Question: What does it show?"},
                                                                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]}]
    body = {"model": MODEL, "tools": TOOLS, "max_tokens": 800, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "reasoning_effort": "low"}
    m, u, ttft, _ = stream_chat(PROXY, dict(body, messages=msgs))
    msgs2 = msgs + [echo(m), {"role": "user", "content": "Thanks. One more short sentence about it, please."}]
    m2, u2, ttft2, _ = stream_chat(PROXY, dict(body, messages=msgs2, max_tokens=64))
    c, p = cached(u2)
    return report("vision", c >= 0.9 * p, screenshot_turn_ttft_s=round(ttft, 2), follow_up_pct_cached=round(100 * c / p, 1),
                  follow_up_ttft_s=round(ttft2, 2), answer=m["content"][:120])


ok = True
for c in CHECKS:
    try:
        ok &= {"kernel": check_kernel, "decode": check_decode, "reuse": check_reuse, "concurrency": check_concurrency,
               "vision": check_vision}[c]()
    except KeyError:
        sys.exit(f"unknown check {c!r}")
    except Exception as e:
        ok &= report(c, False, error=repr(e)[:200])
sys.exit(0 if ok else 1)
