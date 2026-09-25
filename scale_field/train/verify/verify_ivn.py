"""CPU consistency check: released trainer vs stored run records (no training step).
usage: verify_ivn.py <script> <run_meta.json> <step0.pt or -> <tokens.npy> <out.json> -- <trainer args...>
Executes the trainer source up to its preflight block (model construction, init, data sampler), then
 (i) compares the 42 analysis matrices with the stored step-0 checkpoint,
 (ii) applies the arm's RoPE modification exactly as the trainer does and compares inv_freq with the record,
 (iii) draws the first five batches and compares their hashes with batch_hashes_first5,
 (iv) serializes the fp16 full state dict as the trainer does and compares its sha256_16 with the recorded step-0 snapshot.
"""
import sys, json, hashlib, os, tempfile, io
script, meta_p, ck_p, tok_p, out_p = sys.argv[1:6]
targs = sys.argv[7:]
src = open(script, encoding="utf-8").read()
cut = src.index("# ---- Preflight")
body = src[:cut].replace('dev = "cuda" if torch.cuda.is_available() else "cpu"', 'dev = "cpu"')
tmp = tempfile.mkdtemp()
sys.argv = [script] + targs + ["--out", tmp, "--tokens", tok_p, "--config",
                               os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(script))), "llama70m_config.json")]
ns = {"__file__": script, "__name__": "__verify__"}
exec(compile(body, script, "exec"), ns)
import torch, numpy as np
meta = json.load(open(meta_p))
res = {"script": os.path.relpath(script, os.path.dirname(os.path.dirname(os.path.abspath(script)))),
       "script_sha256_16": hashlib.sha256(open(script, "rb").read()).hexdigest()[:16],
       "recorded_script_sha256_16": meta.get("script_sha256_16"), "args": targs}
pmap, ANA = ns["pmap"], ns["ANA"]
# (i) analysis matrices vs step-0 checkpoint
if ck_p != "-":
    ck = torch.load(ck_p, map_location="cpu")
    diffs = {nm: float((pmap[nm].detach().float() - ck[nm].float()).abs().max()) for nm in ANA}
    res["step0_ckpt"] = {"n_matrices": len(ANA), "keys_match": sorted(ck) == sorted(ANA),
                         "n_bitwise_equal": sum(bool(torch.equal(pmap[nm].detach(), ck[nm])) for nm in ANA),
                         "max_abs_diff": max(diffs.values())}
# (ii) RoPE table
model = ns["model"]; arm = ns["ARM"]
# (v) whole-model check through the recorded preflight forward (covers embeddings, norms, lm_head)
pf_p = os.path.join(os.path.dirname(meta_p), "preflight.json")
if os.path.exists(pf_p):
    pf = json.load(open(pf_p))["preflight"]
    a_ = ns["a"]; toks = ns["toks"]; SEQ, BS = ns["SEQ"], ns["BS"]
    rp = np.random.default_rng(np.random.SeedSequence([a_.seed, 8800]))
    pidx = rp.integers(0, len(toks) - SEQ - 1, BS)
    px = torch.tensor(np.stack([toks[i:i + SEQ] for i in pidx]), dtype=torch.long)
    with torch.no_grad():
        o = model(input_ids=px, labels=px, output_hidden_states=True)
    hs = [float(h.float().pow(2).mean().sqrt()) for h in o.hidden_states]
    rec_hs = pf.get("hidden_rms_per_layer")
    res["preflight_forward"] = {"step0_loss": float(o.loss), "recorded_step0_loss": pf.get("step0_loss"),
                                "abs_diff_loss": abs(float(o.loss) - pf.get("step0_loss")),
                                "max_rel_diff_hidden_rms": (max(abs(x - y) / abs(y) for x, y in zip(hs, rec_hs))
                                                            if rec_hs else None)}
    sd = model.state_dict()
    res["non_analysis_tensors"] = sorted(k for k in sd if k not in ANA)
if arm == "ropeperm":
    ns["apply_rope_perm"](model, ns["PI"])
    got = model.model.rotary_emb.inv_freq.detach().float().cpu().tolist()
    rec = (meta.get("s3_ropeperm") or {}).get("inv_freq_after_layer0")
    res["rope"] = {"pi_equal_record": ns["PI"] == meta.get("pi"),
                   "inv_freq_equal_record": rec is not None and np.array_equal(np.float32(got), np.float32(rec)),
                   "inv_freq_max_abs_diff": float(np.max(np.abs(np.float32(got) - np.float32(rec)))) if rec else None}
elif arm == "nope":
    res["rope"] = {"note": "NoPE patches the rotary forward only; weights and inv_freq unchanged"}
# (iii) batch hashes
if "get_batch" in ns:
    try:
        for t in range(1, 6):
            (ns["batch"](t, log_hash=True) if "batch" in ns else ns["get_batch"](log_hash=True))
    except TypeError:
        for t in range(1, 6): ns["get_batch"](log_hash=True)
got_h = ns["batch_hashes"][:5]
res["batch_hashes"] = {"got": got_h, "recorded": meta.get("batch_hashes_first5"),
                       "equal": got_h == meta.get("batch_hashes_first5")}
# (iv) full fp16 snapshot at step 0
fs = meta.get("full_snapshots") or {}
rec0 = [s for s in fs.get("steps", []) if s["step"] == 0]
if rec0:
    sd16 = {k: (v.detach().to(torch.float16).cpu() if v.is_floating_point() else v.detach().cpu())
            for k, v in model.state_dict().items()}
    os.makedirs(os.path.join(tmp, "full"), exist_ok=True); fp = os.path.join(tmp, "full", "step0.pt"); torch.save(sd16, fp)  # same file name as the trainer (the zip archive root is named after it)
    res["full_snapshot_step0"] = {"got_sha256_16": hashlib.sha256(open(fp, "rb").read()).hexdigest()[:16],
                                  "recorded_sha256_16": rec0[0]["sha256_16"]}
    res["full_snapshot_step0"]["equal"] = res["full_snapshot_step0"]["got_sha256_16"] == rec0[0]["sha256_16"]
    res["full_snapshot_step0"]["n_tensors"] = len(sd16)
res["env"] = {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "device": "cpu"}
json.dump(res, open(out_p, "w"), indent=1)
print(json.dumps(res, indent=1))
