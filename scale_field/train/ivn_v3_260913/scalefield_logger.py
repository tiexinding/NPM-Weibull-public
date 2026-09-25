"""Scale-field generation-chain logger (D-item, P5_机理探究_尺度场框架_260911 §6).

Read-only side-car for ivn_train_v2.py: records, in windows of `win` consecutive steps
opened every `every` steps, the per-row / per-column scale statistics of the chain

    x (input activations) , delta (output-side error) -> G = delta^T x
    -> m_hat, sqrt(v_hat) -> u = m_hat/(sqrt(v_hat)+eps) -> applied update -> Delta h_row, Delta h_col

for every analysis matrix W[out, in] (q,k,v,o,gate,up,down x layers).

Hard requirement: the logger never modifies the training trajectory. It only reads
activations (forward hook), grad_output (backward hook), .grad, optimizer state and
parameters; it consumes no RNG.  `--selftest_logger` in ivn_train_v2.py verifies this
bitwise.

Conventions (PyTorch Linear: y = x W^T, W[out, in]):
  rows  = output units  (index i)   -> delta_i, m_hat/v_hat/u/W rows
  cols  = input units   (index j)   -> x_j, columns of the same tensors
  G = dL/dW,  G_ij = sum_tokens delta_i x_j   (so ln RMS_j(x) is the column-side driver,
                                                ln RMS_i(delta) the row-side driver)

Window file: {out}/logger/win{t_start}.npz, keys "{param_name}|{quantity}", float32 arrays.
Per-window quantities (row-vectors of length out or col-vectors of length in):
  x_col_lnrms, delta_row_lnrms                       mean over window steps of ln RMS
  G_row_lnrms, G_col_lnrms
  mhat_row_lnrms, mhat_col_lnrms, sqrtvhat_row_lnrms, sqrtvhat_col_lnrms
  u_row_lnrms, u_col_lnrms                            u = m_hat/(sqrt(v_hat)+eps) (standard Adam)
  uapp_row_lnrms, uapp_col_lnrms                      u_applied = (W_pre(1-lr*wd) - W_post)/lr  (exact, any arm)
  u_row_radial, u_col_radial                          lr-weighted window sum of signed radial projection
                                                        sum_t lr_t * sum_j u_ij W_ij / ||W_i||   (row i), same for cols
  uapp_row_radial, uapp_col_radial                    same with the applied direction
  G_row_signagree, G_col_signagree                    mean_j of [fraction of steps whose sign(G_ij)
                                                        equals sign of the window sum of G_ij]
  Wnorm_row_start, Wnorm_row_end, Wnorm_col_start, Wnorm_col_end   ||W_i||, ||W_j|| at window edges
  h_row_start, h_row_end, h_col_start, h_col_end      ln RMS - median (the field), window edges
Scalars: "__meta__|t_start", "|t_end", "|n_steps", "|lr_sum", "|lwd".
"""
import os
import numpy as np
import torch


def _lnrms_rows(M):   # M: (out, in) float64 -> (out,)
    return torch.log(M.pow(2).mean(1).clamp_min(1e-300)) * 0.5


def _lnrms_cols(M):
    return torch.log(M.pow(2).mean(0).clamp_min(1e-300)) * 0.5


def _field(lnrms):
    return lnrms - lnrms.median()


class ScaleFieldLogger:
    def __init__(self, model, ana_names, pmap, opt, out_dir, every=200, win=20,
                 beta1=0.9, beta2=0.999, eps=1e-8, lwd=0.1, total_steps=None):
        self.model, self.ANA, self.pmap, self.opt = model, list(ana_names), pmap, opt
        self.dir = os.path.join(out_dir, "logger"); os.makedirs(self.dir, exist_ok=True)
        self.every, self.win = int(every), int(win)
        self.b1, self.b2, self.eps, self.lwd = beta1, beta2, eps, lwd
        self.T = total_steps
        # module <-> param name map (Linear modules whose .weight is an analysis matrix)
        self.mod_of = {}
        for mname, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and (mname + ".weight") in self.pmap and (mname + ".weight") in self.ANA:
                self.mod_of[mname + ".weight"] = mod
        assert set(self.mod_of) == set(self.ANA), "every analysis matrix must be a Linear.weight"
        self._handles = []
        for nm, mod in self.mod_of.items():
            self._handles.append(mod.register_forward_hook(self._make_fwd(nm)))
            self._handles.append(mod.register_full_backward_hook(self._make_bwd(nm)))
        self.active = False
        self.acc = None
        self.n_windows = 0

    # ---------- window bookkeeping ----------
    def window_start_for(self, t):
        """If step t opens a window return its start, else None."""
        k, r = divmod(t - 1, self.every)
        return t if r == 0 else None

    def in_window(self, t):
        k, r = divmod(t - 1, self.every)
        return r < self.win

    def _new_acc(self, t_start):
        A = {"t_start": t_start, "t_end": None, "n": 0, "lr_sum": 0.0, "m": {}}
        for nm in self.ANA:
            W = self.pmap[nm].detach().double()
            o, i = W.shape
            z_o = lambda: torch.zeros(o, dtype=torch.float64, device=W.device)
            z_i = lambda: torch.zeros(i, dtype=torch.float64, device=W.device)
            A["m"][nm] = {
                "x_col_lnrms": z_i(), "delta_row_lnrms": z_o(),
                "G_row_lnrms": z_o(), "G_col_lnrms": z_i(),
                "mhat_row_lnrms": z_o(), "mhat_col_lnrms": z_i(),
                "sqrtvhat_row_lnrms": z_o(), "sqrtvhat_col_lnrms": z_i(),
                "u_row_lnrms": z_o(), "u_col_lnrms": z_i(),
                "uapp_row_lnrms": z_o(), "uapp_col_lnrms": z_i(),
                "u_row_radial": z_o(), "u_col_radial": z_i(),
                "uapp_row_radial": z_o(), "uapp_col_radial": z_i(),
                # C1 (260913): 径向分量的跨步自相关。每步的 uapp 径向投影 r_t 除累加外，
                # 另存平方和与 lag-1 交叉积，flush 时算窗内自相关。
                "uapp_row_radial_sq": z_o(), "uapp_col_radial_sq": z_i(),
                "uapp_row_radial_lag1": z_o(), "uapp_col_radial_lag1": z_i(),
                "_prev_row_radial": None, "_prev_col_radial": None, "_n_lag": 0,
                "_first_row_radial": None, "_first_col_radial": None,
                # sign agreement: running sum of G and count of positive signs (fp32 elementwise)
                "G_sum": torch.zeros_like(W, dtype=torch.float32),
                "G_pos": torch.zeros_like(W, dtype=torch.float32),
                "Wnorm_row_start": W.norm(dim=1), "Wnorm_col_start": W.norm(dim=0),
                "h_row_start": _field(_lnrms_rows(W)), "h_col_start": _field(_lnrms_cols(W)),
                # per-step scratch from hooks
                "_x_sumsq": None, "_x_n": 0, "_d_sumsq": None, "_d_n": 0,
            }
        return A

    # ---------- hooks (only read; no-ops outside a window) ----------
    def _make_fwd(self, nm):
        def hook(mod, inputs, output):
            if not self.active: return
            x = inputs[0].detach()
            x2 = x.double().reshape(-1, x.shape[-1])
            m = self.acc["m"][nm]
            s = x2.pow(2).sum(0)
            m["_x_sumsq"] = s if m["_x_sumsq"] is None else m["_x_sumsq"] + s
            m["_x_n"] += x2.shape[0]
        return hook

    def _make_bwd(self, nm):
        def hook(mod, grad_input, grad_output):
            if not self.active: return
            d = grad_output[0].detach()
            d2 = d.double().reshape(-1, d.shape[-1])
            m = self.acc["m"][nm]
            s = d2.pow(2).sum(0)
            m["_d_sumsq"] = s if m["_d_sumsq"] is None else m["_d_sumsq"] + s
            m["_d_n"] += d2.shape[0]
        return hook

    # ---------- training-loop entry points ----------
    def before_forward(self, t):
        ws = self.window_start_for(t)
        if ws is not None:
            self.acc = self._new_acc(ws)
        self.active = self.acc is not None and self.in_window(t)
        if self.active:
            for nm in self.ANA:
                m = self.acc["m"][nm]
                m["_x_sumsq"] = None; m["_x_n"] = 0; m["_d_sumsq"] = None; m["_d_n"] = 0

    @torch.no_grad()
    def after_backward(self, t):
        """Call after loss.backward(), before opt.step(). Reads hooks' scratch and .grad."""
        if not self.active: return
        for nm in self.ANA:
            m = self.acc["m"][nm]
            assert m["_x_sumsq"] is not None and m["_d_sumsq"] is not None, f"hooks did not fire for {nm}"
            m["x_col_lnrms"] += 0.5 * torch.log((m["_x_sumsq"] / m["_x_n"]).clamp_min(1e-300))
            m["delta_row_lnrms"] += 0.5 * torch.log((m["_d_sumsq"] / m["_d_n"]).clamp_min(1e-300))
            G = self.pmap[nm].grad.detach()
            Gd = G.double()
            m["G_row_lnrms"] += _lnrms_rows(Gd); m["G_col_lnrms"] += _lnrms_cols(Gd)
            m["G_sum"] += G.float(); m["G_pos"] += (G > 0).float()

    @torch.no_grad()
    def after_step(self, t, lr_t, W_pre):
        """Call after the parameter update of step t (after any arm-specific rewrite).
        W_pre: dict name -> pre-step weight clone (same as the trainer's W_pre)."""
        if not self.active: return
        A = self.acc; A["n"] += 1; A["lr_sum"] += float(lr_t); A["t_end"] = t
        for nm in self.ANA:
            m = A["m"][nm]; p = self.pmap[nm]
            Wp = W_pre[nm].double(); Wpost = p.detach().double()
            rn = Wp.norm(dim=1).clamp_min(1e-300); cn = Wp.norm(dim=0).clamp_min(1e-300)
            st = self.opt.state.get(p, {})
            if st.get("exp_avg") is not None:
                sv = int(st["step"]) if isinstance(st["step"], torch.Tensor) else st["step"]
                bc1 = 1 - self.b1 ** sv; bc2 = 1 - self.b2 ** sv
                mh = st["exp_avg"].double() / bc1
                sv_ = (st["exp_avg_sq"].double() / bc2).sqrt()
                u = mh / (sv_ + self.eps)
                m["mhat_row_lnrms"] += _lnrms_rows(mh); m["mhat_col_lnrms"] += _lnrms_cols(mh)
                m["sqrtvhat_row_lnrms"] += _lnrms_rows(sv_); m["sqrtvhat_col_lnrms"] += _lnrms_cols(sv_)
                m["u_row_lnrms"] += _lnrms_rows(u); m["u_col_lnrms"] += _lnrms_cols(u)
                m["u_row_radial"] += lr_t * (u * Wp).sum(1) / rn
                m["u_col_radial"] += lr_t * (u * Wp).sum(0) / cn
            # applied direction, exact for every arm: W_post = W_pre(1 - lr*wd) - lr * u_app
            uapp = (Wp * (1.0 - lr_t * self.lwd) - Wpost) / max(lr_t, 1e-300)
            m["uapp_row_lnrms"] += _lnrms_rows(uapp); m["uapp_col_lnrms"] += _lnrms_cols(uapp)
            _rr = lr_t * (uapp * Wp).sum(1) / rn
            _rc = lr_t * (uapp * Wp).sum(0) / cn
            m["uapp_row_radial"] += _rr
            m["uapp_col_radial"] += _rc
            # C1: 平方和与 lag-1 交叉积（用未累加的逐步值）
            m["uapp_row_radial_sq"] += _rr * _rr
            m["uapp_col_radial_sq"] += _rc * _rc
            if m["_prev_row_radial"] is not None:
                m["uapp_row_radial_lag1"] += m["_prev_row_radial"] * _rr
                m["uapp_col_radial_lag1"] += m["_prev_col_radial"] * _rc
                m["_n_lag"] += 1
            if m["_first_row_radial"] is None:
                m["_first_row_radial"] = _rr.clone(); m["_first_col_radial"] = _rc.clone()
            m["_prev_row_radial"] = _rr.clone()
            m["_prev_col_radial"] = _rc.clone()
        # window closes after `win` steps or at the last step
        if A["n"] >= self.win or (self.T is not None and t >= self.T):
            self._flush()

    @torch.no_grad()
    def _flush(self):
        A = self.acc; n = max(A["n"], 1)
        out = {"__meta__|t_start": np.float32(A["t_start"]), "__meta__|t_end": np.float32(A["t_end"]),
               "__meta__|n_steps": np.float32(A["n"]), "__meta__|lr_sum": np.float32(A["lr_sum"]),
               "__meta__|lwd": np.float32(self.lwd)}
        for nm in self.ANA:
            m = A["m"][nm]; W = self.pmap[nm].detach().double()
            for k in ("x_col_lnrms", "delta_row_lnrms", "G_row_lnrms", "G_col_lnrms",
                      "mhat_row_lnrms", "mhat_col_lnrms", "sqrtvhat_row_lnrms", "sqrtvhat_col_lnrms",
                      "u_row_lnrms", "u_col_lnrms", "uapp_row_lnrms", "uapp_col_lnrms"):
                out[f"{nm}|{k}"] = (m[k] / n).float().cpu().numpy()
            for k in ("u_row_radial", "u_col_radial", "uapp_row_radial", "uapp_col_radial",
                      "Wnorm_row_start", "Wnorm_col_start", "h_row_start", "h_col_start"):
                out[f"{nm}|{k}"] = m[k].float().cpu().numpy()
            # sign agreement with the window-sum sign
            pos = m["G_pos"]; Ssum = m["G_sum"]
            agree = torch.where(Ssum > 0, pos / n, 1.0 - pos / n)
            out[f"{nm}|G_row_signagree"] = agree.mean(1).cpu().numpy()
            out[f"{nm}|G_col_signagree"] = agree.mean(0).cpu().numpy()
            out[f"{nm}|Wnorm_row_end"] = W.norm(dim=1).float().cpu().numpy()
            out[f"{nm}|Wnorm_col_end"] = W.norm(dim=0).float().cpu().numpy()
            # C1: 窗内 lag-1 自相关（逐单元）。n_lag = n-1 对交叉积。
            # 标准 lag-1 自相关估计量（同一分母，界在 [-1,1]）：
            #   num = Σ_{t≥2}(x_t-μ)(x_{t-1}-μ) = C1 - μ(2·S1 - x_first - x_last) + (n-1)μ²
            #   den = Σ_t (x_t-μ)² = S2 - n·μ²
            for ax in ("row", "col"):
                s1 = m[f"uapp_{ax}_radial"]; s2 = m[f"uapp_{ax}_radial_sq"]
                c1 = m[f"uapp_{ax}_radial_lag1"]
                xf = m[f"_first_{ax}_radial"]; xl = m[f"_prev_{ax}_radial"]
                if xf is None or xl is None or n < 2:
                    out[f"{nm}|uapp_{ax}_radial_ac1"] = np.zeros(s1.shape[0], dtype=np.float32)
                    out[f"{nm}|uapp_{ax}_radial_sd"] = np.zeros(s1.shape[0], dtype=np.float32)
                    continue
                mu = s1 / n
                num = c1 - mu * (2.0 * s1 - xf - xl) + (n - 1) * mu * mu
                den = s2 - n * mu * mu
                ac = torch.where(den > 0, num / den, torch.zeros_like(den))
                ac = ac.clamp(-1.0, 1.0)
                var = (den / n).clamp_min(0)
                out[f"{nm}|uapp_{ax}_radial_ac1"] = ac.float().cpu().numpy()
                out[f"{nm}|uapp_{ax}_radial_sd"] = var.sqrt().float().cpu().numpy()
            out[f"{nm}|h_row_end"] = _field(_lnrms_rows(W)).float().cpu().numpy()
            out[f"{nm}|h_col_end"] = _field(_lnrms_cols(W)).float().cpu().numpy()
        np.savez_compressed(os.path.join(self.dir, f"win{A['t_start']}.npz"), **out)
        self.n_windows += 1
        self.acc = None; self.active = False

    def close(self):
        if self.acc is not None and self.acc["n"] > 0:
            self._flush()
        for h in self._handles: h.remove()
        self._handles = []

    def meta(self):
        return {"every": self.every, "win": self.win, "n_windows_written": self.n_windows,
                "quantities": ["x_col_lnrms", "delta_row_lnrms", "G_row/col_lnrms", "mhat_row/col_lnrms",
                               "sqrtvhat_row/col_lnrms", "u_row/col_lnrms", "uapp_row/col_lnrms",
                               "u_row/col_radial (lr-weighted sum)", "uapp_row/col_radial",
                               "uapp_row/col_radial_ac1 (C1 260913: 窗内 lag-1 自相关)",
                               "uapp_row/col_radial_sd (C1 260913: 逐步径向的窗内标准差)",
                               "G_row/col_signagree", "Wnorm_*_start/end", "h_row/col_start/end"]}


# ---------------- hook-direction unit test (synthetic) ----------------
def hook_alignment_test(device="cpu", tol=1e-9):
    """Known W, x: check x -> columns, delta -> rows, G = delta^T x via the same hooks."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(5, 3, bias=False).to(device)
    with torch.no_grad():
        lin.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(3, 5) / 10.0)
    x = torch.randn(4, 7, 5, device=device, requires_grad=False)
    x[..., 2] *= 10.0   # column 2 of x is loud -> must show in x_col_lnrms[2]
    got = {}
    def fwd(mod, inp, out): got["x"] = inp[0].detach()
    def bwd(mod, gi, go): got["delta"] = go[0].detach()
    h1 = lin.register_forward_hook(fwd); h2 = lin.register_full_backward_hook(bwd)
    y = lin(x)
    target_w = torch.tensor([1.0, 100.0, 1.0], device=device)  # row 1 error loud
    loss = ((y * target_w) ** 2).sum()
    loss.backward()
    h1.remove(); h2.remove()
    G = lin.weight.grad
    xr = got["x"].reshape(-1, 5); dr = got["delta"].reshape(-1, 3)
    G_re = dr.T @ xr
    ok1 = torch.allclose(G, G_re, rtol=1e-5, atol=1e-6)
    xcol = xr.pow(2).mean(0).sqrt(); drow = dr.pow(2).mean(0).sqrt()
    ok2 = int(torch.argmax(xcol)) == 2 and int(torch.argmax(drow)) == 1
    ok3 = int(torch.argmax(G.pow(2).mean(1))) == 1 and int(torch.argmax(G.pow(2).mean(0))) == 2
    return {"G_equals_deltaT_x": bool(ok1), "x_loud_col_is_2": int(torch.argmax(xcol)),
            "delta_loud_row_is_1": int(torch.argmax(drow)),
            "G_loud_row_is_1_col_is_2": bool(ok3), "pass": bool(ok1 and ok2 and ok3)}
