/* Harmful Brain Activity Classification frontend: hash-routed single page app on top of the FastAPI JSON API. */
(() => {
  "use strict";

  // ------------------------------------------------------------------ helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const app = $("#app");
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v) => (v == null || isNaN(v) ? "N/A" : `${(v * 100).toFixed(2)}%`);
  const num = (v, d = 2) => (v == null || isNaN(v) ? "N/A" : Number(v).toFixed(d));
  const icon = (name) => `<span class="ms" aria-hidden="true">${name}</span>`;
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
  };

  let pending = 0;
  const progress = $("#progress");
  function busy(on) {
    pending += on ? 1 : -1;
    progress.classList.toggle("on", pending > 0);
    progress.style.width = pending > 0 ? "70%" : "100%";
    if (pending <= 0) setTimeout(() => { if (pending <= 0) progress.style.width = "0"; }, 400);
  }

  async function api(path, opts = {}) {
    busy(true);
    try {
      const init = { ...opts, headers: { "X-NeuroScope": "1" } };
      if (opts.json !== undefined) {
        init.method = init.method || "POST";
        init.headers["Content-Type"] = "application/json";
        init.body = JSON.stringify(opts.json);
      }
      const res = await fetch(path, init);
      const isJson = (res.headers.get("content-type") || "").includes("json");
      const body = isJson ? await res.json() : await res.text();
      if (!res.ok) {
        const err = new Error((body && body.detail) || `Request failed (${res.status})`);
        err.status = res.status;
        throw err;
      }
      return body;
    } finally { busy(false); }
  }

  function toast(msg, kind = "info") {
    const icons = { ok: "check_circle", err: "error", info: "info" };
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.innerHTML = `${icon(icons[kind])}<div>${esc(msg)}</div>`;
    $("#toasts").appendChild(el);
    setTimeout(() => el.remove(), kind === "err" ? 7000 : 4000);
  }

  function plot(el, fig, extraLayout = {}) {
    if (!el || !fig || !window.Plotly) return;
    const layout = { ...fig.layout, ...extraLayout, autosize: true };
    delete layout.width;
    Plotly.react(el, fig.data, layout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] });
  }

  async function withButton(btn, fn) {
    const old = btn.innerHTML;
    btn.disabled = true; btn.classList.add("loading");
    const ic = btn.querySelector(".ms"); if (ic) ic.textContent = "progress_activity";
    try { return await fn(); }
    finally { btn.disabled = false; btn.classList.remove("loading"); btn.innerHTML = old; }
  }

  const overlay = (el, on) => {
    let o = el.querySelector(":scope > .loading-overlay");
    if (on && !o) { o = document.createElement("div"); o.className = "loading-overlay"; o.innerHTML = `<div class="spinner"></div>`; el.appendChild(o); }
    if (!on && o) o.remove();
  };

  // ------------------------------------------------------------------ state
  const state = {
    config: null,
    rec: store.get("nsc.rec", "chb01_03"),
    recInfo: null,
    model: null,
    page: null,
  };

  async function setRecording(id, { quiet = false } = {}) {
    const info = await api(`/api/recording/${encodeURIComponent(id)}`);
    state.rec = id; state.recInfo = info; store.set("nsc.rec", id);
    renderStatus();
    if (!quiet) toast(`Loaded ${info.file_name} (${info.duration_sec.toFixed(0)} s, ${info.channels.length} channels)`, "ok");
    return info;
  }

  async function ensureRecording() {
    if (state.recInfo && state.recInfo.id === state.rec) return state.recInfo;
    try { return await setRecording(state.rec, { quiet: true }); }
    catch (e) {
      if (state.rec !== state.config.default_recording) {
        toast("The previous recording is no longer on the server, so the default CHB-MIT recording was loaded.", "info");
        return setRecording(state.config.default_recording, { quiet: true });
      }
      throw e;
    }
  }

  function recBanner(info, context = "Displaying") {
    if (!info) return "";
    const sz = info.seizures.map((s) => `${s.start.toFixed(0)}–${s.end.toFixed(0)} s`).join(", ");
    if (info.kind === "synthetic") return `<div class="banner amber">${icon("science")}<div><strong>${context} a synthetic demonstration recording</strong> (${esc(info.file_name)}). Use <strong>Real</strong> in the sidebar to switch to clinical CHB-MIT data.</div></div>`;
    if (info.kind === "upload") return `<div class="banner sky">${icon("upload_file")}<div><strong>${context} your uploaded recording</strong> <code>${esc(info.file_name)}</code> · first ${info.duration_sec.toFixed(0)} s · no reference annotations.</div></div>`;
    return `<div class="banner green">${icon("verified")}<div><strong>${context} real CHB-MIT scalp EEG:</strong> <code>${esc(info.file_name)}</code> · ${info.sampling_rate.toFixed(0)} Hz · ${info.channels.length} bipolar channels · ${info.duration_sec.toFixed(0)} s${sz ? ` · verified seizure ${sz}` : ""}.</div></div>`;
  }

  // ------------------------------------------------------------------ chrome
  const PAGES = [
    { id: "dashboard", label: "Dashboard", icon: "space_dashboard" },
    { id: "data", label: "EEG Upload & Data", icon: "folder_open" },
    { id: "signals", label: "Signal Explorer", icon: "monitor_heart" },
    { id: "training", label: "Model Training", icon: "model_training" },
    { id: "detection", label: "Seizure Detection", icon: "bolt" },
    { id: "evaluation", label: "Model Evaluation", icon: "analytics" },
    { id: "docs", label: "Documentation", icon: "menu_book" },
  ];

  $("#nav").innerHTML = PAGES.map((p) => `<a href="#/${p.id}" data-page="${p.id}">${icon(p.icon)}<span>${p.label}</span></a>`).join("");

  function renderStatus() {
    const m = state.model, r = state.recInfo;
    const modelCard = m
      ? `<div class="side-card"><div style="margin-bottom:8px"><span class="badge ok">● ${m.is_real ? "Real CHB-MIT model" : "Benchmark model"}</span></div>
          <div class="row-kv"><span>Model</span><span title="${esc(m.name)}">${esc(m.name)}</span></div>
          <div class="row-kv"><span>Threshold</span><span>${num(m.threshold)}</span></div>
          <div class="row-kv"><span>Features</span><span>${m.n_features}</span></div></div>`
      : `<div class="side-card"><span class="badge warn">● Not trained</span></div>`;
    const kindBadge = !r ? "" : r.kind === "real" ? `<span class="badge ok">● Real CHB-MIT data</span>` : r.kind === "upload" ? `<span class="badge info">● Uploaded EDF</span>` : `<span class="badge warn">● Synthetic demo</span>`;
    const recCard = r
      ? `<div class="side-card"><div style="margin-bottom:8px">${kindBadge}</div>
          <div class="row-kv"><span>Recording</span><span title="${esc(r.file_name)}">${esc(r.file_name)}</span></div>
          <div class="row-kv"><span>Patient</span><span>${esc(r.patient_id)}</span></div>
          <div class="row-kv"><span>Duration</span><span>${r.duration_sec.toFixed(1)} s</span></div>
          <div class="row-kv"><span>Channels</span><span>${r.channels.length}</span></div></div>`
      : `<div class="skeleton" style="height:110px;margin-bottom:8px"></div>`;
    $("#side-status").innerHTML = modelCard + recCard;
  }

  $$(".side-actions [data-load]").forEach((b) => b.addEventListener("click", () => withButton(b, async () => {
    try { await setRecording(b.dataset.load); route(true); } catch (e) { toast(e.message, "err"); }
  })));
  $("#menu-btn").addEventListener("click", () => document.body.classList.toggle("nav-open"));
  $("#scrim").addEventListener("click", () => document.body.classList.remove("nav-open"));

  function banner(eyebrow, titleHtml, subtitle, image) {
    return `<section class="page-banner" style="background-image:url('/images/${image}.jpg')">
      <span class="eyebrow"><span class="dot"></span>${esc(eyebrow)}</span><h1>${titleHtml}</h1><p>${subtitle}</p></section>`;
  }
  const sectionLabel = (t) => `<div class="section-label">${esc(t)}</div>`;
  const metric = (label, value, sub = "") => `<div class="card metric"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="sub">${esc(sub)}</div></div>`;
  const segmented = (name, options, active) => `<div class="segmented" data-seg="${name}">${options.map((o) => `<button type="button" data-value="${esc(o.value)}" class="${o.value === active ? "on" : ""}">${o.icon ? icon(o.icon) : ""}${esc(o.label)}</button>`).join("")}</div>`;
  function bindSegmented(root, name, onChange) {
    const seg = $(`[data-seg="${name}"]`, root);
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      $$("button", seg).forEach((x) => x.classList.toggle("on", x === b));
      onChange(b.dataset.value);
    });
  }
  const readOnlyNote = () => state.config.public_demo
    ? `<div class="banner sky">${icon("lock")}<div><strong>Public demo:</strong> retraining is turned off so the shared model and results stay intact for every visitor. Run the project locally (START_PROJECT.bat) to retrain.</div></div>` : "";

  // ------------------------------------------------------------------ pages
  const pages = {};

  pages.dashboard = async (t) => {
    app.innerHTML = `<div class="skeleton" style="height:380px;margin-bottom:26px"></div><div class="grid g4">${'<div class="skeleton" style="height:120px"></div>'.repeat(4)}</div>`;
    await ensureRecording();
    if (isStale(t)) return;
    const d = await api(`/api/overview?rec=${encodeURIComponent(state.rec)}`);
    state.model = d.model; renderStatus();
    if (isStale(t)) return;
    const c = d.cards, m = d.model, ro = state.config.public_demo;
    const steps = [
      ["01", "Load & verify", "MNE reads the EDF; summary and WFDB annotations are cross-checked."],
      ["02", "Clean the signal", "0.5–40 Hz band-pass and 60 Hz notch filter on every channel."],
      ["03", "Window & featurize", "5-second windows become time-frequency band-power features."],
      ["04", "Train & calibrate", "Balanced Random Forest with a validation-tuned threshold."],
      ["05", "Score & evaluate", "Window and event metrics on a held-out, unseen patient."],
    ];
    app.innerHTML = `<div class="page">
      <section class="hero" style="background-image:url('/images/hero.jpg')">
        <span class="eyebrow"><span class="dot"></span>EEG seizure detection · research prototype</span>
        <h1>Reading the <em>electric language</em> of the brain.</h1>
        <p class="lede">This system classifies harmful brain activity and flags seizure episodes in continuous scalp EEG, trained and evaluated on the CHB-MIT pediatric database from PhysioNet.</p>
        <div class="chip-row">
          <span class="chip">Montage <b>18 bipolar ch</b></span><span class="chip">Sampling <b>${d.system.sampling_rate.toFixed(0)} Hz</b></span>
          <span class="chip">Window <b>5 s</b></span><span class="chip">Model <b>${m ? esc(m.name.replace(/_/g, " ")) : "Not trained"}</b></span>
        </div>
        <div class="row mt"><a class="btn primary" href="#/detection">${icon("bolt")}Run seizure detection</a><a class="btn" href="#/signals">${icon("monitor_heart")}Explore signals</a></div>
      </section>
      <div class="grid g4">
        ${metric("PhysioNet records", d.records_indexed, "Seizure records indexed")}
        ${metric("Test accuracy", c ? pct(c.accuracy) : "N/A", c ? `Unseen patient ${c.test_patient} (${c.test_recording})` : "Model not evaluated")}
        ${metric("Seizure sensitivity", c ? pct(c.sensitivity) : "N/A", c ? `Seizure recall (${c.tp}/${c.tp + c.fn} windows)` : "")}
        ${metric("Specificity", c ? pct(c.specificity) : "N/A", c ? `Non-seizure windows (${c.tn}/${c.tn + c.fp})` : "")}
      </div>
      <div class="stack mt">${recBanner(d.recording, "Active dataset:")}<div class="banner rose">${icon("health_and_safety")}<div>${esc(state.config.disclaimer)}</div></div></div>
      ${sectionLabel("The pipeline")}
      <div class="grid g5">${steps.map(([n, h, p]) => `<div class="feature"><div class="num">${n}</div><h4>${h}</h4><p>${p}</p></div>`).join("")}</div>
      ${sectionLabel("Experiment runner")}
      <div class="card"><div class="row spread">
        <div style="flex:1;min-width:260px"><h4 style="margin:0 0 4px">Run the real CHB-MIT experiment end to end</h4>
          <p>Loads <code>chb01_03.edf</code>, verifies annotations, filters, segments, trains the balanced Random Forest with a strict patient-level holdout, and writes reproducible artifacts to <code>results/</code>.</p></div>
        <button class="btn primary" id="run-exp" ${ro ? "disabled" : ""}>${icon("rocket_launch")}Run experiment</button>
      </div>${ro ? `<div class="mt">${readOnlyNote()}</div>` : ""}</div>
      ${sectionLabel("System overview")}
      <div class="grid g2">
        <div class="card"><h4>${icon("database")}Dataset status · CHB-MIT</h4>
          <div class="row-kv"><span>Local disk</span><span>${d.system.free_gb} / ${d.system.total_gb} GB free</span></div>
          <div class="row-kv"><span>Annotation summaries</span><span>${d.system.annotation_summaries}</span></div>
          <div class="row-kv"><span>Raw EDF storage</span><span>data/raw/</span></div>
          <div class="row-kv"><span>Montage</span><span>18 bipolar · 10-20</span></div>
          <div class="row-kv"><span>Verified recording</span><span>chb01_03.edf · 2996–3036 s</span></div></div>
        <div class="card"><h4>${icon("psychology")}Model & evaluation</h4>${m ? `
          <div class="row-kv"><span>Architecture</span><span>${esc(m.name.toUpperCase())}</span></div>
          <div class="row-kv"><span>Features</span><span>${m.n_features} time-frequency</span></div>
          <div class="row-kv"><span>Decision threshold</span><span>${num(m.threshold)} (val-tuned)</span></div>
          <div class="row-kv"><span>Train</span><span>${esc(m.train_patients.join(", "))}</span></div>
          <div class="row-kv"><span>Validation</span><span>${esc(m.val_patients.join(", "))}</span></div>
          <div class="row-kv"><span>Test</span><span>${esc(m.test_patients.join(", "))}</span></div>` : `<p>No trained model is loaded yet.</p>`}</div>
      </div>
      ${sectionLabel("Current recording")}
      <div class="chart"><div class="plot" id="preview"></div></div>
    </div>`;
    plot($("#preview"), d.preview);
    const btn = $("#run-exp");
    if (!ro) btn.addEventListener("click", () => trainReal(btn));
  };

  async function trainReal(btn) {
    await withButton(btn, async () => {
      toast("Training on chb01_03 / chb01_04, testing on unseen patient chb02. This takes about 20 seconds.", "info");
      try {
        const r = await api("/api/train/real", { method: "POST" });
        state.model = r.model; renderStatus();
        toast("Experiment complete. Model and artifacts updated in results/.", "ok");
        route(true);
      } catch (e) { toast(e.message, "err"); }
    });
  }

  pages.data = async (t) => {
    const d = await api("/api/data");
    if (isStale(t)) return;
    const ro = state.config.public_demo;
    app.innerHTML = `<div class="page">
      ${banner("Data manager", "EEG data &amp; <em>EDF upload</em>", "Inspect the bundled PhysioNet CHB-MIT recordings, upload your own .edf file, or load synthetic benchmark recordings.", "data")}
      ${segmented("tab", [{ value: "chb", label: "CHB-MIT recordings", icon: "local_hospital" }, { value: "upload", label: "Upload EDF", icon: "upload_file" }, { value: "demo", label: "Benchmark demos", icon: "science" }], "chb")}
      <div class="mt" id="tab-chb">
        <div class="grid g3">${d.bundled.map((b) => `
          <div class="card"><div class="row spread" style="margin-bottom:8px"><h4 style="margin:0">${icon("description")}${esc(b.file_name)}</h4><span class="badge ${b.role === "Test" ? "info" : "ok"}">${b.role}</span></div>
            <div class="row-kv"><span>Status</span><span>${b.present ? "Present" : "Missing"}</span></div>
            <div class="row-kv"><span>Size</span><span>${b.size_mb ?? "–"} MB</span></div>
            <div class="row-kv"><span>Patient</span><span>${esc(b.note)}</span></div>
            <div class="row-kv"><span>Annotations</span><span>${b.has_summary ? "summary" : "–"}${b.has_seizure_file ? " + WFDB" : ""}</span></div>
            <button class="btn ${state.rec === b.id ? "" : "primary"} block mt" data-rec="${b.id}" ${b.present ? "" : "disabled"}>${icon(state.rec === b.id ? "check" : "download_for_offline")}${state.rec === b.id ? "Loaded" : "Load into workspace"}</button>
          </div>`).join("")}</div>
        <div class="banner sky mt">${icon("link")}<div>Source: <a href="https://physionet.org/content/chbmit/1.0.0/" target="_blank" rel="noopener">PhysioNet CHB-MIT Scalp EEG Database v1.0.0</a> (ODC-PDDL). Train <code>chb01_03</code>, validate <code>chb01_04</code>, test on unseen patient <code>chb02_16</code>.</div></div>
        ${sectionLabel("PhysioNet catalog")}
        <div class="grid g2">
          <div class="card"><h4>${icon("list_alt")}Records with seizures · ${d.records.length}</h4>
            <div class="table-wrap"><table><thead><tr><th>#</th><th>Relative path</th></tr></thead><tbody>${d.records.slice(0, 200).map((r, i) => `<tr><td class="mono">${i + 1}</td><td class="mono">${esc(r)}</td></tr>`).join("") || `<tr><td colspan="2" class="muted">No cached catalog.</td></tr>`}</tbody></table></div></div>
          <div class="card"><h4>${icon("summarize")}Patient summaries</h4>
            <p style="margin-bottom:12px">Summaries list every recording with its seizure start and end times. Cached summaries are used for annotation cross-checks.</p>
            <div class="table-wrap"><table><thead><tr><th>Patient</th><th>Status</th><th></th></tr></thead><tbody>${d.summary_patients.map((p) => `<tr><td class="mono">${p.id}</td><td>${p.cached ? `<span class="badge ok">Cached</span>` : `<span class="badge warn">Not cached</span>`}</td><td style="text-align:right">${p.cached ? "" : `<button class="btn sm" data-summary="${p.id}" ${ro ? "disabled title='Disabled on the public demo'" : ""}>${icon("cloud_download")}Fetch</button>`}</td></tr>`).join("")}</tbody></table></div></div>
        </div>
      </div>
      <div class="mt hidden" id="tab-upload">
        <label class="dropzone" id="drop"><input type="file" accept=".edf" id="file" hidden>
          ${icon("cloud_upload")}<h4>Drop an .edf recording here or click to browse</h4>
          <p class="muted">Up to ${state.config.upload_limit_mb} MB. The first ${state.config.upload_max_seconds.toFixed(0)} s are loaded, channels are matched to the 18-channel bipolar montage.</p></label>
        <div id="upload-result" class="mt"></div>
      </div>
      <div class="mt hidden" id="tab-demo">
        <div class="card"><h4>${icon("science")}Pre-configured demonstration recordings</h4>
          <p style="margin-bottom:14px">Multichannel EEG with realistic background rhythms and injected 3 Hz spike-and-wave discharges. Clearly labelled synthetic.</p>
          <div class="row"><select id="demo-select" style="max-width:420px">${d.demos.map((x) => `<option value="${x.id}">${esc(x.label)}</option>`).join("")}</select>
          <button class="btn primary" id="demo-load">${icon("play_arrow")}Load demo record</button></div></div>
      </div></div>`;

    bindSegmented(app, "tab", (v) => ["chb", "upload", "demo"].forEach((t) => $(`#tab-${t}`).classList.toggle("hidden", t !== v)));
    $$("[data-rec]").forEach((b) => b.addEventListener("click", () => withButton(b, async () => {
      try { await setRecording(b.dataset.rec); pages.data(); } catch (e) { toast(e.message, "err"); }
    })));
    $$("[data-summary]").forEach((b) => b.addEventListener("click", () => withButton(b, async () => {
      try { const r = await api(`/api/summary/${b.dataset.summary}`, { method: "POST" }); toast(`Summary available: ${r.file}`, "ok"); pages.data(); }
      catch (e) { toast(e.message, "err"); }
    })));
    $("#demo-load").addEventListener("click", (e) => withButton(e.currentTarget, async () => {
      try { await setRecording($("#demo-select").value); } catch (err) { toast(err.message, "err"); }
    }));

    const drop = $("#drop"), file = $("#file");
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
    drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
    file.addEventListener("change", () => file.files[0] && upload(file.files[0]));
  };

  async function upload(f) {
    const out = $("#upload-result");
    if (!f.name.toLowerCase().endsWith(".edf")) { toast("Please choose an .edf file.", "err"); return; }
    if (f.size > state.config.upload_limit_mb * 1048576) { toast(`File is larger than ${state.config.upload_limit_mb} MB.`, "err"); return; }
    out.innerHTML = `<div class="card row">${'<div class="spinner"></div>'}<div>Uploading and validating <strong>${esc(f.name)}</strong> (${(f.size / 1048576).toFixed(1)} MB) with MNE-Python…</div></div>`;
    const fd = new FormData(); fd.append("file", f);
    try {
      const info = await api("/api/upload", { method: "POST", body: fd });
      state.rec = info.id; state.recInfo = info; store.set("nsc.rec", info.id); renderStatus();
      out.innerHTML = `<div class="banner green">${icon("check_circle")}<div>Loaded <strong>${esc(info.file_name)}</strong> into the workspace.</div></div>
        <div class="grid g3 mt">${metric("Sampling rate", `${info.sampling_rate.toFixed(1)} Hz`)}${metric("Duration", `${info.duration_sec.toFixed(1)} s`)}${metric("Channels", info.channels.length)}</div>
        ${info.warnings.map((w) => `<div class="banner amber mt">${icon("warning")}<div>${esc(w)}</div></div>`).join("")}
        <div class="row mt"><a class="btn primary" href="#/detection">${icon("bolt")}Run detection on it</a><a class="btn" href="#/signals">${icon("monitor_heart")}View signals</a></div>`;
      toast(`Loaded ${info.file_name}`, "ok");
    } catch (e) {
      out.innerHTML = `<div class="banner rose">${icon("error")}<div>${esc(e.message)}</div></div>`;
    }
  }

  pages.signals = async (t) => {
    const info = await ensureRecording();
    if (isStale(t)) return;
    const ex = store.get(`nsc.ex.${info.id}`, null) || {
      channels: info.channels.slice(0, Math.min(10, info.channels.length)),
      span: 10, start: info.kind === "real" && info.duration_sec >= 3000 ? 2990 : 0, filtered: true, psd: null,
    };
    ex.channels = ex.channels.filter((c) => info.channels.includes(c));
    if (!ex.channels.length) ex.channels = info.channels.slice(0, 10);
    const maxStart = () => Math.max(0, Math.floor(info.duration_sec - ex.span));
    const jumps = info.seizures.length
      ? [{ t: Math.max(0, info.seizures[0].start - 1), label: `Seizure (${info.seizures[0].start.toFixed(0)}–${info.seizures[0].end.toFixed(0)} s)`, ic: "bolt" },
         { t: Math.max(0, info.seizures[0].start - 16), label: "Pre-ictal", ic: "fast_rewind" },
         { t: info.seizures[0].end + 4, label: "Post-ictal", ic: "fast_forward" },
         { t: 0, label: "Recording start", ic: "first_page" }]
      : [{ t: 0, label: "Recording start", ic: "first_page" }];

    app.innerHTML = `<div class="page">
      ${banner("Signal explorer", "Multichannel <em>signal explorer</em>", "Review raw and filtered EEG traces with the ground-truth seizure overlay, and inspect power spectral density across clinical bands.", "signals")}
      ${recBanner(info)}
      <div class="row mt">${jumps.map((j) => `<button class="btn sm" data-jump="${j.t}">${icon(j.ic)}${esc(j.label)}</button>`).join("")}</div>
      <div class="card mt"><div class="grid g3">
        <div class="stack" style="gap:6px"><div class="range-head"><span>Start time</span><b id="start-val"></b></div>
          <input type="range" id="start" min="0" step="1"><div class="row" style="gap:6px"><button class="btn sm" data-step="-1">${icon("chevron_left")}Back</button><button class="btn sm" data-step="1">Forward${icon("chevron_right")}</button></div></div>
        <label class="field">Window span<select id="span">${[5, 10, 15, 20, 30].map((s) => `<option value="${s}">${s} seconds</option>`).join("")}</select></label>
        <div class="stack" style="gap:10px;justify-content:center"><label class="toggle"><input type="checkbox" id="filter"><span class="track"></span>0.5–40 Hz band-pass + 60 Hz notch</label></div>
      </div>
      <div class="mt"><div class="range-head" style="margin-bottom:8px"><span>Channels · <span id="ch-count"></span></span><span><button class="btn sm" id="ch-all">All</button> <button class="btn sm" id="ch-none">Clear</button></span></div>
        <div class="chips" id="chips">${info.channels.map((c) => `<button class="chip-toggle" data-ch="${esc(c)}">${esc(c)}</button>`).join("")}</div></div></div>
      <div class="chart mt" id="eeg-wrap"><div class="plot" id="eeg"></div></div>
      ${sectionLabel("Spectral power density (Welch)")}
      <div class="grid" style="grid-template-columns:minmax(200px,260px) minmax(0,1fr)" id="psd-grid">
        <div class="card"><label class="field">Channel<select id="psd-ch"></select></label><p class="mt">Welch PSD over the full recording for the selected channel, with delta, theta, alpha, beta and gamma bands shaded.</p></div>
        <div class="chart" id="psd-wrap"><div class="plot" id="psd"></div></div>
      </div></div>`;
    if (window.matchMedia("(max-width: 720px)").matches) $("#psd-grid").style.gridTemplateColumns = "minmax(0,1fr)";

    const startEl = $("#start"), spanEl = $("#span"), filterEl = $("#filter");
    const sync = () => {
      startEl.max = maxStart(); ex.start = Math.min(ex.start, maxStart()); startEl.value = ex.start;
      $("#start-val").textContent = `${ex.start.toFixed(0)} – ${(ex.start + ex.span).toFixed(0)} s`;
      spanEl.value = ex.span; filterEl.checked = ex.filtered;
      $$("#chips [data-ch]").forEach((b) => b.classList.toggle("on", ex.channels.includes(b.dataset.ch)));
      $("#ch-count").textContent = `${ex.channels.length} of ${info.channels.length} shown`;
      const psdSel = $("#psd-ch");
      if (!ex.psd || !ex.channels.includes(ex.psd)) ex.psd = ex.channels[0];
      psdSel.innerHTML = ex.channels.map((c) => `<option ${c === ex.psd ? "selected" : ""}>${esc(c)}</option>`).join("");
      store.set(`nsc.ex.${info.id}`, ex);
    };
    let seq = 0, timer = null;
    const draw = (delay = 0) => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        sync();
        if (!ex.channels.length) { $("#eeg").innerHTML = `<div class="empty">${icon("touch_app")}Select at least one channel.</div>`; return; }
        const my = ++seq;
        overlay($("#eeg-wrap"), true); overlay($("#psd-wrap"), true);
        try {
          const q = new URLSearchParams({ rec: info.id, start: ex.start, span: ex.span, channels: ex.channels.join(","), filtered: ex.filtered, psd_channel: ex.psd });
          const r = await api(`/api/explore?${q}`);
          if (my !== seq || isStale(t)) return;
          $("#eeg").innerHTML = "";
          plot($("#eeg"), r.eeg); plot($("#psd"), r.psd);
        } catch (e) { if (!isStale(t)) toast(e.message, "err"); }
        finally { if (my === seq && !isStale(t)) { overlay($("#eeg-wrap"), false); overlay($("#psd-wrap"), false); } }
      }, delay);
    };
    startEl.addEventListener("input", () => { ex.start = +startEl.value; $("#start-val").textContent = `${ex.start} – ${ex.start + ex.span} s`; draw(250); });
    spanEl.addEventListener("change", () => { ex.span = +spanEl.value; draw(); });
    filterEl.addEventListener("change", () => { ex.filtered = filterEl.checked; draw(); });
    $$("[data-jump]").forEach((b) => b.addEventListener("click", () => { ex.start = Math.min(+b.dataset.jump, maxStart()); draw(); }));
    $$("[data-step]").forEach((b) => b.addEventListener("click", () => { ex.start = Math.max(0, Math.min(maxStart(), ex.start + +b.dataset.step * ex.span)); draw(); }));
    $("#chips").addEventListener("click", (e) => {
      const b = e.target.closest("[data-ch]"); if (!b) return;
      const c = b.dataset.ch;
      ex.channels = ex.channels.includes(c) ? ex.channels.filter((x) => x !== c) : info.channels.filter((x) => x === c || ex.channels.includes(x));
      draw(150);
    });
    $("#ch-all").addEventListener("click", () => { ex.channels = [...info.channels]; draw(); });
    $("#ch-none").addEventListener("click", () => { ex.channels = []; draw(); });
    $("#psd-ch").addEventListener("change", (e) => { ex.psd = e.target.value; draw(); });
    draw();
  };

  pages.training = async (t) => {
    const ro = state.config.public_demo;
    const classifiers = await api("/api/classifiers");
    if (isStale(t)) return;
    app.innerHTML = `<div class="page">
      ${banner("Model training", "Training &amp; <em>cross-validation</em>", "Train baseline classifiers on the real CHB-MIT Scalp EEG recordings or on the multi-patient synthetic benchmark generator.", "training")}
      ${readOnlyNote()}
      <div class="mt">${segmented("tab", [{ value: "real", label: "Real CHB-MIT pipeline", icon: "local_hospital" }, { value: "synthetic", label: "Synthetic multi-patient generator", icon: "science" }], "real")}</div>
      <div class="mt" id="tab-real">
        <div class="grid g3">
          <div class="card"><span class="badge ok">Train</span><h4 class="mt">chb01_03.edf</h4><p>Patient chb01 · 3,050 s · 8 seizure windows, 599 non-seizure windows</p></div>
          <div class="card"><span class="badge warn">Validation</span><h4 class="mt">chb01_04.edf</h4><p>Patient chb01 · 1,510 s · threshold tuned here, never on test</p></div>
          <div class="card"><span class="badge info">Test</span><h4 class="mt">chb02_16.edf</h4><p>Unseen patient chb02 · 196 s · 13 seizure, 25 non-seizure windows</p></div>
        </div>
        <div class="banner rose mt">${icon("shield")}<div><strong>Anti-leakage protocol.</strong> Test data comes from patient <code>chb02</code>, unseen during training or validation. No recording or seizure event is split across partitions. Feature scaling and model fitting happen on the training partition only.</div></div>
        <div class="grid g2 mt">
          <div class="card"><h4>${icon("tune")}Configuration</h4>
            <div class="row-kv"><span>Windows</span><span>5 s, non-overlapping, 5 s boundary buffer</span></div>
            <div class="row-kv"><span>Classifier</span><span>RandomForest(n=100, balanced)</span></div>
            <div class="row-kv"><span>Artifacts</span><span>results/</span></div></div>
          <div class="card"><h4>${icon("target")}Evaluation targets</h4>
            <p>Window-level sensitivity, specificity, precision, F1 and ROC-AUC; event-level sensitivity and false alarms per hour; confusion matrix and seizure timeline.</p></div>
        </div>
        <div class="row mt"><button class="btn primary" id="train-real" ${ro ? "disabled" : ""}>${icon("rocket_launch")}Train real CHB-MIT baseline model</button></div>
        <div id="train-real-out" class="mt"></div>
      </div>
      <div class="mt hidden" id="tab-synthetic">
        <div class="card"><div class="grid g3">
          <label class="field">Classifier<select id="s-model">${Object.entries(classifiers).map(([k, v]) => `<option value="${esc(k)}">${esc(v)}</option>`).join("")}</select></label>
          <label class="field">Window duration (s)<input type="number" id="s-win" value="5" min="1" max="20" step="1"></label>
          <label class="field">Window step (s)<input type="number" id="s-step" value="5" min="1" max="20" step="1"></label>
          <label class="field">Boundary buffer (s)<input type="number" id="s-buf" value="5" min="0" max="15" step="1"></label>
          <label class="field">Test patient ratio<input type="number" id="s-test" value="0.33" min="0.2" max="0.5" step="0.05"></label>
          <label class="field">Validation patient ratio<input type="number" id="s-val" value="0.33" min="0.2" max="0.5" step="0.05"></label>
          <label class="field">Random seed<input type="number" id="s-seed" value="42" step="1"></label>
        </div>
        <div class="row mt"><button class="btn primary" id="train-syn" ${ro ? "disabled" : ""}>${icon("play_arrow")}Start synthetic benchmark pipeline</button></div></div>
        <div id="train-syn-out" class="mt"></div>
      </div></div>`;
    bindSegmented(app, "tab", (v) => ["real", "synthetic"].forEach((t) => $(`#tab-${t}`).classList.toggle("hidden", t !== v)));
    if (ro) return;
    $("#train-real").addEventListener("click", (e) => withButton(e.currentTarget, async () => {
      try {
        toast("Training the patient-level model. This takes about 20 seconds.", "info");
        const r = await api("/api/train/real", { method: "POST" });
        state.model = r.model; renderStatus();
        const t = r.metrics && r.metrics.test_metrics;
        $("#train-real-out").innerHTML = t ? `<div class="banner green">${icon("check_circle")}<div>Model trained and saved. Test results on unseen patient chb02:</div></div>
          <div class="grid g4 mt">${metric("Test accuracy", pct(t.accuracy))}${metric("Sensitivity", pct(t.sensitivity))}${metric("Specificity", pct(t.specificity))}${metric("F1 score", pct(t.f1_score))}</div>` : "";
        toast("Training complete.", "ok");
      } catch (err) { toast(err.message, "err"); }
    }));
    $("#train-syn").addEventListener("click", (e) => withButton(e.currentTarget, async () => {
      try {
        const body = { model_type: $("#s-model").value, window_sec: +$("#s-win").value, step_sec: +$("#s-step").value, buffer_sec: +$("#s-buf").value, test_ratio: +$("#s-test").value, val_ratio: +$("#s-val").value, seed: +$("#s-seed").value };
        const r = await api("/api/train/synthetic", { json: body });
        state.model = r.model; renderStatus();
        $("#train-syn-out").innerHTML = `<div class="banner green">${icon("check_circle")}<div>Trained benchmark model <strong>${esc(r.model.name)}</strong> on ${r.windows} windows and saved it to <code>${esc(r.saved_to)}/</code>. It is now the active model.</div></div>`;
        toast("Synthetic benchmark training complete.", "ok");
      } catch (err) { toast(err.message, "err"); }
    }));
  };

  pages.detection = async (t) => {
    const info = await ensureRecording();
    if (!state.model) state.model = await api("/api/model");
    if (isStale(t)) return;
    const m = state.model;
    if (!m) { app.innerHTML = `<div class="page">${banner("Detection", "Continuous <em>seizure detection</em>", "", "detection")}<div class="empty">${icon("model_training")}No trained model. Train one on the <a href="#/training">Model Training</a> page.</div></div>`; return; }
    let threshold = store.get("nsc.threshold", m.threshold);
    app.innerHTML = `<div class="page">
      ${banner("Detection", "Continuous <em>seizure detection</em>", "Window-by-window seizure risk inference on the active recording. Scores are model risk outputs compared against the calibrated decision threshold.", "detection")}
      ${recBanner(info, "Running inference on")}
      <div class="card mt"><div class="grid" style="grid-template-columns:minmax(0,1fr) auto;align-items:end">
        <div class="stack" style="gap:6px"><div class="range-head"><span>Decision threshold · windows with risk ≥ threshold are classified as seizure</span><b id="thr-val"></b></div>
          <input type="range" id="thr" min="0.05" max="0.95" step="0.05"></div>
        <button class="btn sm" id="thr-reset">${icon("restart_alt")}Calibrated (${num(m.threshold)})</button>
      </div></div>
      <div id="det-out" class="mt"><div class="card row">${'<div class="spinner"></div>'}<div><strong>Analysing ${esc(info.file_name)}…</strong><br><span class="muted">Filtering, segmenting into 5 s windows, extracting features and scoring. The first run on a long recording takes a few seconds; threshold changes are instant after that.</span></div></div></div>
    </div>`;
    const thr = $("#thr");
    const setThr = (v) => { threshold = Math.round(v * 100) / 100; thr.value = threshold; $("#thr-val").textContent = threshold.toFixed(2); store.set("nsc.threshold", threshold); };
    setThr(threshold);
    let seq = 0, timer = null;
    const run = (delay = 0) => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        const my = ++seq;
        const out = $("#det-out");
        if (out.querySelector(".chart")) overlay(out, true);
        try {
          const r = await api("/api/detect", { json: { rec: info.id, threshold } });
          if (my !== seq || isStale(t)) return;
          const hasTruth = r.recording.seizures.length > 0;
          out.style.position = "relative";
          out.innerHTML = `
            <div class="grid g4">${metric("Analysed windows", r.windows, "5-second windows")}${metric("Detected episodes", r.num_episodes, "Clustered positive windows")}${metric("Total seizure time", `${r.total_seizure_sec.toFixed(1)} s`)}${metric("Active threshold", r.threshold.toFixed(2), `Calibrated ${r.calibrated_threshold.toFixed(2)}`)}</div>
            ${r.warnings.map((w) => `<div class="banner amber mt">${icon("warning")}<div>${esc(w)}</div></div>`).join("")}
            <div class="chart mt">${r.timeline ? `<div class="plot" id="timeline"></div>` : `<div class="empty">${icon("timer_off")}Recording too short for any 5 s window.</div>`}</div>
            ${sectionLabel("Detected episodes & clinical alignment")}
            ${r.episodes.length ? `<div class="table-wrap"><table><thead><tr><th>#</th><th>Onset</th><th>Offset</th><th>Duration</th><th>Peak score</th><th>Mean score</th><th>Ground truth</th></tr></thead><tbody>
              ${r.episodes.map((e) => `<tr><td class="mono">${e.id}</td><td class="mono">${e.start.toFixed(1)} s</td><td class="mono">${e.end.toFixed(1)} s</td><td class="mono">${e.duration.toFixed(1)} s</td><td class="mono">${e.max_score.toFixed(3)}</td><td class="mono">${e.mean_score.toFixed(3)}</td>
                <td>${e.matches_truth === null ? `<span class="badge info">No reference</span>` : e.matches_truth ? `<span class="badge ok">✓ True positive</span>` : `<span class="badge bad">False alarm</span>`}</td></tr>`).join("")}</tbody></table></div>`
              : `<div class="banner green">${icon("check_circle")}<div>No seizure episodes above threshold ${r.threshold.toFixed(2)} in this recording${hasTruth ? " (the recording does contain an annotated seizure, so try a lower threshold)" : ""}.</div></div>`}`;
          if (r.timeline) plot($("#timeline"), r.timeline);
        } catch (e) {
          if (!isStale(t)) out.innerHTML = `<div class="banner rose">${icon("error")}<div>${esc(e.message)}</div></div>`;
        }
      }, delay);
    };
    thr.addEventListener("input", () => { setThr(+thr.value); run(200); });
    $("#thr-reset").addEventListener("click", () => { setThr(m.threshold); run(); });
    run();
  };

  pages.evaluation = async (t) => {
    const d = await api("/api/evaluation");
    if (isStale(t)) return;
    const real = d.real, active = d.active;
    app.innerHTML = `<div class="page">
      ${banner("Evaluation", "Model <em>evaluation</em> &amp; metrics", "Reproducible window-level metrics and event-level clinical statistics, with training, validation and test splits kept strictly apart.", "evaluation")}
      ${segmented("tab", [{ value: "real", label: "Real CHB-MIT results", icon: "local_hospital" }, { value: "active", label: "Active model bundle", icon: "psychology" }], "real")}
      <div class="mt" id="tab-real"></div><div class="mt hidden" id="tab-active"></div></div>`;
    bindSegmented(app, "tab", (v) => { ["real", "active"].forEach((t) => $(`#tab-${t}`).classList.toggle("hidden", t !== v)); window.dispatchEvent(new Event("resize")); });

    const rt = $("#tab-real");
    if (!real) rt.innerHTML = `<div class="empty">${icon("insights")}No experiment artifacts in results/. Run the experiment from the Dashboard.</div>`;
    else {
      const p = real.partitions, em = real.event_metrics || {};
      const splits = Object.keys(real.splits);
      rt.innerHTML = `
        <div class="banner rose">${icon("science")}<div><strong>Patient-level holdout.</strong> Train <code>${esc(p.train?.recording)}</code> (${esc(p.train?.patient)}) · Validation <code>${esc(p.validation?.recording)}</code> (${esc(p.validation?.patient)}) · Test <code>${esc(p.test?.recording)}</code> (unseen ${esc(p.test?.patient)}). Threshold ${num(real.threshold)}, calibrated on validation. Not clinically approved.</div></div>
        <div class="row spread mt">${segmented("split", splits.map((s) => ({ value: s, label: `${s}${s === "Test" ? ` · unseen ${p.test?.patient}` : ""}` })), splits[0])}</div>
        <div id="split-metrics" class="grid g4 mt"></div>
        <div class="grid g2 mt">
          <div class="chart"><div class="plot" id="real-cm"></div></div>
          <div class="card"><h4>${icon("event_available")}Event-level clinical metrics (test)</h4>
            <div class="row-kv"><span>True seizures</span><span>${em.total_true_events ?? "–"}</span></div>
            <div class="row-kv"><span>Detected seizures</span><span>${em.detected_true_events ?? "–"}</span></div>
            <div class="row-kv"><span>Event sensitivity</span><span>${pct(em.event_sensitivity)}</span></div>
            <div class="row-kv"><span>False alarm episodes</span><span>${em.false_alarms ?? "–"}</span></div>
            <div class="row-kv"><span>False alarms per hour</span><span>${num(em.false_alarms_per_hour)}</span></div>
            <p class="mt" style="font-size:.78rem">FAR/h counts false-positive clustered episodes per hour of non-seizure recording.</p></div>
        </div>
        ${real.audit ? `<details class="disclosure mt"><summary>${icon("fact_check")}Audit report: single-recording chronological split</summary><p style="margin-bottom:10px">${esc(real.audit.audit_finding)}</p><pre>${esc(JSON.stringify(real.audit.chronological_partitions, null, 2))}</pre></details>` : ""}
        ${sectionLabel("Saved artifacts · results/")}
        <div class="grid g2">
          ${real.report_text ? `<div class="card"><h4>${icon("description")}classification_report.txt</h4><pre>${esc(real.report_text)}</pre></div>` : ""}
          ${real.images.map((n) => `<div class="card"><h4>${icon("image")}${esc(n)}</h4><div class="chart" style="padding:0;background:#fff"><img src="/artifacts/${encodeURIComponent(n)}" alt="${esc(n)}" loading="lazy"></div></div>`).join("")}
        </div>`;
      const showSplit = (s) => {
        const m = real.splits[s];
        $("#split-metrics").innerHTML = `${metric(`${s} accuracy`, pct(m.accuracy))}${metric(`${s} sensitivity`, pct(m.sensitivity))}${metric(`${s} specificity`, pct(m.specificity))}${metric(`${s} F1 score`, pct(m.f1_score))}`;
        plot($("#real-cm"), m.confusion);
      };
      bindSegmented(rt, "split", showSplit);
      showSplit(splits[0]);
    }

    const at = $("#tab-active");
    if (!active) { at.innerHTML = `<div class="empty">${icon("psychology")}No active model loaded.</div>`; return; }
    const opts = ["Validation", "Test", "Dummy baseline"].filter((k) => active[k]);
    at.innerHTML = `<div class="banner sky">${icon("psychology")}<div>Active model <code>${esc(active.model.name)}</code> · loaded from <code>${esc(active.model.source)}</code>.</div></div>
      <div class="mt">${segmented("asplit", opts.map((o) => ({ value: o, label: o })), opts[0])}</div>
      <div id="a-metrics" class="grid g4 mt"></div><div id="a-warn"></div>
      <div class="grid g2 mt"><div class="chart"><div class="plot" id="a-cm"></div></div><div class="chart" id="a-roc-wrap"><div class="plot" id="a-roc"></div></div></div>`;
    const showActive = (k) => {
      const r = active[k];
      $("#a-metrics").innerHTML = `${metric("Accuracy", pct(r.accuracy))}${metric("Sensitivity", pct(r.sensitivity))}${metric("Specificity", pct(r.specificity))}${metric("F1 score", pct(r.f1_score))}`;
      $("#a-warn").innerHTML = r.warnings.map((w) => `<div class="banner amber mt">${icon("warning")}<div>${esc(w)}</div></div>`).join("");
      plot($("#a-cm"), r.confusion);
      if (r.roc) { $("#a-roc-wrap").innerHTML = `<div class="plot" id="a-roc"></div>`; plot($("#a-roc"), r.roc); }
      else $("#a-roc-wrap").innerHTML = `<div class="empty">${icon("show_chart")}ROC curve needs both classes in this split.</div>`;
    };
    bindSegmented(at, "asplit", showActive);
    showActive(opts[0]);
  };

  pages.docs = async () => {
    app.innerHTML = `<div class="page">
      ${banner("Documentation", "Methodology &amp; <em>attribution</em>", "Dataset source, preprocessing and modelling choices, evaluation protocol and credits.", "docs")}
      <div class="grid g2 prose">
        <div class="card"><h3>1. Dataset source</h3><ul>
          <li><strong>Database:</strong> <a href="https://physionet.org/content/chbmit/1.0.0/" target="_blank" rel="noopener">CHB-MIT Scalp EEG Database v1.0.0</a> on PhysioNet</li>
          <li><strong>Clinical source:</strong> Children's Hospital Boston &amp; MIT</li>
          <li><strong>Subjects:</strong> pediatric patients with intractable seizures, monitored after withdrawal of anti-epileptic medication</li>
          <li><strong>Sampling:</strong> 256 Hz, 16-bit</li>
          <li><strong>Files used:</strong> <code>chb01_03.edf</code>, <code>chb01_04.edf</code>, <code>chb02_16.edf</code> with WFDB <code>.seizures</code> and <code>*-summary.txt</code> annotations</li>
          <li><strong>License:</strong> Open Data Commons Public Domain Dedication and License v1.0 (ODC-PDDL)</li></ul>
          <blockquote>Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation, 101(23), e215–e220.</blockquote></div>
        <div class="card"><h3>2. Signal processing &amp; features</h3><ol>
          <li><strong>Montage:</strong> standard 18-channel bipolar chain (<code>FP1-F7</code>, <code>F7-T7</code>, <code>T7-P7</code>, <code>P7-O1</code>, …), names normalised to 10-20.</li>
          <li><strong>Filtering:</strong> zero-phase 4th-order Butterworth band-pass 0.5–40 Hz and IIR notch at 60 Hz (Q = 30).</li>
          <li><strong>Windowing:</strong> non-overlapping 5 s windows; label 1 when ≥ 50% overlaps a seizure, 0 when there is no overlap; 5 s transition buffers excluded.</li>
          <li><strong>Features:</strong> time domain (mean, SD, skewness, kurtosis, peak-to-peak, RMS, line length, zero crossings, Hjorth), Welch band powers (δ, θ, α, β, γ), relative powers, spectral entropy, SEF95, and cross-channel aggregates.</li></ol></div>
        <div class="card"><h3>3. Anti-leakage protocol</h3><ul>
          <li>Patient-level holdout: train and validation on <code>chb01</code>, test on unseen <code>chb02</code>.</li>
          <li>No recording or seizure event is split across partitions.</li>
          <li>Scalers are fit on training data only; the threshold is tuned on validation, never on test.</li>
          <li>Small-sample study: treat results as a proof of concept, not a generalisation claim.</li></ul></div>
        <div class="card"><h3>4. Reproduce locally</h3>
          <pre># Windows: double-click START_PROJECT.bat
# Or from a terminal:
python -m pip install -r requirements.txt
python scripts/run_real_experiment.py   # retrain + write results/
python -m pytest -v                      # run the test suite
python run_app.py                        # open http://localhost:8501</pre></div>
      </div>
      <div class="banner rose mt">${icon("health_and_safety")}<div><strong>Medical disclaimer.</strong> ${esc(state.config.disclaimer)}</div></div>
      <p class="muted mt" style="font-size:.78rem">Photography from Unsplash (free license). See assets/images/CREDITS.md.</p></div>`;
  };

  // ------------------------------------------------------------------ router
  let routeToken = 0;
  const isStale = (t) => t !== routeToken;

  async function route(force = false) {
    const legacy = new URLSearchParams(location.search).get("page");
    let id = (location.hash.match(/^#\/([\w-]+)/) || [])[1];
    if (!id && legacy) {
      const map = { dashboard: "dashboard", upload: "data", eeg: "data", signal: "signals", training: "training", seizure: "detection", detection: "detection", evaluation: "evaluation", documentation: "docs" };
      id = Object.entries(map).find(([k]) => legacy.toLowerCase().includes(k))?.[1];
    }
    if (!PAGES.some((p) => p.id === id)) id = "dashboard";
    if (id === state.page && !force) return;
    state.page = id;
    const token = ++routeToken;
    $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === id));
    document.body.classList.remove("nav-open");
    document.title = `${PAGES.find((p) => p.id === id).label} · Harmful Brain Activity Classification`;
    window.scrollTo({ top: 0, behavior: "instant" });
    try { await pages[id](token); state.retries = 0; }
    catch (e) {
      if (isStale(token)) return;
      if ([502, 503, 504].includes(e.status) && (state.retries = (state.retries || 0) + 1) <= 12) {
        app.innerHTML = `<div class="page"><div class="card row"><div class="spinner"></div><div><strong>The server is waking up or updating…</strong><br><span class="muted">The free hosting tier sleeps when idle. Retrying automatically (attempt ${state.retries} of 12).</span></div></div></div>`;
        setTimeout(() => { if (!isStale(token)) route(true); }, 5000);
        return;
      }
      app.innerHTML = `<div class="page"><div class="banner rose">${icon("error")}<div><strong>Could not load this page.</strong> ${esc(e.message)}</div></div><button class="btn mt" onclick="location.reload()">${icon("refresh")}Retry</button></div>`;
    }
  }

  window.addEventListener("hashchange", () => route());

  (async function boot() {
    try {
      state.config = await api("/api/config");
      renderStatus();
      api("/api/model").then((m) => { state.model = m; renderStatus(); }).catch(() => {});
      await route(true);
    } catch (e) {
      app.innerHTML = `<div class="banner rose">${icon("cloud_off")}<div><strong>The server is not responding.</strong> ${esc(e.message)}. If this is the hosted demo, the free instance may be waking up; wait 30–60 seconds and refresh.</div></div>`;
    }
  })();
})();
