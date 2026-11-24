/* DermaLens AI — analysis tool. */
(function () {
  'use strict';
  const { ICONS, FALLBACK_OPTIONS, FALLBACK_CONDITIONS, esc, getInfo, disclaimerHtml } = window.DL;
  const MAX_BYTES = 10 * 1024 * 1024;
  const TYPES = ['image/jpeg', 'image/png', 'image/webp'];
  const RECENT_KEY = 'dermalens.recent.v1';
  const $ = id => document.getElementById(id);

  const form = $('analyse-form'), dz = $('dropzone'), fileInput = $('file-input'), camInput = $('camera-input');
  const preview = $('preview'), previewImg = $('preview-img'), previewName = $('preview-name');
  const fileErr = $('file-error'), submitErr = $('submit-error'), btn = $('analyse-btn'), hint = $('analyse-hint');
  const resultEl = $('result');

  let file = null, objectUrl = null, info = null, options = FALLBACK_OPTIONS;
  const selectedRegions = new Set();

  /* ---------- metadata form ---------- */
  function renderOptions(opts) {
    const o = {
      age_groups: (opts && Array.isArray(opts.age_groups) && opts.age_groups.length) ? opts.age_groups : FALLBACK_OPTIONS.age_groups,
      sexes: (opts && Array.isArray(opts.sexes) && opts.sexes.length) ? opts.sexes : FALLBACK_OPTIONS.sexes,
      body_regions: (opts && Array.isArray(opts.body_regions) && opts.body_regions.length) ? opts.body_regions : FALLBACK_OPTIONS.body_regions
    };
    options = o;
    const age = $('age-group'), cur = age.value;
    age.innerHTML = '<option value="">Prefer not to say</option>' + o.age_groups.map(a => '<option value="' + esc(a.key) + '">' + esc(a.label) + '</option>').join('');
    age.value = cur;
    const curSex = (form.querySelector('input[name="sex"]:checked') || {}).value || '';
    $('sex-group').innerHTML = [{ key: '', label: 'Prefer not to say' }].concat(o.sexes).map(s =>
      '<label><input type="radio" name="sex" value="' + esc(s.key) + '"' + (s.key === curSex ? ' checked' : '') + '><span>' + esc(s.label) + '</span></label>').join('');
    $('region-chips').innerHTML = o.body_regions.map(r =>
      '<button type="button" class="chip" data-key="' + esc(r.key) + '" aria-pressed="' + (selectedRegions.has(r.key) ? 'true' : 'false') + '">' + esc(r.label) + '</button>').join('');
  }
  $('region-chips').addEventListener('click', e => {
    const c = e.target.closest('.chip'); if (!c) return;
    const k = c.dataset.key;
    if (selectedRegions.has(k)) selectedRegions.delete(k); else selectedRegions.add(k);
    c.setAttribute('aria-pressed', selectedRegions.has(k) ? 'true' : 'false');
  });

  /* ---------- file handling ---------- */
  function showFileError(msg) { fileErr.textContent = msg; fileErr.classList.toggle('show', !!msg); }
  function setFile(f) {
    showFileError('');
    if (!f) return;
    if (!TYPES.includes(f.type)) { showFileError('Please choose a JPEG, PNG or WebP image (this file is ' + (f.type || 'an unknown type') + ').'); return; }
    if (f.size > MAX_BYTES) { showFileError('This photo is ' + (f.size / 1048576).toFixed(1) + ' MB. The limit is 10 MB — try a smaller or compressed version.'); return; }
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    file = f; objectUrl = URL.createObjectURL(f);
    previewImg.src = objectUrl;
    previewName.textContent = f.name + ' · ' + (f.size / 1024 < 1024 ? Math.round(f.size / 1024) + ' KB' : (f.size / 1048576).toFixed(1) + ' MB');
    preview.classList.add('show'); dz.hidden = true;
    btn.disabled = false; hint.textContent = 'Ready when you are.';
  }
  function clearFile() {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    file = null; objectUrl = null; previewImg.removeAttribute('src');
    preview.classList.remove('show'); dz.hidden = false;
    fileInput.value = ''; camInput.value = '';
    btn.disabled = true; hint.textContent = 'Add a photo to continue.';
  }
  $('pick-btn').addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
  $('camera-btn').addEventListener('click', e => { e.stopPropagation(); camInput.click(); });
  $('replace-btn').addEventListener('click', () => fileInput.click());
  $('remove-btn').addEventListener('click', () => { clearFile(); dz.focus(); });
  dz.addEventListener('click', () => fileInput.click());
  dz.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
  fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
  camInput.addEventListener('change', () => setFile(camInput.files[0]));
  ['dragenter', 'dragover'].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', e => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) setFile(f); });

  /* ---------- submit ---------- */
  function setLoading(on) {
    btn.disabled = on || !file;
    btn.setAttribute('aria-busy', on ? 'true' : 'false');
    btn.querySelector('.lbl').textContent = on ? 'Analysing…' : 'Analyse photo';
    const ic = btn.querySelector('span:first-child');
    ic.innerHTML = on ? '<span class="spinner" aria-hidden="true"></span>' : ICONS.cpu;
  }
  function labelFor(list, key) { const x = list.find(i => i.key === key); return x ? x.label : key; }

  form.addEventListener('submit', async e => {
    e.preventDefault();
    submitErr.classList.remove('show');
    if (!file) { showFileError('Please add a photo first.'); return; }
    const fd = new FormData();
    fd.append('file', file);
    const age = $('age-group').value;
    const sex = (form.querySelector('input[name="sex"]:checked') || {}).value || '';
    const regions = Array.from(selectedRegions);
    if (age) fd.append('age_group', age);
    if (sex) fd.append('sex', sex);
    if (regions.length) fd.append('body_regions', regions.join(','));
    const meta = {
      age: age ? labelFor(options.age_groups, age) : null,
      sex: sex ? labelFor(options.sexes, sex) : null,
      regions: regions.map(r => labelFor(options.body_regions, r))
    };
    setLoading(true);
    try {
      const res = await fetch('/api/predict', { method: 'POST', body: fd });
      let data = null;
      try { data = await res.json(); } catch (_) { data = null; }
      if (!res.ok) {
        const msg = (data && typeof data.detail === 'string') ? data.detail : 'The server could not analyse this photo (error ' + res.status + ').';
        throw new Error(msg);
      }
      if (!data || !data.status) throw new Error('Unexpected response from the server.');
      renderResult(data, meta);
      saveRecent(data);
    } catch (err) {
      submitErr.textContent = (err && err.message && err.message !== 'Failed to fetch') ? err.message : 'Could not reach the analysis server. Check your connection and try again.';
      submitErr.classList.add('show');
    } finally { setLoading(false); }
  });

  /* ---------- result rendering ---------- */
  const pct = v => (typeof v === 'number' && isFinite(v)) ? Math.round(v * 100) : null;

  function bars(top3, greyed) {
    if (!Array.isArray(top3) || !top3.length) return '';
    return '<div class="bars' + (greyed ? ' greyed' : '') + '">' + top3.slice(0, 3).map(t => {
      const p = pct(t.probability); const w = p == null ? 0 : Math.max(1, Math.min(100, p));
      return '<div class="bar-row"><span class="bl">' + esc(t.label) + '</span><span class="bv">' + (p == null ? '—' : p + '%') + '</span>' +
        '<div class="bar-track" role="progressbar" aria-label="' + esc(t.label) + '" aria-valuemin="0" aria-valuemax="100" aria-valuenow="' + (p || 0) + '"><div class="bar-fill" style="width:' + w + '%"></div></div></div>';
    }).join('') + '</div>';
  }
  function ring(conf, color) {
    const p = pct(conf); if (p == null) return '';
    return '<div class="ring" style="--p:' + p + ';--c:' + color + '" role="img" aria-label="Confidence ' + p + ' percent"><span>' + p + '%<small>confidence</small></span></div>';
  }
  function images(data, showHeat) {
    const orig = '<figure><img src="' + esc(objectUrl || '') + '" alt="Your uploaded photo"><figcaption>Your photo</figcaption></figure>';
    const hm = (showHeat && typeof data.heatmap_png_base64 === 'string' && /^[A-Za-z0-9+/=\s]+$/.test(data.heatmap_png_base64))
      ? '<figure><img src="data:image/png;base64,' + data.heatmap_png_base64.replace(/\s/g, '') + '" alt="Grad-CAM heatmap showing regions that influenced the result"><figcaption>Model attention (Grad-CAM)</figcaption><div class="heat-legend" aria-hidden="true">low <i></i> high</div></figure>' : '';
    return '<div class="images' + (hm ? '' : ' single') + '">' + orig + hm + '</div>';
  }
  function metaBadges(data, meta) {
    const b = [];
    if (data.used_metadata) b.push('<span class="badge badge-meta">' + ICONS.layers + 'Metadata used</span>');
    if (meta.age) b.push('<span class="badge">' + ICONS.user + esc(meta.age) + '</span>');
    if (meta.sex) b.push('<span class="badge">' + esc(meta.sex) + '</span>');
    meta.regions.forEach(r => b.push('<span class="badge">' + ICONS.tag + esc(r) + '</span>'));
    if (typeof data.processing_ms === 'number') b.push('<span class="badge">' + (data.processing_ms / 1000).toFixed(2) + ' s</span>');
    return b.length ? '<div class="badges" aria-label="Analysis details">' + b.join('') + '</div>' : '';
  }
  function conditionInfo(data) {
    const c = data.condition || (info && info.conditions && info.conditions[data.label]) || FALLBACK_CONDITIONS[data.label] || {};
    const parts = [];
    if (c.summary) parts.push('<div class="info-block"><h3>About this condition</h3><p>' + esc(c.summary) + '</p></div>');
    if (c.typical_sites) parts.push('<div class="info-block"><h3>Typical sites</h3><p>' + esc(c.typical_sites) + '</p></div>');
    const next = c.next_steps || 'Share this result with a qualified dermatologist, who can confirm the condition and recommend treatment.';
    parts.push('<div class="next-steps"><h3>Suggested next steps</h3><p>' + esc(next) + '</p></div>');
    return '<div class="info-list">' + parts.join('') + '</div>';
  }

  function renderResult(data, meta) {
    const s = data.status;
    const when = new Date();
    const stamp = when.toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' });
    let cls, icon, label, title, sub, body;
    const msg = data.message ? esc(data.message) : '';

    if (s === 'identified') {
      cls = 'r-identified'; icon = ICONS.doc; label = 'Closest match';
      title = esc(data.display_label || data.label || 'Condition identified');
      sub = msg || 'The photo most closely matches this condition.';
      body = '<div class="split"><div>' + images(data, true) + '</div><div><h3 style="font-size:1rem;margin-bottom:12px">Top matches</h3>' + bars(data.top3, false) +
        '<div style="margin-top:24px">' + conditionInfo(data) + '</div></div></div>';
    } else if (s === 'healthy') {
      cls = 'r-healthy'; icon = ICONS.check; label = 'Healthy skin';
      title = 'No skin condition detected';
      sub = msg || 'This photo does not show signs of the conditions DermaLens covers.';
      body = '<div class="split"><div>' + images(data, true) + '</div><div class="info-list">' +
        '<div class="info-block"><h3>What this means</h3><p>The model did not find patterns of contact dermatitis, eczema, psoriasis, scabies or tinea in this photo.</p></div>' +
        '<div class="next-steps"><h3>Suggested next steps</h3><p>If you still have symptoms such as itching, pain, spreading or bleeding, please see a dermatologist — a photo cannot capture everything.</p></div></div></div>';
    } else if (s === 'not_identified') {
      cls = 'r-not_identified'; icon = ICONS.question; label = 'Not identified';
      title = 'Not identified';
      sub = 'This may be a condition DermaLens does not cover, or the photo is too uncertain to call. Please consult a dermatologist.';
      body = '<div class="split"><div>' + images(data, false) + '</div><div><h3 style="font-size:1rem;margin-bottom:4px">Closest matches</h3><p class="muted small">Shown for reference only — none was confident enough to report.</p>' + bars(data.top3, true) +
        '<div class="next-steps" style="margin-top:24px"><h3>Suggested next steps</h3><p>Book a visit with a dermatologist. If you retake the photo, use daylight, get closer and keep it in focus.</p></div></div></div>';
    } else if (s === 'not_skin') {
      cls = 'r-not_skin'; icon = ICONS.image; label = 'Photo not usable';
      title = 'This doesn’t look like a photo of skin';
      sub = 'Please upload a close, well-lit photo of the affected area.';
      body = '<div class="split"><div>' + images(data, false) + '</div><div class="info-list"><div class="info-block"><h3>Tips for a better photo</h3>' +
        '<ul class="hint-list" style="margin-top:8px"><li>' + ICONS.check + 'Hold the camera 10–20 cm from the skin</li><li>' + ICONS.check + 'Use natural daylight; avoid flash and filters</li><li>' + ICONS.check + 'Make sure the affected area fills most of the frame</li><li>' + ICONS.check + 'Tap to focus and keep your hand steady</li></ul></div></div></div>';
    } else {
      cls = 'r-not_skin'; icon = ICONS.question; label = 'Unknown result'; title = 'Unexpected result'; sub = msg || 'The server returned a status this page does not recognise.'; body = '';
    }

    const showRing = (s === 'identified' || s === 'healthy');
    const ringColor = s === 'healthy' ? 'var(--ok)' : 'var(--primary)';
    resultEl.innerHTML =
      '<div class="print-only print-head" style="display:none"><h1>DermaLens AI — analysis report</h1><span>' + esc(stamp) + '</span></div>' +
      '<article class="result-card ' + cls + '" aria-labelledby="result-title">' +
        '<header class="result-head"><div class="status-icon" aria-hidden="true">' + icon + '</div>' +
          '<div class="rh-text"><div class="status-label">' + label + '</div><h2 id="result-title">' + title + '</h2><p>' + sub + '</p></div>' +
          (showRing ? ring(data.confidence, ringColor) : '') + '</header>' +
        '<div class="result-body">' + metaBadges(data, meta) + body +
          '<p class="muted small" style="margin:0">Analysed ' + esc(stamp) + '</p></div>' +
        '<div class="result-foot">' + disclaimerHtml(data.disclaimer || (info && info.disclaimer)) + '</div>' +
        '<div class="result-actions"><button type="button" class="btn btn-primary" id="again-btn">' + ICONS.refresh + 'Analyse another photo</button>' +
          '<button type="button" class="btn btn-ghost" id="print-btn">' + ICONS.printer + 'Print / save report</button></div>' +
      '</article>';
    $('again-btn').addEventListener('click', () => {
      resultEl.innerHTML = ''; clearFile(); window.scrollTo({ top: 0 }); dz.focus();
    });
    $('print-btn').addEventListener('click', () => window.print());
    resultEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    resultEl.focus({ preventScroll: true });
  }

  /* ---------- recent analyses (localStorage, no images) ---------- */
  function loadRecent() {
    try { const v = JSON.parse(localStorage.getItem(RECENT_KEY) || '[]'); return Array.isArray(v) ? v : []; } catch (_) { return []; }
  }
  function saveRecent(data) {
    const names = { identified: data.display_label || data.label, healthy: 'Healthy skin', not_identified: 'Not identified', not_skin: 'Not a skin photo' };
    const entry = { label: String(names[data.status] || data.status).slice(0, 80), confidence: (data.status === 'identified' || data.status === 'healthy') ? data.confidence : null, status: data.status, date: new Date().toISOString() };
    const list = [entry].concat(loadRecent()).slice(0, 8);
    try { localStorage.setItem(RECENT_KEY, JSON.stringify(list)); } catch (_) { /* storage unavailable */ }
    renderRecent(list);
  }
  function renderRecent(list) {
    list = list || loadRecent();
    const panel = $('recent-panel');
    if (!list.length) { panel.hidden = true; return; }
    panel.hidden = false;
    $('recent-list').innerHTML = list.map(r => {
      const p = pct(r.confidence);
      let d = ''; try { d = new Date(r.date).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' }); } catch (_) { d = ''; }
      return '<li><span class="rl">' + esc(r.label) + (p != null ? ' <span class="muted" style="font-weight:400">· ' + p + '%</span>' : '') + '</span><span class="rd">' + esc(d) + '</span></li>';
    }).join('');
  }
  $('clear-recent').addEventListener('click', () => { try { localStorage.removeItem(RECENT_KEY); } catch (_) { /* ignore */ } renderRecent([]); });

  /* ---------- init ---------- */
  renderOptions(null);
  renderRecent();
  getInfo().then(i => { info = i; if (i && i.metadata_options) renderOptions(i.metadata_options); });
})();
