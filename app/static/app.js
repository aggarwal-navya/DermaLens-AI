/* DermaLens AI — shared helpers: icons, nav, footer, /api/info. */
(function () {
  'use strict';

  const ICONS = {
    logo: '<svg viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="9" fill="var(--primary)"/><circle cx="16" cy="16" r="8.5" fill="none" stroke="var(--primary-ink)" stroke-width="2.2"/><circle cx="16" cy="16" r="3.2" fill="var(--accent)"/><path d="M22 22l4.5 4.5" stroke="var(--primary-ink)" stroke-width="2.6" stroke-linecap="round"/></svg>',
    menu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>',
    close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
    alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l9.5 17h-19z"/><path d="M12 10v4M12 17.5v.01"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 15V4M7.5 8.5L12 4l4.5 4.5"/><path d="M4 15v3a2 2 0 002 2h12a2 2 0 002-2v-3"/></svg>',
    camera: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/></svg>',
    image: '<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="6" y="9" width="36" height="30" rx="5"/><circle cx="17" cy="20" r="4"/><path d="M6 33l10-9 8 7 6-5 12 10"/></svg>',
    layers: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></svg>',
    user: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/></svg>',
    heat: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5.5"/><circle cx="12" cy="12" r="2" fill="currentColor"/></svg>',
    question: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 114 2c-1 .7-1.5 1.2-1.5 2.5M12 17v.01"/></svg>',
    gauge: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 17a8 8 0 1116 0"/><path d="M12 17l4-6"/><circle cx="12" cy="17" r="1.2" fill="currentColor"/></svg>',
    shield: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l8 3v6c0 4.5-3.4 8-8 9-4.6-1-8-4.5-8-9V6z"/><path d="M9 12l2 2 4-4"/></svg>',
    skin: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 9c3-2 6 2 9 0s6-2 9 0"/><path d="M3 15c3-2 6 2 9 0s6-2 9 0"/><circle cx="8" cy="12" r=".8" fill="currentColor"/><circle cx="16" cy="12" r=".8" fill="currentColor"/></svg>',
    cpu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/><path d="M10 10h4v4h-4z"/></svg>',
    doc: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5M10 13h6M10 17h6"/></svg>',
    printer: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 9V3h10v6"/><rect x="3" y="9" width="18" height="8" rx="2"/><path d="M7 14h10v7H7z"/></svg>',
    refresh: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 11a8 8 0 10-2.3 5.7"/><path d="M20 4v7h-7"/></svg>',
    arrow: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
    trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/></svg>',
    tag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12V4h8l10 10-8 8z"/><circle cx="7.5" cy="8.5" r="1.3"/></svg>'
  };

  const FALLBACK_OPTIONS = {
    age_groups: [{ key: '0-10', label: '0–10 years' }, { key: '10-20', label: '10–20 years' },
      { key: '20-40', label: '20–40 years' }, { key: '40-60', label: '40–60 years' },
      { key: '60+', label: '60 years and above' }],
    sexes: [{ key: 'female', label: 'Female' }, { key: 'male', label: 'Male' }],
    body_regions: [{ key: 'face', label: 'Face' }, { key: 'scalp', label: 'Scalp' },
      { key: 'neck', label: 'Neck' }, { key: 'chest_abdomen', label: 'Chest / abdomen' },
      { key: 'back', label: 'Back' }, { key: 'buttocks', label: 'Buttocks' },
      { key: 'groin_genital', label: 'Groin / genital area' }, { key: 'arm', label: 'Arm / elbow' },
      { key: 'hand', label: 'Hand / fingers / palm' }, { key: 'leg', label: 'Thigh / knee / leg' },
      { key: 'foot', label: 'Foot / toes / sole' }]
  };

  const FALLBACK_CONDITIONS = {
    'Contact Dermatitis': { summary: 'An itchy, red rash caused by direct contact with an irritant or allergen such as detergents, metals, cosmetics or plants.', typical_sites: 'Hands, wrists, face, neck — wherever the trigger touched the skin.' },
    'Eczema': { summary: 'A chronic, relapsing inflammatory condition causing dry, itchy, scaly patches; it can thicken with repeated scratching.', typical_sites: 'Skin folds (elbows, behind knees), hands, face and neck.' },
    'Healthy': { summary: 'Skin without visible signs of the conditions DermaLens is trained on.', typical_sites: 'Any body site.' },
    'Psoriasis': { summary: 'An immune-mediated condition producing well-defined, thickened plaques with silvery-white scale; on darker skin it may look violet or grey.', typical_sites: 'Elbows, knees, scalp, lower back.' },
    'Scabies': { summary: 'A contagious mite infestation causing intense itching, worse at night, with small bumps and burrow lines.', typical_sites: 'Finger webs, wrists, waist, groin, buttocks.' },
    'Tinea Corporis': { summary: 'A superficial fungal infection (ringworm) forming ring-shaped, scaly patches with a raised, active border.', typical_sites: 'Trunk, groin, thighs, arms and neck.' }
  };
  const FALLBACK_DISCLAIMER = 'DermaLens AI is a research prototype for decision support only. It is not a medical device and does not provide a diagnosis. Always consult a qualified dermatologist.';

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  let infoPromise = null;
  function getInfo() {
    if (!infoPromise) {
      infoPromise = fetch('/api/info', { headers: { Accept: 'application/json' } })
        .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .catch(() => null);
    }
    return infoPromise;
  }

  function fmtPct(v, digits) {
    if (typeof v !== 'number' || !isFinite(v)) return '—';
    return (v * 100).toFixed(digits == null ? 0 : digits) + '%';
  }
  function fmtNum(v, digits) {
    if (typeof v !== 'number' || !isFinite(v)) return '—';
    return v.toFixed(digits == null ? 2 : digits);
  }

  function disclaimerHtml(text) {
    return '<div class="disclaimer" role="note">' + ICONS.alert +
      '<div><strong>Research prototype — not a diagnosis.</strong> <span data-disclaimer-text>' + esc(text || FALLBACK_DISCLAIMER) + '</span></div></div>';
  }

  function renderChrome() {
    const page = document.body.dataset.page;
    const header = document.getElementById('site-header');
    if (header) {
      const link = (href, key, label) => '<li><a href="' + href + '"' + (page === key ? ' aria-current="page"' : '') + '>' + label + '</a></li>';
      header.className = 'site-header';
      header.innerHTML = '<nav class="container nav" aria-label="Main">' +
        '<a class="brand" href="index.html" aria-label="DermaLens AI home">' + ICONS.logo + '<span>DermaLens <small>AI</small></span></a>' +
        '<button class="nav-toggle" type="button" aria-expanded="false" aria-controls="nav-links" aria-label="Open menu">' + ICONS.menu + '</button>' +
        '<ul class="nav-links" id="nav-links">' + link('index.html', 'home', 'Home') + link('analyze.html', 'analyze', 'Analyse') + link('about.html', 'about', 'Model &amp; Research') + '</ul></nav>';
      const btn = header.querySelector('.nav-toggle');
      const list = header.querySelector('.nav-links');
      const setOpen = open => {
        list.classList.toggle('open', open);
        btn.setAttribute('aria-expanded', String(open));
        btn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
        btn.innerHTML = open ? ICONS.close : ICONS.menu;
      };
      btn.addEventListener('click', () => setOpen(!list.classList.contains('open')));
      document.addEventListener('keydown', e => { if (e.key === 'Escape' && list.classList.contains('open')) { setOpen(false); btn.focus(); } });
    }
    const footer = document.getElementById('site-footer');
    if (footer) {
      footer.className = 'site-footer';
      footer.innerHTML = '<div class="container"><div class="footer-grid"><div>' +
        '<a class="brand" href="index.html">' + ICONS.logo + '<span>DermaLens <small>AI</small></span></a>' +
        '<p class="muted small" style="margin:12px 0 16px">Skin-condition decision support tuned for Indian and South Asian skin. B.Tech CSE major project, Meerut Institute of Technology.</p>' +
        '<div id="footer-disclaimer">' + disclaimerHtml() + '</div></div>' +
        '<ul class="footer-links"><li><a href="index.html">Home</a></li><li><a href="analyze.html">Analyse a photo</a></li><li><a href="about.html">Model &amp; Research</a></li><li><a href="about.html#privacy">Privacy</a></li></ul></div>' +
        '<div class="footer-bottom"><span>© ' + new Date().getFullYear() + ' DermaLens AI project team</span><span id="footer-version">Photos are processed in memory and never stored.</span></div></div>';
    }
    document.querySelectorAll('[data-icon]').forEach(el => { const i = ICONS[el.dataset.icon]; if (i) el.innerHTML = i; });
    getInfo().then(info => {
      if (!info) return;
      if (info.disclaimer) document.querySelectorAll('[data-disclaimer-text]').forEach(el => { el.textContent = info.disclaimer; });
      const v = document.getElementById('footer-version');
      if (v && info.app_version) v.textContent = 'Version ' + info.app_version + ' · Photos are processed in memory and never stored.';
    });
  }

  function fillMetrics(info) {
    const m = (info && info.metrics) || {};
    document.querySelectorAll('[data-metric]').forEach(el => {
      const k = el.dataset.metric;
      const v = m[k];
      if (k === 'test_images') el.textContent = (typeof v === 'number') ? v.toLocaleString('en-IN') : '—';
      else if (k === 'open_set_auroc') el.textContent = fmtNum(v, 2);
      else el.textContent = fmtPct(v, 0);
    });
  }

  window.DL = { ICONS, FALLBACK_OPTIONS, FALLBACK_CONDITIONS, FALLBACK_DISCLAIMER, esc, getInfo, fmtPct, fmtNum, disclaimerHtml, fillMetrics };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', renderChrome); else renderChrome();
})();
