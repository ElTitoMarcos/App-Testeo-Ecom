const MARGIN = 12;
const GAP = 10;
let tipEl = null;
let activeBtn = null;

function ensureTipEl() {
  if (tipEl) return tipEl;
  tipEl = document.createElement('div');
  tipEl.id = 'floating-help-tooltip';
  tipEl.innerHTML = '<div class="arrow"></div><div class="content"></div>';
  document.body.appendChild(tipEl);
  return tipEl;
}

function getHoveredButton() {
  return document.querySelector('.chart-help:hover');
}

function showTip(btn) {
  if (!btn) return;
  const tip = ensureTipEl();
  const content = tip.querySelector('.content');
  const arrow = tip.querySelector('.arrow');

  activeBtn = btn;
  content.textContent = btn.getAttribute('data-tip') || '';

  tip.classList.add('show');
  tip.classList.remove('top', 'bottom');
  tip.style.transform = 'translate(-9999px, -9999px)';
  tip.style.left = '0px';
  tip.style.top = '0px';
  arrow.style.left = '50%';
  arrow.style.right = 'auto';
  arrow.style.top = '';
  arrow.style.bottom = '';

  requestAnimationFrame(() => {
    if (!activeBtn || activeBtn !== btn) {
      return;
    }

    const rect = btn.getBoundingClientRect();
    const vw = window.innerWidth || document.documentElement.clientWidth || 0;
    const vh = window.innerHeight || document.documentElement.clientHeight || 0;
    const ttW = tip.offsetWidth;
    const ttH = tip.offsetHeight;

    let left = rect.left + rect.width / 2 - ttW / 2;
    left = Math.max(MARGIN, Math.min(left, vw - ttW - MARGIN));

    let top = rect.bottom + GAP;
    let placement = 'bottom';

    if (top + ttH + MARGIN > vh && rect.top - GAP - ttH >= MARGIN) {
      top = rect.top - GAP - ttH;
      placement = 'top';
    } else if (top + ttH + MARGIN > vh) {
      top = Math.max(MARGIN, vh - ttH - MARGIN);
    }

    top = Math.max(MARGIN, Math.min(top, vh - ttH - MARGIN));

    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
    tip.classList.add(placement);

    const arrowX = rect.left + rect.width / 2 - left - 5; // 5 = half arrow width
    const clampedArrowX = Math.max(8, Math.min(arrowX, ttW - 18));
    arrow.style.left = `${clampedArrowX}px`;
    arrow.style.right = 'auto';

    tip.style.transform = '';
  });
}

function hideTip(btn) {
  if (!tipEl) return;
  if (btn && btn !== activeBtn) return;

  activeBtn = null;
  tipEl.classList.remove('show', 'top', 'bottom');
  tipEl.style.transform = 'translate(-9999px, -9999px)';
}

function handleEnter(event) {
  const btn = event.target.closest('.chart-help');
  if (!btn) return;
  showTip(btn);
}

function handleLeave(event) {
  const btn = event.target.closest('.chart-help');
  if (!btn) return;
  hideTip(btn);
}

document.addEventListener('mouseenter', handleEnter, true);
document.addEventListener('mouseleave', handleLeave, true);

document.addEventListener('focusin', (event) => {
  const btn = event.target.closest('.chart-help');
  if (!btn) return;
  showTip(btn);
});

document.addEventListener('focusout', (event) => {
  const btn = event.target.closest('.chart-help');
  if (!btn) return;
  hideTip(btn);
});

function handleViewportChange() {
  if (!tipEl || !tipEl.classList.contains('show')) return;
  if (activeBtn && document.body.contains(activeBtn)) {
    showTip(activeBtn);
    return;
  }
  const hovered = getHoveredButton();
  if (hovered) {
    showTip(hovered);
  } else {
    hideTip();
  }
}

window.addEventListener('scroll', handleViewportChange, { passive: true });
document.addEventListener('scroll', handleViewportChange, true);
window.addEventListener('resize', handleViewportChange, { passive: true });

export {};
