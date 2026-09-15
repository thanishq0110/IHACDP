/* Shared navigation: collapses to a hamburger on small screens.
   Hand-built rather than pulled from a CDN, because the application must run
   with no network access at all. */
function syncNavHeight() {
  const bar = document.querySelector('.nav');
  if (bar) document.documentElement.style.setProperty('--nav-h', bar.offsetHeight + 'px');
}
syncNavHeight();
window.addEventListener('resize', syncNavHeight, { passive: true });

(function () {
  const nav = document.querySelector('.nav-in');
  if (!nav) return;
  const links = nav.querySelector('.nav-links');
  if (!links) return;

  const btn = document.createElement('button');
  btn.className = 'nav-toggle';
  btn.setAttribute('aria-label', 'Menu');
  btn.setAttribute('aria-expanded', 'false');
  btn.setAttribute('aria-controls', 'nav-links');
  btn.innerHTML = '<span class="bars" aria-hidden="true"><i></i><i></i><i></i></span>';
  links.id = 'nav-links';
  nav.appendChild(btn);

  const scrim = document.createElement('div');
  scrim.className = 'nav-scrim';
  document.body.appendChild(scrim);

  function setOpen(open) {
    links.classList.toggle('open', open);
    scrim.classList.toggle('show', open);
    btn.classList.toggle('open', open);
    btn.setAttribute('aria-expanded', String(open));
    document.body.classList.toggle('nav-locked', open);
  }

  btn.addEventListener('click', () => setOpen(!links.classList.contains('open')));
  scrim.addEventListener('click', () => setOpen(false));
  links.addEventListener('click', e => { if (e.target.tagName === 'A') setOpen(false); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') setOpen(false); });
  window.addEventListener('resize', () => { if (window.innerWidth > 640) setOpen(false); }, { passive: true });
})();
