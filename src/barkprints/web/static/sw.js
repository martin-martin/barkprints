// Minimal service worker: cache the app shell so the page opens offline.
// Inference still requires the server, but the UI loads instantly.

const CACHE = 'barkprints-shell-v3';
const SHELL = [
  '/',
  '/static/icon.svg',
  '/manifest.webmanifest',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  // Never cache API calls — results are computed per upload.
  if (url.pathname.startsWith('/api/')) return;

  // Network-first for the HTML shell, cache fallback when offline. Only the
  // start page refreshes the '/' cache entry — caching whatever page was
  // navigated last (e.g. /gallery or /login) under '/' would make an offline
  // open of the app show the wrong page.
  if (req.mode === 'navigate' || req.destination === 'document') {
    event.respondWith(
      fetch(req)
        .then((res) => {
          if (url.pathname === '/' && res.ok) {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put('/', copy));
          }
          return res;
        })
        .catch(() => caches.match('/'))
    );
    return;
  }

  // Stale-while-revalidate for static assets: answer from cache immediately
  // for speed, but always refresh the cached copy in the background so a
  // deploy shows up on the next load (plain cache-first served stale JS/CSS
  // until the cache version was bumped by hand).
  if (url.pathname.startsWith('/static/') || url.pathname === '/manifest.webmanifest') {
    event.respondWith(
      caches.match(req).then((hit) => {
        const refresh = fetch(req)
          .then((res) => {
            if (res.ok) {
              const copy = res.clone();
              caches.open(CACHE).then((c) => c.put(req, copy));
            }
            return res;
          })
          .catch(() => hit);
        return hit || refresh;
      })
    );
  }
});
