const CACHE_NAME = 'imdb-sentiment-v1';
const ASSETS = [
  '/',
  '/static/style.css',
  '/static/app.js',
  '/static/manifest.json',
  '/static/favicon.ico',
  '/static/favicon-16x16.png',
  '/static/favicon-32x32.png',
  '/static/apple-touch-icon.png',
  '/static/android-chrome-192x192.png',
  '/static/android-chrome-512x512.png'
];

// Install Event - Pre-cache Static Assets
self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        return cache.addAll(ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate Event - Clean old caches
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch Event - Network First with Cache Fallback for GET requests
self.addEventListener('fetch', (e) => {
  // Only intercept GET requests (skip API POST predictions)
  if (e.request.method !== 'GET') return;

  e.respondWith(
    fetch(e.request)
      .then((res) => {
        // Clone response and cache it
        if (res.status === 200) {
          const resClone = res.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(e.request, resClone);
          });
        }
        return res;
      })
      .catch(() => {
        // Fallback to cache if network is unavailable
        return caches.match(e.request);
      })
  );
});
