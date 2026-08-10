// ─── Offline Status Utility ──────────────────────────────────────────────────

export const OfflineManager = (() => {
  const indicator = document.getElementById('offline-banner');
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');

  function updateUI(isOnline) {
    if (indicator) {
      indicator.classList.toggle('hidden', isOnline);
    }
    if (statusDot && statusText) {
      if (isOnline) {
        statusDot.className = 'status-dot online';
        statusText.textContent = 'Online';
      } else {
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'Offline mode';
      }
    }
  }

  function init() {
    updateUI(navigator.onLine);
    window.addEventListener('online', () => updateUI(true));
    window.addEventListener('offline', () => updateUI(false));
  }

  function isOnline() {
    return navigator.onLine;
  }

  return { init, isOnline };
})();

// ─── Service Worker Registration ─────────────────────────────────────────────

export async function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) return;
  try {
    const reg = await navigator.serviceWorker.register('./service-worker.js');
    console.log('[SW] Registered:', reg.scope);

    reg.addEventListener('updatefound', () => {
      const newWorker = reg.installing;
      newWorker.addEventListener('statechange', () => {
        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
          console.log('[SW] New version available');
        }
      });
    });
  } catch (err) {
    console.error('[SW] Registration failed:', err);
  }
}
