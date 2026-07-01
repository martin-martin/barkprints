// Shared Leaflet setup for Barkprints. Load the vendored leaflet.js BEFORE this.
//
// Two jobs:
//   1. Point Leaflet's default marker at our vendored images. Leaflet's
//      auto-detection of the image path is fragile, so we set it explicitly.
//   2. Expose bpTileLayer() so every map uses the same OSM raster tiles and
//      attribution. Tiles are fetched from the OpenStreetMap tile servers at
//      runtime (not vendored) and are subject to the OSM tile usage policy —
//      fine for personal, low-volume use.
(function () {
  if (!window.L) return;
  const base = '/static/vendor/leaflet/images/';
  // Drop Leaflet's default _getIconUrl, which prepends an auto-detected
  // imagePath and would double our absolute URLs. With it gone, the explicit
  // URLs below are used verbatim. (This is the standard bundler-era fix.)
  delete L.Icon.Default.prototype._getIconUrl;
  L.Icon.Default.mergeOptions({
    iconRetinaUrl: base + 'marker-icon-2x.png',
    iconUrl: base + 'marker-icon.png',
    shadowUrl: base + 'marker-shadow.png',
  });

  window.bpTileLayer = function () {
    return L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    });
  };
})();
