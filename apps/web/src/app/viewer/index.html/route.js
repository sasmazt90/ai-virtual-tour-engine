const HTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
    <title>Photo Tour Viewer</title>

    <style>
      :root {
        --ui-bg: rgba(7, 8, 10, 0.82);
        --ui-border: rgba(255, 255, 255, 0.14);
        --ui-text: rgba(255, 255, 255, 0.92);
        --ui-subtext: rgba(255, 255, 255, 0.66);
        --ui-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);

        --hotspot-bg: rgba(255, 255, 255, 0.95);
        --hotspot-border: rgba(0, 0, 0, 0.18);
        --hotspot-shadow: 0 12px 24px rgba(0, 0, 0, 0.35);
      }

      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
        background: #000;
        overflow: hidden;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      }

      /* Fullscreen stage */
      #stage {
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        background: #000;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      /* Keeps a stable coordinate system regardless of image aspect ratio */
      #mediaWrap {
        position: relative;
        width: 100vw;
        height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      #mainImage {
        max-width: 100%;
        max-height: 100%;
        width: auto;
        height: auto;
        object-fit: contain;
        display: block;
        user-select: none;
        -webkit-user-drag: none;
      }

      /* Overlay is dynamically sized to match the rendered image box */
      #hotspotOverlay {
        position: absolute;
        left: 0;
        top: 0;
        width: 0;
        height: 0;
        pointer-events: none;
      }

      .hotspot {
        position: absolute;
        width: 18px;
        height: 18px;
        border-radius: 999px;
        background: var(--hotspot-bg);
        border: 1px solid var(--hotspot-border);
        box-shadow: var(--hotspot-shadow);
        transform: translate(-50%, -50%);
        pointer-events: auto;
        cursor: pointer;
      }

      .hotspot:before {
        content: "";
        position: absolute;
        inset: 5px;
        border-radius: 999px;
        background: rgba(0, 0, 0, 0.35);
        opacity: 0.55;
      }

      .hotspot:focus {
        outline: 2px solid rgba(255, 255, 255, 0.85);
        outline-offset: 3px;
      }

      .ui {
        position: fixed;
        left: env(safe-area-inset-left, 0px);
        top: env(safe-area-inset-top, 0px);
        z-index: 10;
        padding: 14px;
        pointer-events: none;
      }

      .ui-panel {
        pointer-events: auto;
        display: inline-flex;
        flex-direction: column;
        gap: 8px;
        padding: 10px 12px;
        border-radius: 12px;
        background: var(--ui-bg);
        border: 1px solid var(--ui-border);
        box-shadow: var(--ui-shadow);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        max-width: min(92vw, 520px);
      }

      .ui-title {
        margin: 0;
        font-weight: 600;
        font-size: 13px;
        line-height: 1.2;
        color: var(--ui-text);
        user-select: none;
      }

      .ui-hint {
        margin: 0;
        font-size: 12px;
        line-height: 1.2;
        color: rgba(255, 255, 255, 0.66);
        user-select: none;
      }

      .error {
        position: fixed;
        left: 50%;
        top: calc(env(safe-area-inset-top, 0px) + 12px);
        transform: translateX(-50%);
        z-index: 20;
        width: min(92vw, 820px);
        background: rgba(140, 20, 20, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: var(--ui-shadow);
        display: none;
      }

      .error strong {
        display: block;
        margin-bottom: 4px;
        font-size: 13px;
      }

      .error span {
        font-size: 12px;
        opacity: 0.95;
      }

      @media (max-width: 480px) {
        .ui { padding: 10px; }
        .ui-panel { padding: 10px; max-width: 94vw; }
      }
    </style>
  </head>

  <body>
    <div id="stage">
      <div id="mediaWrap">
        <img id="mainImage" alt="Scene" />
        <div id="hotspotOverlay" aria-hidden="true"></div>
      </div>
    </div>

    <div class="ui">
      <div class="ui-panel">
        <p class="ui-title">Photo tour</p>
        <p class="ui-hint">Tap hotspots to move. No dragging / no 360.</p>
      </div>
    </div>

    <div id="errorBanner" class="error" role="alert" aria-live="polite">
      <strong>Viewer error</strong>
      <span id="errorText"></span>
    </div>

    <script>
      function byId(id) {
        return document.getElementById(id);
      }

      function showError(message) {
        var banner = byId("errorBanner");
        var text = byId("errorText");
        text.textContent = message;
        banner.style.display = "block";
      }

      function hideError() {
        var banner = byId("errorBanner");
        banner.style.display = "none";
      }

      async function loadScenesFromDataUrl(dataUrl) {
        var res = await fetch(dataUrl, { method: "GET" });
        if (!res.ok) {
          throw new Error(
            "Failed to fetch data URL. Response was [" + res.status + "] " + res.statusText,
          );
        }

        var json = await res.json();
        var scenes = json && Array.isArray(json.scenes) ? json.scenes : null;
        if (!scenes) {
          throw new Error(
            "Invalid JSON: expected { scenes: [ { id, image, hotspots: [ { id, x, y, target_image } ] } ] }",
          );
        }

        // Normalize without inventing missing data.
        var safeScenes = scenes
          .filter(function (s) {
            return s && typeof s === "object";
          })
          .map(function (s) {
            var id = typeof s.id === "string" && s.id.trim() ? s.id.trim() : null;
            var image = typeof s.image === "string" ? s.image.trim() : "";
            var hotspotsIn = Array.isArray(s.hotspots) ? s.hotspots : [];
            var hotspots = hotspotsIn
              .filter(function (h) {
                return h && typeof h === "object";
              })
              .map(function (h) {
                return {
                  id: typeof h.id === "string" && h.id.trim() ? h.id.trim() : null,
                  x: Number(h.x),
                  y: Number(h.y),
                  target_image:
                    typeof h.target_image === "string" ? h.target_image.trim() : "",
                };
              })
              .filter(function (h) {
                // Render only what the JSON meaningfully provides.
                return (
                  Number.isFinite(h.x) &&
                  Number.isFinite(h.y) &&
                  h.target_image &&
                  h.x >= 0 &&
                  h.x <= 100 &&
                  h.y >= 0 &&
                  h.y <= 100
                );
              });

            return { id: id, image: image, hotspots: hotspots };
          })
          .filter(function (s) {
            return !!s.image;
          });

        if (!safeScenes.length) {
          throw new Error("No valid scenes found.");
        }

        return { scenes: safeScenes };
      }

      function clearHotspots() {
        var overlay = byId("hotspotOverlay");
        overlay.innerHTML = "";
      }

      function syncOverlayToImageBox() {
        var wrap = byId("mediaWrap");
        var img = byId("mainImage");
        var overlay = byId("hotspotOverlay");

        var wrapRect = wrap.getBoundingClientRect();
        var imgRect = img.getBoundingClientRect();

        // If image not laid out yet, bail.
        if (!imgRect.width || !imgRect.height) return;

        overlay.style.left = (imgRect.left - wrapRect.left) + "px";
        overlay.style.top = (imgRect.top - wrapRect.top) + "px";
        overlay.style.width = imgRect.width + "px";
        overlay.style.height = imgRect.height + "px";
      }

      function renderScene(opts) {
        var scenesData = opts.scenesData;
        var currentScene = opts.currentScene;

        var img = byId("mainImage");
        var overlay = byId("hotspotOverlay");

        if (!currentScene || !currentScene.image) {
          showError("Missing current scene image.");
          return;
        }

        // Update image
        img.src = currentScene.image;

        // Wait for image to layout to position overlay + hotspots.
        var doRenderHotspots = function () {
          hideError();
          syncOverlayToImageBox();

          clearHotspots();

          var hs = Array.isArray(currentScene.hotspots) ? currentScene.hotspots : [];
          for (var i = 0; i < hs.length; i++) {
            (function () {
              var h = hs[i];

              var el = document.createElement("button");
              el.type = "button";
              el.className = "hotspot";
              el.style.left = h.x + "%";
              el.style.top = h.y + "%";
              el.setAttribute("aria-label", "Go to next photo");

              el.addEventListener("click", function () {
                if (!scenesData || !Array.isArray(scenesData.scenes)) {
                  showError("Scenes are not loaded.");
                  return;
                }

                var target = String(h.target_image || "").trim();
                if (!target) return;

                var next = scenesData.scenes.find(function (s) {
                  return s && typeof s.image === "string" && s.image.trim() === target;
                });

                if (!next) {
                  showError(
                    "Target scene not found for hotspot. (target_image did not match any scene.image)",
                  );
                  return;
                }

                // Update state
                window.currentScene = next;
                renderScene({ scenesData: scenesData, currentScene: next });
              });

              overlay.appendChild(el);
            })();
          }
        };

        img.onload = function () {
          doRenderHotspots();
        };

        img.onerror = function () {
          showError("Could not load scene image.");
        };
      }

      (async function boot() {
        hideError();

        // State (as requested)
        window.scenesData = null;
        window.currentScene = null;

        var params = new URLSearchParams(window.location.search);
        var dataUrl = params.get("data");

        if (!dataUrl) {
          showError("Missing data URL. Provide ?data=<url-to-json>.");
          return;
        }

        try {
          // 1) Store the API response in a state called scenesData
          window.scenesData = await loadScenesFromDataUrl(dataUrl);

          // 2) Create another state called currentScene
          // 3) On load, set currentScene = scenesData.scenes[0]
          window.currentScene = window.scenesData.scenes[0];

          renderScene({ scenesData: window.scenesData, currentScene: window.currentScene });

          // Keep hotspot alignment correct on resize/orientation changes.
          window.addEventListener("resize", function () {
            syncOverlayToImageBox();
          });
        } catch (e) {
          console.error(e);
          showError(e && e.message ? e.message : String(e));
        }
      })();
    </script>
  </body>
</html>`;

export async function GET() {
  return new Response(HTML, {
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}
