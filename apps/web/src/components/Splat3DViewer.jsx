import { useEffect, useMemo, useRef, useState } from "react";

function sceneFormatFromExtension(GaussianSplats3D, rawFormat) {
  const format = String(rawFormat || "").toLowerCase();
  const sceneFormat = GaussianSplats3D?.SceneFormat;

  if (!sceneFormat) return undefined;
  if (format === "ply") return sceneFormat.Ply;
  if (format === "splat") return sceneFormat.Splat;
  if (format === "ksplat") return sceneFormat.KSplat;
  return undefined;
}

function safeVector(raw, fallback) {
  if (!Array.isArray(raw)) return fallback;
  const next = raw.slice(0, 3).map((v) => Number(v));
  return next.every((v) => Number.isFinite(v)) ? next : fallback;
}

function safeNumber(raw, fallback, min, max) {
  const value = Number(raw);
  if (!Number.isFinite(value)) return fallback;
  return Math.max(min, Math.min(max, value));
}

async function fitCameraToScene(viewer) {
  const splatMesh =
    typeof viewer?.getSplatMesh === "function" ? viewer.getSplatMesh() : null;
  const count =
    typeof splatMesh?.getSplatCount === "function"
      ? Number(splatMesh.getSplatCount() || 0)
      : 0;

  if (!splatMesh || !Number.isFinite(count) || count <= 0) return count;

  const THREE = await import("three");
  const sample = new THREE.Vector3();
  const xs = [];
  const ys = [];
  const zs = [];
  const step = Math.max(1, Math.floor(count / 6000));

  for (let i = 0; i < count; i += step) {
    splatMesh.getSplatCenter(i, sample, true);
    if (
      Number.isFinite(sample.x) &&
      Number.isFinite(sample.y) &&
      Number.isFinite(sample.z)
    ) {
      xs.push(sample.x);
      ys.push(sample.y);
      zs.push(sample.z);
    }
  }

  if (xs.length < 10) {
    return count;
  }

  const pick = (values, q) => {
    values.sort((a, b) => a - b);
    return values[Math.max(0, Math.min(values.length - 1, Math.floor(values.length * q)))];
  };

  // Gaussian splat exports often contain a few distant outliers. Fitting the
  // camera to the full min/max bounds can place the room far away or inside a
  // blurred outlier cloud, so fit to the central 90% of sampled points.
  const min = new THREE.Vector3(pick(xs, 0.05), pick(ys, 0.05), pick(zs, 0.05));
  const max = new THREE.Vector3(pick(xs, 0.95), pick(ys, 0.95), pick(zs, 0.95));
  const box = new THREE.Box3(min, max);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 1);
  const fov = viewer.camera?.fov ? (viewer.camera.fov * Math.PI) / 180 : Math.PI / 4;
  const distance = Math.max(maxDim / (2 * Math.tan(fov / 2)), 2) * 1.65;
  const viewDirection = new THREE.Vector3(0.35, -1, 0.35).normalize();

  if (viewer.camera) {
    viewer.camera.near = Math.max(0.01, distance / 500);
    viewer.camera.far = Math.max(1000, distance * 100);
    viewer.camera.position.copy(center.clone().add(viewDirection.multiplyScalar(distance)));
    viewer.camera.lookAt(center);
    viewer.camera.updateProjectionMatrix?.();
  }

  if (viewer.controls) {
    viewer.controls.target.copy(center);
    viewer.controls.update?.();
  }

  return count;
}

export default function Splat3DViewer({ tourPayload, height }) {
  const rootRef = useRef(null);
  const viewerRef = useRef(null);
  const [activeSceneIndex, setActiveSceneIndex] = useState(0);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");

  const payload = useMemo(() => tourPayload || {}, [tourPayload]);
  const scenes = useMemo(() => {
    const list = Array.isArray(payload.scenes)
      ? payload.scenes.filter((scene) => scene?.fileUrl || scene?.url || scene?.src)
      : [];

    if (list.length) return list;
    return [
      {
        title: "Original tour",
        fileUrl: payload.fileUrl || payload.url || payload.src || "",
        format: payload.format || "",
        camera: payload.camera || {},
      },
    ];
  }, [payload]);
  const activeScene = scenes[Math.min(activeSceneIndex, scenes.length - 1)] || scenes[0] || {};
  const fileUrl = activeScene.fileUrl || activeScene.url || activeScene.src || "";
  const format = activeScene.format || payload.format || "";
  const camera = activeScene.camera || payload.camera || {};
  const containerHeight = height ?? 480;

  useEffect(() => {
    setActiveSceneIndex(0);
  }, [payload]);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const root = rootRef.current;
      if (!root || !fileUrl) {
        setStatus("error");
        setError("No 3D tour file is attached to this tour.");
        return;
      }

      setStatus("loading");
      setError("");
      root.innerHTML = "";

      try {
        const GaussianSplats3D = await import(
          "@mkkellogg/gaussian-splats-3d"
        );

        if (cancelled) return;

        const viewer = new GaussianSplats3D.Viewer({
          rootElement: root,
          cameraUp: safeVector(camera.up, [0, -1, 0]),
          initialCameraPosition: safeVector(camera.position, [0, -4, 2]),
          initialCameraLookAt: safeVector(camera.lookAt, [0, 0, 0]),
          sharedMemoryForWorkers: false,
          gpuAcceleratedSort: false,
          integerBasedSort: false,
          ignoreDevicePixelRatio: false,
          optimizeSplatData: true,
          inMemoryCompressionLevel: 1,
          focalAdjustment: 1.15,
          sphericalHarmonicsDegree: 0,
        });

        viewerRef.current = viewer;

        const sceneOptions = {
          showLoadingUI: true,
          progressiveLoad: false,
          splatAlphaRemovalThreshold: safeNumber(
            activeScene.alphaRemovalThreshold ?? payload.alphaRemovalThreshold,
            8,
            1,
            40,
          ),
          position: [0, 0, 0],
          rotation: [0, 0, 0, 1],
          scale: [1, 1, 1],
        };

        const sceneFormat = sceneFormatFromExtension(GaussianSplats3D, format);
        if (sceneFormat) {
          sceneOptions.format = sceneFormat;
        }

        await viewer.addSplatScene(fileUrl, sceneOptions);

        const splatCount = await fitCameraToScene(viewer);

        if (!Number.isFinite(splatCount) || splatCount <= 0) {
          throw new Error("The 3D tour file did not contain renderable points.");
        }

        if (cancelled) {
          await viewer.dispose();
          return;
        }

        viewer.start();
        setStatus("ready");
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setStatus("error");
          setError(
            err instanceof Error
              ? err.message
              : "Could not load the 3D tour.",
          );
        }
      }
    }

    load();

    return () => {
      cancelled = true;
      const viewer = viewerRef.current;
      viewerRef.current = null;
      if (viewer) {
        viewer.dispose().catch(() => {});
      }
    };
  }, [camera.lookAt, camera.position, camera.up, fileUrl, format]);

  return (
    <div
      className="relative overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700 bg-black"
      style={{ height: containerHeight }}
    >
      <div ref={rootRef} className="absolute inset-0" />

      {status === "loading" ? (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60 text-sm text-white font-jetbrains-mono">
          Loading 3D tour...
        </div>
      ) : null}

      {status === "error" ? (
        <div className="absolute inset-0 flex items-center justify-center p-6 text-center bg-black text-sm text-white font-jetbrains-mono">
          {error || "Could not load the 3D tour."}
        </div>
      ) : null}

      {status === "ready" ? (
        <div className="pointer-events-none absolute left-3 bottom-3 rounded-md bg-black/55 px-3 py-2 text-xs text-white font-jetbrains-mono">
          Drag to orbit - right-drag to pan - scroll to zoom
        </div>
      ) : null}

      {status === "ready" && scenes.length > 1 ? (
        <div className="absolute left-3 top-3 flex max-w-[calc(100%-1.5rem)] flex-wrap gap-2">
          {scenes.map((scene, index) => (
            <button
              key={`${scene.key || scene.title || index}-${index}`}
              type="button"
              onClick={() => setActiveSceneIndex(index)}
              className={`rounded-md px-3 py-2 text-xs font-medium font-jetbrains-mono shadow-sm ${
                index === activeSceneIndex
                  ? "bg-white text-gray-900"
                  : "bg-black/55 text-white hover:bg-black/70"
              }`}
            >
              {scene.title || `Area ${index + 1}`}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
