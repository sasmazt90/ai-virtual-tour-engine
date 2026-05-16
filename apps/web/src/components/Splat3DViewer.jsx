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
  const min = new THREE.Vector3(Infinity, Infinity, Infinity);
  const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
  const step = Math.max(1, Math.floor(count / 2500));

  for (let i = 0; i < count; i += step) {
    splatMesh.getSplatCenter(i, sample, true);
    min.min(sample);
    max.max(sample);
  }

  if (
    !Number.isFinite(min.x) ||
    !Number.isFinite(min.y) ||
    !Number.isFinite(min.z) ||
    !Number.isFinite(max.x) ||
    !Number.isFinite(max.y) ||
    !Number.isFinite(max.z)
  ) {
    return count;
  }

  const box = new THREE.Box3(min, max);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.length() * 0.55, 2);

  if (viewer.camera) {
    viewer.camera.position.set(center.x, center.y - radius * 2.2, center.z + radius * 0.7);
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
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");

  const payload = useMemo(() => tourPayload || {}, [tourPayload]);
  const fileUrl = payload.fileUrl || payload.url || payload.src || "";
  const format = payload.format || "";
  const camera = payload.camera || {};
  const containerHeight = height ?? 480;

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const root = rootRef.current;
      if (!root || !fileUrl) {
        setStatus("error");
        setError("No 3D scan file is attached to this tour.");
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
          cameraUp: safeVector(camera.up, [0, -1, -0.6]),
          initialCameraPosition: safeVector(camera.position, [-1, -4, 6]),
          initialCameraLookAt: safeVector(camera.lookAt, [0, 0, 0]),
          sharedMemoryForWorkers: false,
          gpuAcceleratedSort: false,
          integerBasedSort: false,
          ignoreDevicePixelRatio: true,
          sphericalHarmonicsDegree: 0,
        });

        viewerRef.current = viewer;

        const sceneOptions = {
          showLoadingUI: true,
          progressiveLoad: false,
          splatAlphaRemovalThreshold: 0,
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
          throw new Error("The 3D scan file did not contain renderable points.");
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
              : "Could not load the 3D scan.",
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
    </div>
  );
}
