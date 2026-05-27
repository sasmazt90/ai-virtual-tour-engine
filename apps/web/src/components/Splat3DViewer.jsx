import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Home,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

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

function isFiniteVector(raw) {
  return (
    Array.isArray(raw) &&
    raw.length >= 3 &&
    raw.slice(0, 3).every((value) => Number.isFinite(Number(value)))
  );
}

function configureOrbitControls(controls) {
  if (!controls) return;
  controls.enableDamping = true;
  controls.dampingFactor = 0.14;
  controls.enableRotate = false;
  controls.zoomSpeed = 0.55;
  controls.panSpeed = 0.45;
  controls.enablePan = true;
  controls.screenSpacePanning = true;
  controls.update?.();
}

function cameraStateFromViewer(viewer) {
  const camera = viewer?.camera;
  if (!camera) return null;
  return {
    position: camera.position.clone(),
    up: camera.up.clone(),
    fov: Number(camera.fov || 0) || null,
    target: viewer.controls?.target?.clone?.() || null,
  };
}

function applyFirstPersonLook(viewer, yawDelta, pitchDelta) {
  const camera = viewer?.camera;
  if (!camera) return;

  const controls = viewer.controls;
  const target = controls?.target;
  if (!target) return;

  const forward = target.clone().sub(camera.position);
  const distance = Math.max(0.4, forward.length());
  const up = camera.up.clone().normalize();
  const right = forward.clone().normalize().cross(up).normalize();

  if (!Number.isFinite(right.x) || right.lengthSq() < 0.0001) return;

  forward.normalize();
  if (yawDelta) {
    forward.applyAxisAngle(up, yawDelta);
  }
  if (pitchDelta) {
    const candidate = forward.clone().applyAxisAngle(right, pitchDelta).normalize();
    if (Math.abs(candidate.dot(up)) < 0.96) {
      forward.copy(candidate);
    }
  }

  const nextTarget = camera.position.clone().add(forward.multiplyScalar(distance));
  controls.target.copy(nextTarget);
  camera.lookAt(nextTarget);
  camera.updateProjectionMatrix?.();
  controls.update?.();
  viewer.forceRenderNextFrame?.();
}

function moveViewerCamera(viewer, action, resetState) {
  const camera = viewer?.camera;
  if (!camera) return;

  const controls = viewer.controls;
  const target = controls?.target;

  if (action === "reset" && resetState?.position) {
    camera.position.copy(resetState.position);
    camera.up.copy(resetState.up);
    if (resetState.fov && camera.fov) {
      camera.fov = resetState.fov;
    }
    if (controls && resetState.target) {
      controls.target.copy(resetState.target);
    }
    camera.lookAt(resetState.target || target || camera.position.clone().add(camera.getWorldDirection(camera.up.clone())));
    camera.updateProjectionMatrix?.();
    controls?.update?.();
    viewer.forceRenderNextFrame?.();
    return;
  }

  if (action === "left") applyFirstPersonLook(viewer, 0.12, 0);
  if (action === "right") applyFirstPersonLook(viewer, -0.12, 0);
  if (action === "up") applyFirstPersonLook(viewer, 0, 0.08);
  if (action === "down") applyFirstPersonLook(viewer, 0, -0.08);

  if (action === "zoomIn" || action === "zoomOut") {
    const nextFov =
      action === "zoomIn"
        ? Math.max(24, Number(camera.fov || 48) * 0.9)
        : Math.min(85, Number(camera.fov || 48) * 1.1);
    camera.fov = nextFov;
    camera.updateProjectionMatrix?.();
    controls?.update?.();
    viewer.forceRenderNextFrame?.();
  }
}

async function fitCameraToScene(viewer, preferredCamera) {
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
  const hasPreferredCamera =
    isFiniteVector(preferredCamera?.position) &&
    isFiniteVector(preferredCamera?.lookAt);

  if (hasPreferredCamera) {
    const position = new THREE.Vector3(
      ...preferredCamera.position.slice(0, 3).map(Number),
    );
    const target = new THREE.Vector3(
      ...preferredCamera.lookAt.slice(0, 3).map(Number),
    );

    if (viewer.camera) {
      viewer.camera.near = Math.max(0.01, distance / 700);
      viewer.camera.far = Math.max(1000, distance * 120);
      viewer.camera.position.copy(position);
      viewer.camera.lookAt(target);
      viewer.camera.updateProjectionMatrix?.();
    }

    if (viewer.controls) {
      viewer.controls.target.copy(target);
      configureOrbitControls(viewer.controls);
      viewer.controls.update?.();
    }

    return count;
  }

  if (viewer.camera) {
    viewer.camera.near = Math.max(0.01, distance / 500);
    viewer.camera.far = Math.max(1000, distance * 100);
    viewer.camera.position.copy(center.clone().add(viewDirection.multiplyScalar(distance)));
    viewer.camera.lookAt(center);
    viewer.camera.updateProjectionMatrix?.();
  }

  if (viewer.controls) {
    viewer.controls.target.copy(center);
    configureOrbitControls(viewer.controls);
    viewer.controls.update?.();
  }

  return count;
}

async function renderFlatRoomScene(root) {
  const THREE = await import("three");
  const { OrbitControls } = await import(
    "three/examples/jsm/controls/OrbitControls.js"
  );

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111318);

  const width = root.clientWidth || 960;
  const height = root.clientHeight || 540;
  const camera = new THREE.PerspectiveCamera(48, width / height, 0.01, 100);
  camera.position.set(3.2, 2.2, 4.2);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.setSize(width, height);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  root.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 1.15, 0);
  configureOrbitControls(controls);
  controls.minDistance = 2.1;
  controls.maxDistance = 8;
  controls.update();

  const wallMaterial = new THREE.MeshStandardMaterial({
    color: 0xd8d4cc,
    roughness: 0.92,
    metalness: 0.0,
  });
  const sideMaterial = new THREE.MeshStandardMaterial({
    color: 0xc9c5bd,
    roughness: 0.95,
    metalness: 0.0,
  });
  const floorMaterial = new THREE.MeshStandardMaterial({
    color: 0xa69f94,
    roughness: 0.85,
    metalness: 0.0,
  });

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 5), floorMaterial);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, 0, 0.35);
  scene.add(floor);

  const backWall = new THREE.Mesh(new THREE.PlaneGeometry(6, 2.8), wallMaterial);
  backWall.position.set(0, 1.4, -2.15);
  scene.add(backWall);

  const leftWall = new THREE.Mesh(new THREE.PlaneGeometry(5, 2.8), sideMaterial);
  leftWall.rotation.y = Math.PI / 2;
  leftWall.position.set(-3, 1.4, 0.35);
  scene.add(leftWall);

  const rightWall = new THREE.Mesh(new THREE.PlaneGeometry(5, 2.8), sideMaterial);
  rightWall.rotation.y = -Math.PI / 2;
  rightWall.position.set(3, 1.4, 0.35);
  scene.add(rightWall);

  const grid = new THREE.GridHelper(6, 12, 0x5d6470, 0x3a4048);
  grid.position.y = 0.006;
  scene.add(grid);

  const ambient = new THREE.HemisphereLight(0xffffff, 0x2b3038, 2.3);
  scene.add(ambient);

  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(3, 4, 3);
  scene.add(key);

  let frameId = 0;
  let disposed = false;

  function resize() {
    if (disposed) return;
    const nextWidth = root.clientWidth || width;
    const nextHeight = root.clientHeight || height;
    camera.aspect = nextWidth / nextHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(nextWidth, nextHeight);
  }

  function animate() {
    if (disposed) return;
    controls.update();
    renderer.render(scene, camera);
    frameId = window.requestAnimationFrame(animate);
  }

  window.addEventListener("resize", resize);
  animate();

  return {
    camera,
    controls,
    forceRenderNextFrame() {},
    dispose() {
      disposed = true;
      window.removeEventListener("resize", resize);
      if (frameId) window.cancelAnimationFrame(frameId);
      controls.dispose();
      scene.traverse((object) => {
        if (object.geometry) object.geometry.dispose?.();
        const material = object.material;
        if (Array.isArray(material)) {
          material.forEach((item) => item.dispose?.());
        } else {
          material?.dispose?.();
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}

export default function Splat3DViewer({ tourPayload, height }) {
  const rootRef = useRef(null);
  const viewerRef = useRef(null);
  const initialCameraStateRef = useRef(null);
  const dragStateRef = useRef(null);
  const [activeSceneIndex, setActiveSceneIndex] = useState(0);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");

  const payload = useMemo(() => tourPayload || {}, [tourPayload]);
  const scenes = useMemo(() => {
    const list = Array.isArray(payload.scenes)
      ? payload.scenes.filter(
          (scene) =>
            scene?.fileUrl ||
            scene?.url ||
            scene?.src ||
            scene?.fallbackType === "flat_room" ||
            scene?.type === "placeholder",
        )
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
  const isPlaceholder =
    activeScene.fallbackType === "flat_room" || activeScene.type === "placeholder";
  const containerHeight = height ?? 480;

  function handlePointerDown(event) {
    if (status !== "ready" || event.button !== 0) return;
    if (event.target?.closest?.("button")) return;

    dragStateRef.current = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
    };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    event.preventDefault();
  }

  function handlePointerMove(event) {
    const dragState = dragStateRef.current;
    if (!dragState || dragState.pointerId !== event.pointerId) return;

    const dx = event.clientX - dragState.x;
    const dy = event.clientY - dragState.y;
    dragState.x = event.clientX;
    dragState.y = event.clientY;

    applyFirstPersonLook(viewerRef.current, -dx * 0.003, -dy * 0.003);
  }

  function handlePointerUp(event) {
    if (dragStateRef.current?.pointerId === event.pointerId) {
      dragStateRef.current = null;
      event.currentTarget.releasePointerCapture?.(event.pointerId);
    }
  }

  useEffect(() => {
    setActiveSceneIndex(0);
  }, [payload]);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const root = rootRef.current;
      if (!root || (!fileUrl && !isPlaceholder)) {
        setStatus("error");
        setError("No 3D tour file is attached to this tour.");
        return;
      }

      setStatus("loading");
      setError("");
      root.innerHTML = "";

      try {
        if (isPlaceholder) {
          const viewer = await renderFlatRoomScene(root);
          if (cancelled) {
            viewer.dispose();
            return;
          }
          viewerRef.current = viewer;
          initialCameraStateRef.current = cameraStateFromViewer(viewer);
          setStatus("ready");
          return;
        }

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
          ignoreDevicePixelRatio: true,
          optimizeSplatData: false,
          inMemoryCompressionLevel: 0,
          focalAdjustment: 1.15,
          sphericalHarmonicsDegree: 0,
        });

        viewerRef.current = viewer;
        configureOrbitControls(viewer.controls);

        const sceneOptions = {
          showLoadingUI: true,
          progressiveLoad: true,
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

        const splatCount = await fitCameraToScene(viewer, camera);

        if (!Number.isFinite(splatCount) || splatCount <= 0) {
          throw new Error("The 3D tour file did not contain renderable points.");
        }

        if (cancelled) {
          await viewer.dispose();
          return;
        }

        viewer.start();
        initialCameraStateRef.current = cameraStateFromViewer(viewer);
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
      initialCameraStateRef.current = null;
      if (viewer) {
        try {
          const result = viewer.dispose();
          result?.catch?.(() => {});
        } catch {
          // Ignore teardown errors while React is replacing the viewer.
        }
      }
    };
  }, [camera.lookAt, camera.position, camera.up, fileUrl, format, isPlaceholder]);

  return (
    <div
      className="relative cursor-grab overflow-hidden rounded-lg border border-gray-200 bg-black active:cursor-grabbing dark:border-gray-700"
      style={{ height: containerHeight }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onPointerLeave={handlePointerUp}
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
        <>
          <div className="absolute right-3 top-3 grid grid-cols-3 gap-1 rounded-md border border-white/15 bg-black/60 p-1 shadow-lg backdrop-blur-sm">
            {[
              { action: "zoomOut", icon: ZoomOut, label: "Zoom out" },
              { action: "up", icon: ArrowUp, label: "Look up" },
              { action: "zoomIn", icon: ZoomIn, label: "Zoom in" },
              { action: "left", icon: ArrowLeft, label: "Look left" },
              { action: "reset", icon: Home, label: "Reset view" },
              { action: "right", icon: ArrowRight, label: "Look right" },
              null,
              { action: "down", icon: ArrowDown, label: "Look down" },
              null,
            ].map((item, index) => item ? (
              <button
                key={item.label}
                type="button"
                aria-label={item.label}
                title={item.label}
                onClick={() =>
                  moveViewerCamera(
                    viewerRef.current,
                    item.action,
                    initialCameraStateRef.current,
                  )
                }
                className="flex h-9 w-9 items-center justify-center rounded-md border border-white/10 bg-white/10 text-white transition hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-amber-400"
              >
                <item.icon className="h-4 w-4" aria-hidden="true" />
              </button>
            ) : (
              <div key={`spacer-${index}`} className="h-9 w-9" aria-hidden="true" />
            ))}
          </div>
          <div className="pointer-events-none absolute left-3 bottom-3 rounded-md bg-black/55 px-3 py-2 text-xs text-white font-jetbrains-mono">
            Drag to look around from your current position - use buttons for precise navigation
          </div>
        </>
      ) : null}

      {status === "ready" && isPlaceholder ? (
        <div className="pointer-events-none absolute right-3 bottom-3 max-w-xs rounded-md bg-black/60 px-3 py-2 text-xs leading-relaxed text-white font-jetbrains-mono">
          {activeScene.message ||
            "This area is shown as a clean room outline because the scan needs clearer video."}
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
