import { useState, useEffect, useMemo, useCallback } from "react";
import { clamp } from "@/utils/fake360Utils";

export function useFake360LegacyScenes(
  effectivePayload,
  isFramesMode,
  isPointsMode,
) {
  const scenes =
    !isFramesMode && !isPointsMode ? effectivePayload?.scenes || [] : [];

  const [activeSceneId, setActiveSceneId] = useState(
    scenes?.[0]?.sceneId || null,
  );
  const [yaw, setYaw] = useState(0); // -1..1
  const [dragging, setDragging] = useState(false);
  const [transitioning, setTransitioning] = useState(false);

  useEffect(() => {
    if (isFramesMode || isPointsMode) return;
    if (!activeSceneId && scenes.length > 0) {
      setActiveSceneId(scenes[0].sceneId);
    }
  }, [activeSceneId, scenes, isFramesMode, isPointsMode]);

  const activeScene = useMemo(() => {
    if (isFramesMode || isPointsMode) return null;
    return scenes.find((s) => s.sceneId === activeSceneId) || null;
  }, [activeSceneId, scenes, isFramesMode, isPointsMode]);

  // Honor initialYaw from payload (degrees) when scene changes.
  useEffect(() => {
    if (isFramesMode || isPointsMode) return;
    if (!activeScene) return;
    const degrees = Number(activeScene.initialYaw || 0);
    const normalized = clamp(degrees / 180, -1, 1);
    setYaw(normalized);
  }, [activeSceneId, activeScene, isFramesMode, isPointsMode]);

  const goToScene = useCallback(
    (sceneId) => {
      if (!sceneId) return;
      if (sceneId === activeSceneId) return;

      setTransitioning(true);
      setTimeout(() => {
        setActiveSceneId(sceneId);
        setTransitioning(false);
      }, 180);
    },
    [activeSceneId],
  );

  return {
    scenes,
    activeSceneId,
    activeScene,
    yaw,
    setYaw,
    dragging,
    setDragging,
    transitioning,
    goToScene,
  };
}
