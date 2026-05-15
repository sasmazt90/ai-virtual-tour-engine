import { useState } from "react";

export function useFake360Parallax() {
  const [parallax, setParallax] = useState({ x: 0, y: 0 });

  return { parallax, setParallax };
}
