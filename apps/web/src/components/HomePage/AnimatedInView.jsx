import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { useInViewOnce } from "@/hooks/useInViewOnce";

export function AnimatedInView({ children, delay = 0, className = "", style }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const inViewOptions = useMemo(() => {
    return { threshold: 0.18, rootMargin: "0px 0px -10% 0px" };
  }, []);

  const { ref, inView } = useInViewOnce(inViewOptions);

  // On the server / before mount, don't apply motion initial styles.
  const initial = useMemo(() => {
    return mounted ? { opacity: 0, y: 18 } : false;
  }, [mounted]);

  const animate = useMemo(() => {
    if (!mounted) {
      return undefined;
    }
    return inView ? { opacity: 1, y: 0 } : undefined;
  }, [inView, mounted]);

  const transition = useMemo(() => {
    if (!mounted) {
      return undefined;
    }
    return { duration: 0.7, ease: [0.22, 1, 0.36, 1], delay };
  }, [delay, mounted]);

  const resolvedRef = mounted ? ref : undefined;

  return (
    <motion.div
      ref={resolvedRef}
      initial={initial}
      animate={animate}
      transition={transition}
      className={className}
      style={style}
    >
      {children}
    </motion.div>
  );
}
