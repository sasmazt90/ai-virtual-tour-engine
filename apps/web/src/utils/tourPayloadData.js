export function getTourPayload(stagedShowcase) {
  const img = stagedShowcase;

  return {
    scenes: [
      {
        sceneId: "A0",
        imageUrl: img,
        initialYaw: 0,
        hotspots: [
          {
            x: 0.2,
            y: 0.62,
            toSceneId: "A-90",
            direction: "left",
          },
          {
            x: 0.8,
            y: 0.62,
            toSceneId: "A90",
            direction: "right",
          },
          {
            x: 0.5,
            y: 0.62,
            toSceneId: "B0",
            direction: "forward",
          },
        ],
      },
      {
        sceneId: "A90",
        imageUrl: img,
        initialYaw: 90,
        hotspots: [
          {
            x: 0.2,
            y: 0.62,
            toSceneId: "A0",
            direction: "left",
          },
          {
            x: 0.8,
            y: 0.62,
            toSceneId: "A180",
            direction: "right",
          },
          {
            x: 0.5,
            y: 0.62,
            toSceneId: "B90",
            direction: "forward",
          },
        ],
      },
      {
        sceneId: "A180",
        imageUrl: img,
        initialYaw: 180,
        hotspots: [
          {
            x: 0.2,
            y: 0.62,
            toSceneId: "A90",
            direction: "left",
          },
          {
            x: 0.8,
            y: 0.62,
            toSceneId: "A-90",
            direction: "right",
          },
          {
            x: 0.5,
            y: 0.62,
            toSceneId: "B180",
            direction: "forward",
          },
        ],
      },
      {
        sceneId: "A-90",
        imageUrl: img,
        initialYaw: -90,
        hotspots: [
          {
            x: 0.2,
            y: 0.62,
            toSceneId: "A180",
            direction: "left",
          },
          {
            x: 0.8,
            y: 0.62,
            toSceneId: "A0",
            direction: "right",
          },
          {
            x: 0.5,
            y: 0.62,
            toSceneId: "B-90",
            direction: "forward",
          },
        ],
      },
      {
        sceneId: "B0",
        imageUrl: img,
        initialYaw: 0,
        hotspots: [
          {
            x: 0.2,
            y: 0.62,
            toSceneId: "B-90",
            direction: "left",
          },
          {
            x: 0.8,
            y: 0.62,
            toSceneId: "B90",
            direction: "right",
          },
          {
            x: 0.5,
            y: 0.76,
            toSceneId: "A0",
            direction: "back",
          },
        ],
      },
      {
        sceneId: "B90",
        imageUrl: img,
        initialYaw: 90,
        hotspots: [
          {
            x: 0.2,
            y: 0.62,
            toSceneId: "B0",
            direction: "left",
          },
          {
            x: 0.8,
            y: 0.62,
            toSceneId: "B180",
            direction: "right",
          },
          {
            x: 0.5,
            y: 0.76,
            toSceneId: "A90",
            direction: "back",
          },
        ],
      },
      {
        sceneId: "B180",
        imageUrl: img,
        initialYaw: 180,
        hotspots: [
          {
            x: 0.2,
            y: 0.62,
            toSceneId: "B90",
            direction: "left",
          },
          {
            x: 0.8,
            y: 0.62,
            toSceneId: "B-90",
            direction: "right",
          },
          {
            x: 0.5,
            y: 0.76,
            toSceneId: "A180",
            direction: "back",
          },
        ],
      },
      {
        sceneId: "B-90",
        imageUrl: img,
        initialYaw: -90,
        hotspots: [
          {
            x: 0.2,
            y: 0.62,
            toSceneId: "B180",
            direction: "left",
          },
          {
            x: 0.8,
            y: 0.62,
            toSceneId: "B0",
            direction: "right",
          },
          {
            x: 0.5,
            y: 0.76,
            toSceneId: "A-90",
            direction: "back",
          },
        ],
      },
    ],
  };
}

// NEW: Polycam-like preview payload for the marketing homepage.
// This uses multiple angles (frames) and supports moving between "points" via floor markers.
export function getPolycamPreviewPayload() {
  const frames = [
    "https://ucarecdn.com/4afa8729-a232-4536-8bc6-bc4684c08927/-/format/auto/",
    "https://ucarecdn.com/7042433d-ad54-4066-b269-a86b6c40c25d/-/format/auto/",
    "https://ucarecdn.com/e13d4cb0-4968-4603-95d9-165d54cdb1d7/-/format/auto/",
    "https://ucarecdn.com/966132bc-9376-4f07-9487-ae81d11efe7e/-/format/auto/",
    "https://ucarecdn.com/a883bb4c-215d-4f7f-9008-ca7bf6763dad/-/format/auto/",
    "https://ucarecdn.com/93ab76b1-ac3b-438e-89c1-2e807292477e/-/format/auto/",
    "https://ucarecdn.com/cf078260-095b-4ced-adfe-8e51267b7cd0/-/format/auto/",
    "https://ucarecdn.com/033ea138-eddb-472b-9d60-1a07eac6557b/-/format/auto/",
    "https://ucarecdn.com/f134b576-0145-4e3f-b3d9-628bde7f05fd/-/format/auto/",
    "https://ucarecdn.com/2c162fc6-3233-4461-8f57-0d73ef733607/-/format/auto/",
    "https://ucarecdn.com/acaf74ca-dcc4-4665-a7d5-d348f69a0a10/-/format/auto/",
    "https://ucarecdn.com/5671bb56-1135-4b37-8dcb-512e327ffc3d/-/format/auto/",
    "https://ucarecdn.com/1a875b5d-790a-4c2f-afa2-586120a277fb/-/format/auto/",
    "https://ucarecdn.com/4bf3a6d3-07f5-49ea-8a05-40adf53637fc/-/format/auto/",
    "https://ucarecdn.com/a2da65b2-40e1-4ee6-95c9-02cb2a06f633/-/format/auto/",
    "https://ucarecdn.com/c1ebf752-0f8d-4b2e-ad10-4828ac631ffd/-/format/auto/",
    "https://ucarecdn.com/84d48b1c-81df-4d42-9fc6-f0f7025d262b/-/format/auto/",
    "https://ucarecdn.com/a860ce7d-6926-4dca-8be5-745c80157ac3/-/format/auto/",
    "https://ucarecdn.com/f10b1e23-b086-473a-b967-d1aa0b384602/-/format/auto/",
    "https://ucarecdn.com/7f899452-c20f-4c1d-bfc9-0f084acb3b10/-/format/auto/",
    "https://ucarecdn.com/d84be1b0-598f-4b38-a3c6-4780cac59dad/-/format/auto/",
  ];

  return {
    initialPointId: "P1",
    points: [
      {
        pointId: "P1",
        frames,
        initialIndex: 0,
        hotspots: [{ x: 0.5, y: 0.66, toPointId: "P2", direction: "forward" }],
      },
      {
        pointId: "P2",
        frames,
        initialIndex: 6,
        hotspots: [{ x: 0.5, y: 0.78, toPointId: "P1", direction: "back" }],
      },
    ],
  };
}
