import React, { useEffect, useRef } from "react";
import { Viewer, Ion, BingMapsImageryProvider } from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";

function CesiumMap() {
  const viewerRef = useRef(null);

  useEffect(() => {
    if (viewerRef.current) {
      // Optional: set your Ion token
      Ion.defaultAccessToken =
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI0NDk0MmJjNC1hNTNiLTQ2MWQtYWNiZS0xZjU0OWQzYzY3OTQiLCJpZCI6MzIxMDE4LCJpYXQiOjE3NTIzOTc2MzB9.1ZJ2zyldeG_EOIQYARpI9J5Sq_vpiOLDqRnf4t_7o2c";

      new Viewer(viewerRef.current, {
        imageryProvider: new BingMapsImageryProvider({
          url: "https://dev.virtualearth.net",
          key: "Aiz...REPLACE_WITH_VALID_BING_KEY",
        }),
        shouldAnimate: true,
      });
    }
  }, []);

  return <div ref={viewerRef} style={{ width: "100%", height: "100vh" }} />;
}

export default CesiumMap;
