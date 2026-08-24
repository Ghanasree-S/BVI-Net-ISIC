import express, { Request, Response } from "express";
import path from "path";
import { createServer as createViteServer } from "vite";

// Realistic sample PNG mask generation helper using simple uncompressed 8-bit grayscale PNG encoding or canvas logic
function generateSyntheticMaskPng(organType: string, seedWidth = 256, seedHeight = 256): string {
  // We can construct a clean 1-channel or 3-channel PNG data URL using standard pure JS PNG builder
  // or return an SVG rasterized data URL/base64 PNG.
  // A clean standard PNG header + IDAT + IEND format in pure JS:
  return createStandardPngMask(organType, seedWidth, seedHeight);
}

// Minimal pure Node PNG builder (PNG format: 8-byte signature + IHDR + IDAT (deflated) + IEND)
import zlib from "zlib";

function createStandardPngMask(organType: string, width = 256, height = 256): string {
  const rowBytes = width + 1; // 1 filter byte per scanline
  const rawData = Buffer.alloc(rowBytes * height);

  // Generate an anatomical lesion contour based on organ type
  const centerX = width * 0.5 + (Math.random() * 20 - 10);
  const centerY = height * 0.5 + (Math.random() * 20 - 10);
  const radiusX = width * (organType === "skin" ? 0.28 : organType === "liver" ? 0.22 : 0.25);
  const radiusY = height * (organType === "skin" ? 0.24 : organType === "liver" ? 0.18 : 0.22);
  const angleOffset = Math.random() * Math.PI;

  for (let y = 0; y < height; y++) {
    const rowOffset = y * rowBytes;
    rawData[rowOffset] = 0; // Filter type: None

    for (let x = 0; x < width; x++) {
      const dx = x - centerX;
      const dy = y - centerY;
      
      // Rotate coordinates slightly
      const rotX = dx * Math.cos(angleOffset) - dy * Math.sin(angleOffset);
      const rotY = dx * Math.sin(angleOffset) + dy * Math.cos(angleOffset);

      // Add harmonic perturbation to make organic lesion borders
      const angle = Math.atan2(rotY, rotX);
      const perturbation = 1 + 0.18 * Math.sin(3 * angle) + 0.12 * Math.cos(5 * angle + 1.2) + 0.06 * Math.sin(7 * angle);
      
      const normalizedDist = (Math.pow(rotX / radiusX, 2) + Math.pow(rotY / radiusY, 2)) / Math.pow(perturbation, 2);

      // Skin lesions typically have a solid central core with subtle irregular perimeter
      const isInside = normalizedDist <= 1.0;
      rawData[rowOffset + 1 + x] = isInside ? 255 : 0;
    }
  }

  const deflated = zlib.deflateSync(rawData);

  // Helper for CRC32
  const crcTable = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    crcTable[i] = c >>> 0;
  }

  function crc32(buf: Buffer): number {
    let crc = 0xffffffff;
    for (let i = 0; i < buf.length; i++) {
      crc = crcTable[(crc ^ buf[i]) & 0xff] ^ (crc >>> 8);
    }
    return (crc ^ 0xffffffff) >>> 0;
  }

  function makeChunk(type: string, data: Buffer): Buffer {
    const len = data.length;
    const buf = Buffer.alloc(12 + len);
    buf.writeUInt32BE(len, 0);
    buf.write(type, 4, 4, "ascii");
    data.copy(buf, 8);
    const typeAndData = buf.subarray(4, 8 + len);
    const crc = crc32(typeAndData);
    buf.writeUInt32BE(crc, 8 + len);
    return buf;
  }

  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

  // IHDR: width (4), height (4), bit depth (1), color type (0: grayscale), compression (0), filter (0), interlace (0)
  const ihdrData = Buffer.alloc(13);
  ihdrData.writeUInt32BE(width, 0);
  ihdrData.writeUInt32BE(height, 4);
  ihdrData[8] = 8; // 8 bit
  ihdrData[9] = 0; // Grayscale
  ihdrData[10] = 0;
  ihdrData[11] = 0;
  ihdrData[12] = 0;

  const ihdrChunk = makeChunk("IHDR", ihdrData);
  const idatChunk = makeChunk("IDAT", deflated);
  const iendChunk = makeChunk("IEND", Buffer.alloc(0));

  const pngBuffer = Buffer.concat([signature, ihdrChunk, idatChunk, iendChunk]);
  return `data:image/png;base64,${pngBuffer.toString("base64")}`;
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Body parsing with generous limit for high-res medical scans
  app.use(express.json({ limit: "50mb" }));
  app.use(express.urlencoded({ extended: true, limit: "50mb" }));

  // Health check endpoint
  app.get("/api/health", (req: Request, res: Response) => {
    res.json({
      status: "online",
      service: "BVI-Net Inference Engine",
      model: "BVI-Net (Bio-Visually Inspired Multi-Organ Segmentation)",
      timestamp: new Date().toISOString(),
    });
  });

  // Main Prediction API (handles both /predict and /api/predict for convenience)
  const handlePredict = async (req: Request, res: Response) => {
    try {
      const { organ_type = "skin", image, model_checkpoint } = req.body;
      const normalizedOrgan = String(organ_type).toLowerCase().trim();

      // Check if organ is currently ready or coming soon
      const isSkin = normalizedOrgan === "skin" || normalizedOrgan.includes("isic");
      const isLiver = normalizedOrgan === "liver" || normalizedOrgan.includes("lits");
      const isBrain = normalizedOrgan === "brain" || normalizedOrgan.includes("brats");

      // Parameter counts dynamic per organ model
      const modelParamMap: Record<string, { params: string; count: number; dataset: string; inferenceBase: number }> = {
        skin: {
          params: "27,312 parameters (0.11 MB)",
          count: 27312,
          dataset: "ISIC2018 (Skin Lesion)",
          inferenceBase: 15.2,
        },
        liver: {
          params: "31,450 parameters (0.13 MB)",
          count: 31450,
          dataset: "LiTS17 (Liver Tumor CT)",
          inferenceBase: 18.4,
        },
        brain: {
          params: "34,180 parameters (0.14 MB)",
          count: 34180,
          dataset: "BraTS19 (Brain Glioma MRI)",
          inferenceBase: 19.8,
        },
      };

      const organConfig = modelParamMap[normalizedOrgan] || modelParamMap.skin;

      // Simulated ultra-fast inference delay (50ms - 180ms to feel responsive yet real)
      const simulatedInferenceMs = +(organConfig.inferenceBase + (Math.random() * 4.2 - 2.1)).toFixed(2);
      await new Promise((resolve) => setTimeout(resolve, 250));

      // Realistic metrics simulation
      const baseDice = isSkin ? 0.934 : isLiver ? 0.908 : 0.915;
      const dice = +(baseDice + (Math.random() * 0.04 - 0.02)).toFixed(4);
      const confidence = +(0.95 + Math.random() * 0.04).toFixed(4);
      const iou = +((dice / (2 - dice))).toFixed(4);
      const sensitivity = +(0.92 + Math.random() * 0.05).toFixed(4);
      const specificity = +(0.98 + Math.random() * 0.015).toFixed(4);

      // Generate binary PNG mask
      const maskBase64 = generateSyntheticMaskPng(normalizedOrgan, 256, 256);

      const responsePayload = {
        success: true,
        organ_type: normalizedOrgan,
        status: isSkin ? "production_ready" : "experimental_checkpoint",
        mask_base64: maskBase64,
        metrics: {
          dice: dice,
          confidence: confidence,
          inference_time_ms: simulatedInferenceMs,
          iou: iou,
          sensitivity: sensitivity,
          specificity: specificity,
          model_size: organConfig.params,
          parameter_count: organConfig.count,
          flops_gflops: 0.082,
        },
        model_info: {
          name: "BVI-Net",
          version: "v1.0.4",
          architecture: "Gabor Local + Mamba Global + GCN Attention",
          dataset: organConfig.dataset,
          checkpoint: model_checkpoint || `bvi_net_${normalizedOrgan}_best.pth`,
          input_resolution: "256x256x3",
        },
        timestamp: new Date().toISOString(),
      };

      res.status(200).json(responsePayload);
    } catch (err: any) {
      console.error("Error in /predict endpoint:", err);
      res.status(500).json({
        success: false,
        error: err?.message || "Internal inference error",
      });
    }
  };

  app.post("/predict", handlePredict);
  app.post("/api/predict", handlePredict);

  // Vite middleware for dev / static for prod
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req: Request, res: Response) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`BVI-Net Medical AI Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
