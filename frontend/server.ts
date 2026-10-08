import express, { Request, Response } from "express";
import path from "path";
import { createServer as createViteServer } from "vite";

// Real BVI-Net inference runs in the Python FastAPI backend (backend/app.py,
// `uvicorn backend.app:app --port 8000`). This server just proxies to it.
const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

async function proxyJson(req: Request, res: Response, backendPath: string) {
  try {
    const upstream = await fetch(`${BACKEND_URL}${backendPath}`, {
      method: req.method,
      headers: { "Content-Type": "application/json" },
      body: req.method === "GET" ? undefined : JSON.stringify(req.body),
    });
    const body = await upstream.text();
    res.status(upstream.status).type("application/json").send(body);
  } catch (err: any) {
    console.error(`Backend unreachable at ${BACKEND_URL}:`, err?.message);
    res.status(503).json({
      success: false,
      detail: `Inference backend not reachable at ${BACKEND_URL}. Start it with: uvicorn backend.app:app --port 8000`,
    });
  }
}

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  // Body parsing with generous limit for high-res medical scans
  app.use(express.json({ limit: "50mb" }));
  app.use(express.urlencoded({ extended: true, limit: "50mb" }));

  app.get("/api/health", (req, res) => proxyJson(req, res, "/api/health"));
  app.post("/predict", (req, res) => proxyJson(req, res, "/predict"));
  app.post("/api/predict", (req, res) => proxyJson(req, res, "/predict"));

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
    console.log(`BVI-Net frontend on http://localhost:${PORT} -> backend ${BACKEND_URL}`);
  });
}

startServer();
