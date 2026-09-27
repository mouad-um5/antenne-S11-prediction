import { ANTENNA_CATALOG_ENDPOINT } from "../data/antennaCatalog";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const REQUEST_TIMEOUT_MS = 15000;

async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const requestId = globalThis.crypto?.randomUUID?.()
    || `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      signal: options.signal || controller.signal,
      headers: {
        Accept: "application/json",
        "X-Request-ID": requestId,
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
    });
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("Le serveur ne répond pas dans le délai attendu.");
    }
    throw new Error("Impossible de joindre le service de prédiction.");
  } finally {
    window.clearTimeout(timeout);
  }

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : { detail: await response.text() };

  if (!response.ok) {
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map((item) => item.msg).join(" · ")
      : payload.detail;
    throw new Error(detail || `Erreur API (${response.status})`);
  }

  return payload;
}

export function getModelInfo() {
  return request("/api/model-info");
}

export function getAntennaCatalog() {
  return request(ANTENNA_CATALOG_ENDPOINT);
}

export function predictPoint(payload) {
  return request("/api/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function predictSweep(payload) {
  return request("/api/predict-sweep", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
