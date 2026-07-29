import { API } from "@/lib/api";
import {
  PredictionResponse,
  GradcamResponse,
} from "@/lib/type";

export async function predictXray(
  file: File,
  model: string,
  type: string
): Promise<PredictionResponse> {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("model", model);
  formData.append("type", type);

  const response = await fetch(API.ENDPOINTS.PREDICT, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Prediction failed");
  }

  return response.json();
}

export async function generateGradcam(
  file: File,
  model: string,
  type: string
): Promise<GradcamResponse> {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("model", model);
  formData.append("type", type);

  const response = await fetch(API.ENDPOINTS.GRADCAM, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "GradCAM generation failed");
  }

  return response.json();
}