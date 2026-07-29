export type PredictionResponse = {
  prediction: string;
  confidence: number;
  probability: number;
  // gradcam: string;
};

export type GradcamResponse = {
  gradcam: string
}