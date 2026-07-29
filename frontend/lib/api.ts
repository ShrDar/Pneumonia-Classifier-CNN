const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
const GRADCAM_API_BASE_URL = process.env.NEXT_PUBLIC_GRADCAM_API_URL;

if(!API_BASE_URL) {
    throw new Error("NEXT_PUBLIC_API_URL is not Defined")
}

if (!GRADCAM_API_BASE_URL) {
  throw new Error("NEXT_PUBLIC_GRADCAM_API_URL is not Defined");
}

export const API = {
  BASE_URL: API_BASE_URL,
  GRADCAM_BASE_URL: GRADCAM_API_BASE_URL,

  ENDPOINTS: {
    PREDICT: `${API_BASE_URL}/predict`,
    GRADCAM: `${GRADCAM_API_BASE_URL}/gradcam`,
  },
};