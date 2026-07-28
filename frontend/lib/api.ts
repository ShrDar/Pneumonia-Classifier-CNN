const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

if(!API_BASE_URL) {
    throw new Error("NEXT_PUBLIC_API_URL is not Define")
}

export const API = {
    BASE_URL: API_BASE_URL,

    ENDPOINTS: {
        PREDICT: `${API_BASE_URL}/predict`
    }
}