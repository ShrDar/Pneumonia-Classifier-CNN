"use client";

import { useRef, useState } from "react";
import { Upload, Image as ImageIcon, ChevronDown, CircleX } from "lucide-react";
import { predictXray } from "@/services/prediction.services";

import Image from "next/image";
import Popup from "./Popup";
import { PredictionResponse } from "@/lib/type";


export default function Body() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState("");

  const [model, setModel] = useState("transfer");
  const [baselineType, setBaselineType] = useState<string>("model1");
  const [transferType, setTransferType] = useState<string>("frozen");
  const [gradCam, setGradCam] = useState<string>("")
  const [showGradcam, setShowGradcam] = useState<boolean>(false)

  const handleFile = (file: File) => {
    if (!file.type.startsWith("image/")) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setPrediction(null);
    setError("");
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();

    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handlePredict = async () => {
    if (!image) {
      alert("Please select an image");
      return;
    }

    try {
      setLoading(true);
      setPrediction(null);
      setError("");

      const selectedType = model === "baseline" ? baselineType : transferType;

      const result = await predictXray(
        image,
        model,
        selectedType
      );


      setPrediction(result);
    } catch (err: any) {
      console.error(err);

      setError(
        err?.response?.data?.detail || "Prediction failed"
      );
    } finally {
      setLoading(false);
    }
  };

  const handleGradcam = () => {
    setShowGradcam((prev) => !prev)
  }

  return (
    <main className="relative flex w-full md:h-full items-center justify-center bg-zinc-950 px-6 py-10">
      <div className="w-full flex flex-col justify-center md:h-full max-w-[80%] rounded-3xl border border-zinc-800 bg-zinc-900 p-8 shadow-2xl">
        <h2 className="mb-2 text-center text-3xl font-bold text-white">
          Pneumonia Detector
        </h2>

        <p className="mb-10 text-center text-zinc-400">
          Upload a chest X-ray image and choose the model for prediction.
        </p>

        <div className="grid gap-10 md:grid-cols-2">
          <div className="flex flex-col justify-between space-y-4">
            <div className="relative">
              <label className="mb-2 block text-sm font-medium text-zinc-300">
                Select Model
              </label>

              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full cursor-pointer appearance-none rounded-lg border border-zinc-700 bg-zinc-800 px-4 py-3 text-white outline-none transition focus:border-zinc-500"
              >
                <option value="baseline">Baseline CNN</option>
                <option value="transfer">
                  Transfer Learning - ResNet18
                </option>
              </select>

              <ChevronDown className="pointer-events-none absolute right-4 top-13 h-5 w-5 -translate-y-1/2 text-zinc-400" />
            </div>

            {model === "baseline" && (
              <div className="relative">
                <label className="mb-2 block text-sm font-medium text-zinc-300">
                  Baseline Model
                </label>

                <select
                  value={baselineType}
                  onChange={(e) => setBaselineType(e.target.value)}
                  className="w-full cursor-pointer appearance-none rounded-lg border border-zinc-700 bg-zinc-800 px-4 py-3 text-white outline-none transition focus:border-zinc-500"
                >
                  <option value="model1">Model 1</option>
                  <option value="model2">Model 2</option>
                </select>

                <ChevronDown className="pointer-events-none absolute right-4 top-13 h-5 w-5 -translate-y-1/2 text-zinc-400" />
              </div>
            )}

            {model === "transfer" && (
              <div className="relative">
                <label className="mb-2 block text-sm font-medium text-zinc-300">
                  Transfer Learning Type
                </label>

                <select
                  value={transferType}
                  onChange={(e) => setTransferType(e.target.value)}
                  className="w-full cursor-pointer appearance-none rounded-lg border border-zinc-700 bg-zinc-800 px-4 py-3 text-white outline-none transition focus:border-zinc-500"
                >
                  <option value="frozen">Frozen Model</option>
                  <option value="finetuned">Fine-tuned Model</option>
                </select>

                <ChevronDown className="pointer-events-none absolute right-4 top-13 h-5 w-5 -translate-y-1/2 text-zinc-400" />
              </div>
            )}

            <button
              onClick={handlePredict}
              disabled={loading}
              className="mt-4 cursor-pointer rounded-lg border border-zinc-700 bg-zinc-800 py-3 text-lg font-medium text-white transition-all hover:border-zinc-500 hover:bg-zinc-700 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Predicting..." : "Predict"}
            </button>

            {error && (
              <div className="rounded-lg bg-red-900/20 p-3 text-sm text-red-400">
                {error}
              </div>
            )}

            {prediction && (
              <div className="rounded-xl border border-zinc-700 bg-zinc-800 p-5 text-white">
                <div className="flex items-center justify-between ">
                  <span className="text-sm uppercase tracking-wider text-zinc-400">
                    Prediction
                  </span>

                  <span
                    className={`rounded-full px-3 py-1 text-sm font-semibold ${
                      prediction.prediction === "PNEUMONIA"
                        ? " text-red-400 "
                        : "bg-green-500/15 text-green-400 "
                    }`}
                  >
                    {prediction.prediction}
                  </span>
                </div>

                <div className="mt-4 rounded-lg bg-[#1a1a1ab4] p-3 max-h-[20vh] overflow-y-scroll lg:overflow-auto">
                  <p className="text-sm leading-6 text-">
                    <span className="font-semibold">Medical Disclaimer:</span> This
                    prediction is generated by an AI model and is intended for educational
                    and research purposes only. It is a medical
                    diagnosis and should not be used to make clinical decisions. Always
                    consult a qualified healthcare professional for proper evaluation and
                    diagnosis.
                  </p>
                </div>
              </div>
            )}
          </div>

          <div>
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="relative flex h-80 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-zinc-700 from-zinc-900 via-zinc-950 to-black transition-all hover:border-zinc-500"
            >
              {!preview && (
                <>
                  <Image
                    src="/xray.jpg"
                    alt="Chest X-ray"
                    fill
                    sizes="20vh"
                    priority
                    className="pointer-events-none select-none object-cover opacity-30"
                  />

                  <div className="absolute inset-0 bg-zinc-950/60" />

                  <div className="relative z-10 flex flex-col items-center">
                    <Upload className="mb-4 h-12 w-12 text-zinc-400" />

                    <p className="text-center text-lg font-medium text-white">
                      Drop Chest X-ray Here
                    </p>

                    <p className="mt-2 text-center text-sm text-zinc-400">
                      or click to browse from your computer
                    </p>
                  </div>
                </>
              )}

              {preview && (
                <>
                  <Image
                  src={preview}
                  alt="Preview"
                  width={500}
                  height={500}
                  unoptimized
                  className="relative z-10 w-full max-h-full max-w-full rounded-lg object-cover"
                />
                <CircleX onClick={(e) => {
                  e.stopPropagation()
                  setPreview(null)
                  setImage(null)
                  setPrediction(null)
                  }} size={20} className="absolute z-20 bg-red-400 rounded-full right-2 top-2 hover:bg-red-600" />
                </>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => {
                  if (e.target.files?.[0]) {
                    handleFile(e.target.files[0]);
                  }
                }}
              />


            </div>

            {image && (
              <div>
                <div className="mt-3 flex items-center gap-2 text-sm text-zinc-400">
                  <ImageIcon className="h-4 w-4" />
                  <span className="truncate">{image.name}</span>
                </div>
                {
                  gradCam !== "" &&
                  <div>
                    <button onClick={() => handleGradcam()} className="w-full flex justify-center items-center bg-zinc-800 text-white p-3 cursor-pointer rounded-xl text-xs hover:bg-zinc-700 border border-zinc-700 my-5">Gradcam</button>
                  </div>
                }
              </div>
            )}
          </div>
        </div>
      </div>
      <Popup image = {gradCam} showGradcam = {showGradcam} setShowGradcam = {setShowGradcam} />
    </main>
  );
}