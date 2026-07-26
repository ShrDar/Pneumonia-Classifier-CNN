import { Download } from "lucide-react"

export default function Popup({imageUrl, showGradcam, setShowGradcam} : {imageUrl: string, showGradcam: boolean, setShowGradcam: React.Dispatch<React.SetStateAction<boolean>>}) {

    const handleDownload = async () => {
    try {
        const downloadUrl = `${process.env.NEXT_PUBLIC_API_URL}${imageUrl}`;

        const response = await fetch(downloadUrl);

        if (!response.ok) {
        throw new Error("Failed to download image");
        }

        const blob = await response.blob();

        const url = window.URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "gradcam.png";

        document.body.appendChild(a);
        a.click();

        a.remove();
        window.URL.revokeObjectURL(url);
    } catch (err) {
        console.error(err);
        alert("Failed to download Grad-CAM image.");
    }
    };

    return (
        <> 
            {
                showGradcam && 
                <>
                    <div className="z-10 fixed bg-[#0000009e] w-screen h-screen top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" onClick={() => setShowGradcam((prev) => !prev)}></div>
                    <div className="z-20 fixed flex justify-center items-center w-[80%] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                        <img
                            src={`${process.env.NEXT_PUBLIC_API_URL}${imageUrl}`}
                            alt="GradCAM"
                            className="object-contains rounded-xl"
                        />
                        <div onClick={() => { handleDownload() }} className="downloadBtn flex justify-center items-center text-white fixed right-3 top-1 bg-[#363636] p-1 lg:p-2 cursor-pointer rounded-full">
                            <Download size={18} />
                        </div>
                    </div>
                </>
            }
        </>
    )
}