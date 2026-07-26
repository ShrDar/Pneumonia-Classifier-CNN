import { Download } from "lucide-react"

export default function Popup({image, showGradcam, setShowGradcam} : {image: string, showGradcam: boolean, setShowGradcam: React.Dispatch<React.SetStateAction<boolean>>}) {

    const handleDownload = () => {
        const a = document.createElement("a");

        a.href = `data:image/png;base64,${image}`;
        a.download = "gradcam.png";

        document.body.appendChild(a);
        a.click();
        a.remove();
    }

    return (
        <> 
            {
                showGradcam && 
                <>
                    <div className="z-10 fixed bg-[#0000009e] w-screen h-screen top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" onClick={() => setShowGradcam((prev) => !prev)}></div>
                    <div className="z-20 fixed flex justify-center items-center w-[80%] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                        <img
                            src={`data:image/png;base64,${image}`}
                            alt="GradCAM"
                            className="rounded-xl"
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