import Image from "next/image";

import xray from "../public/xray.jpg"

export default function NavBar() {
  return (
    <nav className="sticky top-0 z-50 border-b border-zinc-800 bg-zinc-800 backdrop-blur-lg">
      <div className="flex h-16 max-w-7xl items-center px-6">
        <div className="flex items-center gap-3">
          <div className="">
            <Image className="w-[8vh] md:w-[7vh] rounded-md" src={xray} width={500} height={500} alt="xray logo" />
          </div>

          <div>
            <h1 className="text-lg font-semibold tracking-wide text-white">
              Penumonia Detection
            </h1>
            <p className="text-xs text-zinc-400">
              CNN Penumonia Detection
            </p>
          </div>
        </div>
      </div>
    </nav>
  );
};
