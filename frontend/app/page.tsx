import Image from "next/image";
import xray from "../public/xray.jpg"
import NavBar from "@/components/Navbar";
import Body from "@/components/Body";

export default function Home() {
  return (
    <div className="relative w-full h-screen min-h-screen">
      <Body />
    </div>
  );
}
