import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Papa from "papaparse";

export default function Home() {
  const [file, setFile] = useState(null);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleContinue = () => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        navigate("/preprocess", {
          state: { data: results.data }
        });
      }
    });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="bg-white w-[380px] p-6 rounded-lg shadow">
        <h1 className="text-xl font-semibold text-center">
          Upload Dataset
        </h1>

        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="mt-4"
        />

        <button
          onClick={handleContinue}
          disabled={!file}
          className="mt-6 w-full bg-blue-600 text-white py-2 rounded disabled:bg-gray-400"
        >
          Preprocess
        </button>
      </div>
    </div>
  );
}
