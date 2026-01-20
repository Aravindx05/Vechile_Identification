import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState(null);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const selected = e.target.files[0];

    if (!selected) return;

    if (!selected.name.endsWith(".csv")) {
      setError("Only CSV files are allowed");
      setFile(null);
    } else {
      setError("");
      setFile(selected);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="bg-white w-[380px] rounded-lg shadow-md p-6">
        
        <h1 className="text-xl font-semibold text-gray-800 text-center">
          VANET Data Upload
        </h1>
        <p className="text-sm text-gray-500 text-center mt-1">
          Upload CSV dataset to continue
        </p>

        <div className="mt-5">
          <label className="flex flex-col items-center justify-center border border-dashed border-gray-300 rounded-md p-4 cursor-pointer hover:bg-gray-50 transition">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
            />
            <span className="text-sm text-gray-600">
              Click to select CSV file
            </span>
          </label>

          {file && (
            <p className="mt-3 text-sm text-green-600">
              ✔ {file.name}
            </p>
          )}

          {error && (
            <p className="mt-3 text-sm text-red-500">
              {error}
            </p>
          )}
        </div>

        <button
          disabled={!file}
          className={`mt-6 w-full py-2 text-sm rounded-md font-medium text-white
          ${file ? "bg-blue-600 hover:bg-blue-700" : "bg-gray-400 cursor-not-allowed"}`}
        >
          Continue
        </button>
      </div>
    </div>
  );
}
