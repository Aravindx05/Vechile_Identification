import { useLocation, useNavigate } from "react-router-dom";

export default function Preprocess() {
  const { state } = useLocation();
  const navigate = useNavigate();

  const data = state?.data || [];

  if (data.length === 0) {
    return (
      <div className="p-6">
        <h2 className="text-red-600">No dataset received</h2>
      </div>
    );
  }

  const columns = Object.keys(data[0]);
  const previewRows = data.slice(0, 10);

  return (
    <div className="p-6">
      {/* Title */}
      <h1 className="text-2xl font-semibold mb-4">
        Dataset Preprocessing
      </h1>

      {/* Summary */}
      <div className="mb-4 text-sm text-gray-700">
        <span className="mr-6">
          <b>Total Rows:</b> {data.length}
        </span>
        <span>
          <b>Total Columns:</b> {columns.length}
        </span>
      </div>

      {/* ===== TABLE (WITH CLEAR LINES) ===== */}
     <table className="data-table">
        <thead>
          <tr>
            {columns.map((col, idx) => (
              <th key={idx}>{col}</th>
            ))}
          </tr>
        </thead>

        <tbody>
          {previewRows.map((row, rowIdx) => (
            <tr key={rowIdx}>
              {columns.map((col, colIdx) => (
                <td key={colIdx}>
                  {row[col] === "" || row[col] === null ? "NA" : row[col]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-gray-500">
        Showing first {previewRows.length} rows
      </p>

      <button
        onClick={() => navigate("/results")}
        className="mt-6 px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700"
      >
        View Model Results
      </button>
    </div>
  );
}
