export default function Results() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-semibold mb-4">
        Model Performance Metrics
      </h1>

      <div className="overflow-x-auto">
        <table className="border-collapse border border-black text-sm">
          <thead className="bg-gray-200">
            <tr>
              <th className="border border-black px-4 py-2">Model</th>
              <th className="border border-black px-4 py-2">MSE</th>
              <th className="border border-black px-4 py-2">MAE</th>
              <th className="border border-black px-4 py-2">R²</th>
            </tr>
          </thead>

          <tbody>
            <tr>
              <td className="border border-black px-4 py-2">SGD</td>
              <td className="border border-black px-4 py-2">7.436253e+00</td>
              <td className="border border-black px-4 py-2">2.310080e+00</td>
              <td className="border border-black px-4 py-2">0.016422</td>
            </tr>

            <tr>
              <td className="border border-black px-4 py-2">KNN</td>
              <td className="border border-black px-4 py-2">1.681601e+00</td>
              <td className="border border-black px-4 py-2">9.279386e-01</td>
              <td className="border border-black px-4 py-2">0.777578</td>
            </tr>

            <tr>
              <td className="border border-black px-4 py-2">Gaussian Process</td>
              <td className="border border-black px-4 py-2">2.089426e-20</td>
              <td className="border border-black px-4 py-2">1.181414e-10</td>
              <td className="border border-black px-4 py-2">1.000000</td>
            </tr>

            <tr>
              <td className="border border-black px-4 py-2">MLP</td>
              <td className="border border-black px-4 py-2">1.937125e-01</td>
              <td className="border border-black px-4 py-2">3.305890e-01</td>
              <td className="border border-black px-4 py-2">0.974378</td>
            </tr>

            <tr>
              <td className="border border-black px-4 py-2">Random Forest</td>
              <td className="border border-black px-4 py-2">1.071665e-02</td>
              <td className="border border-black px-4 py-2">6.849464e-02</td>
              <td className="border border-black px-4 py-2">0.998583</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
