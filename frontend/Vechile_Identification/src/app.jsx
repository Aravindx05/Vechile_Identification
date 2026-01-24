import { Routes, Route } from "react-router-dom";
import Home from "./components/Home";
import Preprocess from "./components/Preprocess";
import Results from "./components/Results";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/preprocess" element={<Preprocess />} />
      <Route path="/results" element={<Results />} />
    </Routes>
  );
}

export default App;
