import { useState, useRef } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [resultImg, setResultImg] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Chọn ảnh MRI để bắt đầu");

  // ref để mở file picker
  const fileInputRef = useRef(null);

  // mở file picker khi bấm nút Upload
  const handleOpenFile = () => {
    fileInputRef.current.click();
  };

  // chọn ảnh
  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResultImg(null);
    setStatus("📌 Ảnh đã được tải lên");
  };

  // predict
  const handlePredict = async () => {
    if (!file) {
      setStatus("⚠️ Vui lòng chọn ảnh trước");
      return;
    }

    setLoading(true);
    setStatus("⏳ Đang phân tích AI...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResultImg(`data:image/png;base64,${data.prediction_image}`);
        setStatus("✅ Dự đoán hoàn tất!");
      } else {
        setStatus("❌ Dự đoán thất bại");
      }
    } catch (err) {
      console.error(err);
      setStatus("❌ Lỗi kết nối server");
    }

    setLoading(false);
  };

  return (
    <div className="app">
      {/* hidden input */}
      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: "none" }}
      />

      {/* NOTIFICATION */}
      <div className="notification">{status}</div>

      {/* IMAGES */}
      <div className="image-row">
        <div className="image-box">
          <h3>INPUT MRI</h3>
          {preview ? (
            <img src={preview} alt="input" />
          ) : (
            <div className="placeholder">No Image</div>
          )}
        </div>

        <div className="image-box">
          <h3>PREDICTION</h3>
          {resultImg ? (
            <img src={resultImg} alt="result" />
          ) : (
            <div className="placeholder">No Result</div>
          )}
        </div>
      </div>

      {/* BUTTONS */}
      <div className="button-group">
        {/* CUSTOM UPLOAD BUTTON */}
        <button className="btn upload-btn" onClick={handleOpenFile}>
          Upload Image
        </button>

        {/* PREDICT */}
        <button
          className="btn predict-btn"
          onClick={handlePredict}
          disabled={loading}
        >
          {loading ? "Processing..." : "Run Prediction"}
        </button>
      </div>
    </div>
  );
}

export default App;
