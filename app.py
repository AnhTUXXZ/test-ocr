import os
import cv2
import numpy as np
import urllib.request
from flask import Flask, jsonify, request
from rapidocr_onnxruntime import RapidOCR

app = Flask(__name__)

# Khởi tạo Engine OCR một lần khi chạy server để tối ưu hiệu suất
print("Đang khởi tạo model RapidOCR...")
ocr_engine = RapidOCR()

def fast_ocr_process(img, ocr_engine):
    """Hàm xử lý OCR siêu tốc được trích xuất từ source gốc"""
    h, w = img.shape[:2]
    scale_ratio = 1.0
    
    # Resize nếu ảnh quá lớn để quét nhanh hơn
    if w > 720: 
        scale_ratio = 720 / w
        new_w = int(w * scale_ratio)
        new_h = int(h * scale_ratio)
        img = cv2.resize(img, (new_w, new_h))
        
    # Chuyển sang ảnh xám
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thực hiện OCR
    raw_results, _ = ocr_engine(gray_img)
    
    formatted_results = []
    if raw_results:
        for dt_box in raw_results:
            box = dt_box[0]
            text = dt_box[1]
            score = float(dt_box[2])
            
            # Trả tọa độ về tỷ lệ gốc
            original_box = [[int(pt[0] / scale_ratio), int(pt[1] / scale_ratio)] for pt in box]
            
            # Chuyển thành Dictionary để Flask có thể dễ dàng trả về JSON
            formatted_results.append({
                "box": original_box,
                "text": text,
                "score": round(score, 4)
            })
            
    return formatted_results

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API OCR đang hoạt động. Truy cập /test để chạy thử."})

@app.route('/test', methods=['GET'])
def test_ocr():
    """Route test quét OCR trực tiếp từ URL ảnh"""
    image_url = "https://mywebip.ddns.net/img.png"
    
    try:
        # Cấu hình header để tránh bị server nguồn chặn (403 Forbidden)
        req = urllib.request.Request(
            image_url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        
        # Tải ảnh về RAM
        with urllib.request.urlopen(req, timeout=10) as response:
            image_bytes = bytearray(response.read())
            
        image_array = np.asarray(image_bytes, dtype=np.uint8)
        
        # Decode ảnh bằng OpenCV
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"status": "error", "message": "Không thể giải mã hình ảnh từ URL."}), 400
            
        # Đưa vào hàm OCR xử lý
        results = fast_ocr_process(img, ocr_engine)
        
        return jsonify({
            "status": "success",
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "text_count": len(results),
            "data": results
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Render.com sẽ tự động cấp phát PORT thông qua biến môi trường
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)