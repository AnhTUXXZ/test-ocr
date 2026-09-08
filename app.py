import os
import cv2
import numpy as np
import gc
import datetime
from flask import Flask, jsonify, request
from rapidocr_onnxruntime import RapidOCR

app = Flask(__name__)

# =====================================================================
# --- HÀM HỖ TRỢ ĐỌC RAM TRỰC TIẾP TỪ HỆ THỐNG LINUX (RENDER) ---
# =====================================================================
def print_ram_usage(step_name):
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if 'VmRSS' in line:
                    ram_mb = int(line.split()[1]) / 1024
                    print(f"[{step_name}] RAM đang dùng: {ram_mb:.2f} MB")
                    return
    except:
        print(f"[{step_name}] Không thể đọc dung lượng RAM.")

# Khởi tạo Engine OCR một lần khi chạy server để tối ưu hiệu suất
print("Đang khởi tạo model RapidOCR...")
ocr_engine = RapidOCR()
print_ram_usage("SAU KHI KHỞI TẠO MODEL")

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
            
    # XÓA BIẾN CỤC BỘ BÊN TRONG HÀM QUÉT ĐỂ ÉP GIẢI PHÓNG RAM
    del gray_img
    del raw_results
            
    return formatted_results

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API OCR đang hoạt động. Hãy cấu hình bot gửi POST request chứa ảnh tới endpoint /ocr_upload."})

@app.route('/ocr_upload', methods=['POST'])
def ocr_upload():
    """Route nhận ảnh trực tiếp từ Tool Tiktok Lite gửi lên"""
    time_received = datetime.datetime.now()
    
    print("\n" + "="*60)
    print(f">>> [YÊU CẦU MỚI] Đã nhận ảnh vào lúc: {time_received.strftime('%H:%M:%S - %d/%m/%Y')}")
    
    # DỌN RAM NGAY TRƯỚC KHI XỬ LÝ ẢNH MỚI
    gc.collect() 
    print_ram_usage("TRƯỚC KHI XỬ LÝ")
    
    try:
        # Đọc dữ liệu ảnh nhị phân từ request
        image_bytes = request.data
        img_size_kb = len(image_bytes) / 1024
        print(f"[THÔNG TIN] Kích thước ảnh tải lên: {img_size_kb:.2f} KB")
        
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        
        # Decode ảnh bằng OpenCV
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is None:
            print("[LỖI] Giải mã ảnh thất bại.")
            return jsonify({"status": "error", "message": "Không thể giải mã hình ảnh từ byte array."}), 400
            
        # Đưa vào hàm OCR xử lý
        print(f"[XỬ LÝ] Bắt đầu đẩy ảnh ({img.shape[1]}x{img.shape[0]}) qua model AI...")
        results = fast_ocr_process(img, ocr_engine)
        
        time_finished = datetime.datetime.now()
        processing_time = (time_finished - time_received).total_seconds()
        
        print(f"[XỬ LÝ] Quét được lúc: {time_finished.strftime('%H:%M:%S')}")
        print(f"[KẾT QUẢ] Tốn {processing_time:.2f} giây. Tìm thấy {len(results)} khối chữ hợp lệ.")
        
        response_data = {
            "status": "success",
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "text_count": len(results),
            "data": results,
            "processing_time_seconds": processing_time
        }
        
        # ==============================================================
        # ÉP XÓA TOÀN BỘ DATA ẢNH & KẾT QUẢ RỒI DỌN RÁC NGAY LẬP TỨC
        # ==============================================================
        del image_bytes
        del image_array
        del img
        del results
        gc.collect() 
        
        print_ram_usage("SAU KHI XỬ LÝ & DỌN DẸP")
        print("="*60 + "\n")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[LỖI NGHIÊM TRỌNG] {str(e)}")
        # Cứu hộ RAM nếu xảy ra lỗi giữa chừng
        gc.collect()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Render.com sẽ tự động cấp phát PORT thông qua biến môi trường
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
