import os
import cv2
import numpy as np
import gc
import datetime
import json
from flask import Flask, jsonify, request, render_template_string
from rapidocr_onnxruntime import RapidOCR

app = Flask(__name__)
LOG_FILE = "server_logs.json"

# =====================================================================
# --- HÀM QUẢN LÝ DỮ LIỆU LOG (TỰ ĐỘNG XÓA KHI ĐẠT 48KB) ---
# =====================================================================
def save_server_log(img_size_kb, text_count, processing_time):
    time_str = datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')
    log_entry = {
        "time": time_str,
        "size_kb": f"{img_size_kb:.2f}",
        "process_time": f"{processing_time:.2f}",
        "text_count": text_count
    }
    
    try:
        logs = []
        # Kiểm tra dung lượng file log
        if os.path.exists(LOG_FILE):
            file_size_bytes = os.path.getsize(LOG_FILE)
            # Nếu dung lượng >= 48KB (48 * 1024 bytes) -> Xóa trắng
            if file_size_bytes >= 49152: 
                print(f"[HỆ THỐNG] File log đạt {file_size_bytes/1024:.2f}KB (>= 48KB). Tự động dọn dẹp data!")
                logs = []
            else:
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    try:
                        logs = json.load(f)
                    except:
                        logs = []
                        
        # Chèn log mới nhất lên đầu danh sách
        logs.insert(0, log_entry)
        
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"[LỖI LOG] Không thể ghi log: {e}")

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

# =====================================================================
# --- GIAO DIỆN CONSOLE WEB HIỆN ĐẠI TẠI ROUTE MẶC ĐỊNH ---
# =====================================================================
HTML_CONSOLE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Render OCR Console Server</title>
    <style>
        body {
            background-color: #0d1117;
            color: #00ff00;
            font-family: 'Courier New', Courier, monospace;
            padding: 20px;
            line-height: 1.6;
        }
        .console-box {
            background-color: #000;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.2);
            border: 1px solid #30363d;
            max-width: 900px;
            margin: 0 auto;
        }
        h2 {
            color: #58a6ff;
            text-align: center;
            border-bottom: 1px dashed #30363d;
            padding-bottom: 15px;
            margin-top: 0;
        }
        .status-bar {
            color: #8b949e;
            font-size: 0.9em;
            margin-bottom: 20px;
            text-align: right;
        }
        .log-entry {
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px dashed #21262d;
        }
        .log-entry:last-child {
            border-bottom: none;
        }
        .time { color: #ff7b72; font-weight: bold; }
        .highlight { color: #f2cc60; font-weight: bold; }
        .highlight-success { color: #3fb950; font-weight: bold; }
        .prefix { color: #79c0ff; }
    </style>
    <script>
        // Tự động tải lại trang mỗi 5 giây để xem log mới nhất như terminal thật
        setInterval(function() {
            window.location.reload();
        }, 5000);
    </script>
</head>
<body>
    <div class="console-box">
        <h2>🚀 HỆ THỐNG XỬ LÝ ẢNH AI ĐÁM MÂY (RENDER)</h2>
        <div class="status-bar">
            Trạng thái: Đang hoạt động | Auto-refresh: 5s | Dung lượng Data: {{ current_size }} KB
        </div>
        <div id="logs-container">
            {% if logs|length == 0 %}
                <p style="color: #8b949e;">[HỆ THỐNG] Chưa có dữ liệu nhận vào hoặc log vừa được dọn dẹp (Reset ở 48KB)...</p>
            {% else %}
                {% for log in logs %}
                <div class="log-entry">
                    <span class="time">[{{ log.time }}]</span>
                    <br><span class="prefix">root@render:~#</span> Đã nhận ảnh từ thiết bị với dung lượng <span class="highlight">{{ log.size_kb }} KB</span>.
                    <br><span class="prefix">root@render:~#</span> Đã xử lý ảnh qua hệ thống OCR nhận diện được <span class="highlight-success">{{ log.text_count }}</span> khối chữ (Thời gian hoàn thành: <span class="highlight">{{ log.process_time }}s</span>).
                </div>
                {% endfor %}
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    logs = []
    file_size_kb = 0.0
    if os.path.exists(LOG_FILE):
        try:
            file_size_kb = os.path.getsize(LOG_FILE) / 1024
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except:
            pass
            
    return render_template_string(HTML_CONSOLE, logs=logs, current_size=f"{file_size_kb:.2f}")

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
        # GHI LOG HIỂN THỊ RA UI TRƯỚC KHI DỌN RAM
        # ==============================================================
        save_server_log(img_size_kb, len(results), processing_time)
        
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
