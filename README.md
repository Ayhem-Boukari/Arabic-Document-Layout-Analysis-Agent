# 📄 Arabic Document Layout Analysis API

<div align="center">

**FastAPI Service for Advanced Document Structure Detection using YOLOv8**

![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-00FFFF?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv)

*High-performance document layout analysis for newspapers, forms, and converted PDFs*

</div>

## 📋 Overview

This **FastAPI service** provides robust endpoints for document layout analysis using a fine-tuned **YOLOv8 model**. The API processes document images (newspapers, forms, converted PDFs) and returns precise detection of layout elements including titles, paragraphs, tables, images, and other structural components.

## 🔌 API Endpoints

### `GET /health`
**Service Status Check**
- Returns API status and configuration details
- Includes model information and loaded classes
- Quick verification of service availability

### `POST /infer`
**JSON Detection Results**
- **Input**: `multipart/form-data` with image file
- **Parameters**:
  - `imgsz` (int, default: 1280) - Inference image size
  - `iou` (float, default: 0.5) - NMS IoU threshold
  - `conf_min` (float, default: 0.001) - Minimum confidence threshold
- **Output**: Structured JSON with bounding boxes, classes, and confidence scores

### `POST /infer_image`
**Annotated Image Output**
- Same input as `/infer`
- Returns PNG image with visualized detections
- Perfect for visual verification and debugging

### `POST /infer_yolo_txt`
**YOLO Format Export**
- Same input as `/infer`
- Returns YOLO-formatted text file with normalized coordinates
- Format: `cls cx cy w h` (coordinates ∈ [0,1])

## 🏗 Project Structure

```
document-layout-api/
├── 📁 app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and endpoints
│   ├── utils.py             # YAML class loading and configuration
│   ├── postprocess.py       # Class-specific filtering and layout rules
│   └── draw.py              # Bounding box visualization utilities
├── 📁 weights/
│   └── best.pt              # YOLOv8 trained weights
├── 📁 data/
│   └── data.yaml            # YOLO class definitions
├── 📁 config/
│   └── thresholds.json      # Class-specific confidence thresholds
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.11** or higher
- Virtual environment recommended

### Step-by-Step Installation

1. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure Essential Files**
   - Place trained model: `weights/best.pt`
   - Configure classes: `data/data.yaml`
   - Set thresholds: `config/thresholds.json`

## 🚀 Quick Start

### Launch Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access Points
- **Interactive Documentation**: http://127.0.0.1:8000/docs
- **Alternative Documentation**: http://127.0.0.1:8000/redoc
- **Health Check**: http://127.0.0.1:8000/health

## ⚡ API Testing

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### JSON Detection
```bash
curl -X POST "http://127.0.0.1:8000/infer" \
  -F "file=@document.jpg" \
  -F "imgsz=1280" -F "iou=0.5" -F "conf_min=0.001"
```

### Annotated Image
```bash
curl -X POST "http://127.0.0.1:8000/infer_image" \
  -F "file=@document.jpg" --output annotated.png
```

## 📊 Response Format

### Example JSON Response
```json
{
  "width": 1654,
  "height": 2338,
  "detections": [
    {
      "cls_name": "Title",
      "cls_id": 1,
      "conf": 0.73,
      "x1": 120, "y1": 65, "x2": 1540, "y2": 210,
      "cx": 0.50, "cy": 0.06, "w": 0.86, "h": 0.06
    }
  ]
}
```

## ⚙️ Configuration

### Class Definitions (`data/data.yaml`)
```yaml
names:
  - Header
  - Title
  - Text
  - Table
  - Image
  - Footer
  - Stamp or Signature
  - Caption
  - Keyvalue
  - List-item
  - Check-box
```

### Confidence Thresholds (`config/thresholds.json`)
```json
{
  "per_class_conf": {
    "Header": 0.25,
    "Title": 0.30,
    "Text": 0.35,
    "Table": 0.30,
    "Image": 0.30,
    "Footer": 0.25,
    "Stamp or Signature": 0.30,
    "Caption": 0.25,
    "Keyvalue": 0.25,
    "List-item": 0.25,
    "Check-box": 0.15
  }
}
```

## 📦 Dependencies

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
ultralytics==8.3.175
opencv-python-headless==4.10.0.84
pydantic==2.8.2
python-multipart==0.0.9
PyYAML==6.0.2
numpy==1.26.4
```

## 🐳 Docker Deployment

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ca-certificates

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY weights ./weights
COPY data ./data
COPY config ./config

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and Run:**
```bash
docker build -t document-layout-api .
docker run -p 8000:8000 document-layout-api
```

## 🔧 Troubleshooting

### Common Issues
- **Missing weights**: Ensure `weights/best.pt` exists
- **Unknown classes**: Verify class names in `data/data.yaml`
- **Slow performance**: Reduce `imgsz` parameter (e.g., 960)
- **OpenCV errors**: Use `opencv-python-headless` version

---

<div align="center">

**Arabic Document Layout Analysis API** - *Precise document structure detection powered by YOLOv8*

</div>
