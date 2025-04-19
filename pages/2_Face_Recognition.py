import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import cv2

st.set_page_config(page_title="Nhận dạng đối tượng - YOLOv8", layout="centered")
st.title("🎯 Nhận dạng đối tượng bằng YOLOv8")

# Chọn mô hình YOLO
model_type = st.selectbox("Chọn mô hình YOLO", ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'])
model = YOLO(model_type)

# Chọn nguồn
source_type = st.radio("Nguồn dữ liệu", ['Webcam', 'Tải ảnh'])

# ======================= XỬ LÝ ẢNH ======================
if source_type == 'Tải ảnh':
    uploaded_file = st.file_uploader("📁 Tải ảnh lên", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        results = model(image, verbose=False)[0]
        annotated_image = results.plot()

        st.image(annotated_image, channels="BGR", caption="🖼️ Kết quả nhận dạng")

# ===================== XỬ LÝ WEBCAM =====================
else:
    class YOLOTransformer(VideoTransformerBase):
        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            results = model(img, verbose=False)[0]
            annotated = results.plot()

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(key="yolo", video_transformer_factory=YOLOTransformer, media_stream_constraints={"video": True, "audio": False})
