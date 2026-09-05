from __future__ import annotations

import base64
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from nudenet import NudeDetector
from transformers import pipeline as hf_pipeline


st.set_page_config(page_title="Video NSFW detector", page_icon="🔎", layout="wide")


# ============================================================================
# داده‌ی هر فریم در طول کل پایپ‌لاین (گام‌های ۱ تا ۳)
# ============================================================================
@dataclass
class FrameRecord:
    scene_index: int
    frame_number: int
    timestamp: float
    image: np.ndarray
    nudenet_score: float = 0.0
    nudenet_labels: str = ""
    nudenet_raw: list[dict[str, Any]] | None = None  # همه‌ی تشخیص‌های خام NudeNet، برای دیباگ
    falconsai_score: float = 0.0
    is_nsfw_stage2: bool = False
    vlm_result: dict[str, Any] | None = None


# ============================================================================
# گام ۱: تشخیص صحنه با PySceneDetect + استخراج فقط فریم میانی هر صحنه
# ============================================================================
def _middle_frame_number(start_frame: int, end_frame: int) -> int:
    """
    end_frame در PySceneDetect exclusive است، پس آخرین فریم واقعی صحنه = end_frame - 1.
    برای صحنه‌های با طول >= 3 فریم، حداقل یک فریم فاصله از دو سرِ بازه هم تضمین می‌شود
    تا هرگز دقیقاً فریم اول یا آخر صحنه انتخاب نشود.
    """
    last_frame = end_frame - 1
    mid = (start_frame + last_frame) // 2
    if last_frame - start_frame >= 2:
        mid = max(start_frame + 1, min(mid, last_frame - 1))
    return mid


def extract_middle_frames(video_path: str, threshold: float, min_scene_len: int) -> tuple[list[FrameRecord], dict[str, Any]]:
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    scene_manager.detect_scenes(video=video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    fps = video.frame_rate
    if not scene_list:
        total_frames = video.duration.get_frames()
        scene_list = [(video.base_timecode, video.base_timecode + total_frames)]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("OpenCV could not open the uploaded video.")

    records: list[FrameRecord] = []
    for idx, (start_tc, end_tc) in enumerate(scene_list, start=1):
        start_frame = start_tc.get_frames()
        end_frame = end_tc.get_frames()
        mid_frame_no = _middle_frame_number(start_frame, end_frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_no)
        ok, frame = cap.read()
        if not ok:
            continue

        records.append(
            FrameRecord(
                scene_index=idx,
                frame_number=mid_frame_no,
                timestamp=(mid_frame_no / float(fps)) if fps else 0.0,
                image=frame,
            )
        )
    cap.release()
    return records, {"fps": fps, "scene_count": len(scene_list)}


# ============================================================================
# گام ۲: پیش‌فیلتر NSFW با ترکیب NudeNet + Falconsai/nsfw_image_detection
# ============================================================================
NUDENET_EXPLICIT_LABELS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}


@st.cache_resource(show_spinner="در حال بارگذاری مدل NudeNet ...")
def load_nudenet() -> NudeDetector:
    return NudeDetector()


@st.cache_resource(show_spinner="در حال بارگذاری مدل Falconsai/nsfw_image_detection ...")
def load_falconsai():
    return hf_pipeline("image-classification", model="Falconsai/nsfw_image_detection")


def run_stage2(records: list[FrameRecord], nudenet_threshold: float, falconsai_threshold: float) -> None:
    detector = load_nudenet()
    classifier = load_falconsai()

    for record in records:
        # --- اصلاح باگ رنگ NudeNet (nudenet==3.4.2) ---
        # کتابخانه در _read_image بدون قید و شرط cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)
        # را اجرا می‌کند و فرض می‌کند ورودی RGBA است، در حالی که فریم واقعی ما BGR
        # سه‌کاناله است. نتیجه این می‌شود که کانال‌های قرمز و آبی جابه‌جا می‌شوند
        # (رنگ پوست به‌اشتباه آبی/سبز دیده می‌شود) و امتیاز تشخیص برهنگی به‌شدت
        # افت می‌کند و همیشه صفر یا نزدیک صفر می‌ماند.
        # راه‌حل: قبل از detect()، فریم را از BGR به RGB تبدیل می‌کنیم تا سوآپ
        # اشتباه داخلی کتابخانه خنثی شود و تصویر با رنگ درست به مدل برسد.
        # این کار همچنین نیاز به نوشتن/پاک‌کردن فایل موقت روی دیسک را از بین می‌برد،
        # چون detect() مستقیماً numpy array هم قبول می‌کند.
        rgb_workaround = cv2.cvtColor(record.image, cv2.COLOR_BGR2RGB)
        detections = detector.detect(rgb_workaround)
        record.nudenet_raw = detections  # همه‌ی برچسب‌ها (نه فقط برچسب‌های صریح) برای دیباگ ذخیره می‌شود

        max_score = 0.0
        found_labels: list[str] = []
        for det in detections:
            label = det.get("class") or det.get("label")
            score = float(det.get("score", 0.0))
            if label in NUDENET_EXPLICIT_LABELS:
                max_score = max(max_score, score)
                if score >= nudenet_threshold:
                    found_labels.append(f"{label}:{score:.2f}")
        record.nudenet_score = max_score
        record.nudenet_labels = "|".join(found_labels)

        pil_image = Image.fromarray(cv2.cvtColor(record.image, cv2.COLOR_BGR2RGB))
        preds = classifier(pil_image)
        falconsai_score = 0.0
        for p in preds:
            if p["label"].lower() == "nsfw":
                falconsai_score = float(p["score"])
        record.falconsai_score = falconsai_score

        # ترکیب OR: کافیست یکی از دو مدل، حتی با احتمال کم، فریم را NSFW تشخیص دهد.
        record.is_nsfw_stage2 = (max_score >= nudenet_threshold) or (falconsai_score >= falconsai_threshold)


# ============================================================================
# گام ۳: ارسال فقط فریم‌های پرچم‌خورده در گام ۲ به یک مدل VLM (از طریق OpenRouter)
# ============================================================================
def jpeg_data_url(image: np.ndarray, quality: int = 82) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode frame as JPEG.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


VLM_SYSTEM_PROMPT = """You are a strict, careful content-safety classifier used in an automated video moderation pipeline.
You will be shown exactly one still frame extracted from a video. Your only job is to decide whether THIS SINGLE FRAME
depicts NSFW (Not Safe For Work) content, according to the rules below.

Mark is_nsfw = true ONLY when the frame clearly and unambiguously shows at least one of:
- Explicit nudity: visible exposed genitals, exposed female nipples/areola, or exposed anus/buttocks in a sexual context.
- Sexual activity or a sexual act, real or simulated.
- Overtly pornographic or sexualized content.

Mark is_nsfw = false for all of the following, even if a naive skin-color heuristic might flag them:
- Ordinary exposed skin: faces, arms, legs, shoulders, midriff, feet.
- Swimwear, underwear, sportswear, or fitness/athletic content without exposed genitals or nipples.
- Medical, breastfeeding, artistic, educational, or news context without explicit sexual content.
- Cartoons, paintings, or statues that are not photorealistic depictions of explicit acts.
- Any case that is blurry, low-resolution, too small, ambiguous, or where you are not confident.
  In ambiguous cases, prefer is_nsfw=false, use a lower confidence, and explain the ambiguity in "reason".

Output rules:
- Respond with ONLY one raw JSON object. No markdown code fences, no extra commentary, no text before or after it.
- The JSON object must have EXACTLY these keys:
  {
    "is_nsfw": <boolean>,
    "confidence": <number between 0 and 1, your confidence in the is_nsfw decision>,
    "categories": <array of short strings naming what was detected, e.g. ["explicit_nudity"]; empty array if none>,
    "reason": <short string, at most ~25 words, explaining the decision>
  }
"""


def call_openrouter(record: FrameRecord, api_key: str, model: str, timeout: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 200,
        "messages": [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify this video frame for NSFW content and return the JSON object."},
                    {"type": "image_url", "image_url": {"url": jpeg_data_url(record.image)}},
                ],
            },
        ],
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    content = content.replace("```json", "").replace("```", "").strip()
    result = json.loads(content)
    return {
        "is_nsfw": bool(result.get("is_nsfw", False)),
        "confidence": float(result.get("confidence", 0)),
        "categories": result.get("categories", []),
        "reason": str(result.get("reason", "")),
    }


# ============================================================================
# کمکی: ساخت contact sheet برای نمایش شبکه‌ای فریم‌ها
# ============================================================================
def make_contact_sheet(records: list[FrameRecord], columns: int = 4, label_fn=None) -> np.ndarray:
    if not records:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    thumbs = []
    for record in records:
        thumb = cv2.resize(record.image, (240, 150))
        label = label_fn(record) if label_fn else f"scene {record.scene_index} | {record.timestamp:.1f}s"
        cv2.rectangle(thumb, (0, 126), (240, 150), (0, 0, 0), -1)
        cv2.putText(thumb, label, (5, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    rows = []
    for start in range(0, len(thumbs), columns):
        row = thumbs[start:start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


# ============================================================================
# رابط کاربری Streamlit
# ============================================================================
def main() -> None:
    st.title("Video NSFW detector")
    st.caption(
        "گام ۱: فریم میانی هر صحنه (PySceneDetect) → "
        "گام ۲: پیش‌فیلتر NudeNet + Falconsai → "
        "گام ۳: فقط فریم‌های مشکوک به یک VLM (OpenRouter) فرستاده می‌شوند"
    )

    with st.sidebar:
        st.header("گام ۱ - تشخیص صحنه")
        scene_threshold = st.slider("آستانه حساسیت تغییر صحنه (ContentDetector)", 5.0, 60.0, 22.0, 1.0)
        min_scene_len = st.number_input("حداقل طول صحنه (فریم)", 1, 300, 10, 1)

        st.header("گام ۲ - NudeNet + Falconsai")
        nudenet_threshold = st.slider("آستانه NudeNet", 0.0, 1.0, 0.25, 0.01)
        falconsai_threshold = st.slider("آستانه Falconsai", 0.0, 1.0, 0.20, 0.01)

        st.header("گام ۳ - VLM (OpenRouter)")
        api_key = st.text_input("OpenRouter API key", value=os.getenv("OPENROUTER_API_KEY", ""), type="password")
        model = st.text_input("Vision model", value=os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001"))
        timeout = st.number_input("Timeout هر درخواست (ثانیه)", 10, 300, 90, 10)

    uploaded = st.file_uploader("یک ویدئو آپلود کنید", type=["mp4", "mov", "avi", "mkv", "webm"])
    if not uploaded:
        st.stop()

    if st.button("اجرای گام ۱ و ۲ (استخراج صحنه + پیش‌فیلتر NudeNet/Falconsai)", type="primary"):
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            temp_path = tmp.name
        try:
            with st.spinner("گام ۱: تشخیص صحنه و استخراج فریم میانی ..."):
                records, meta = extract_middle_frames(temp_path, float(scene_threshold), int(min_scene_len))
            if not records:
                st.error("هیچ فریمی از ویدئو استخراج نشد.")
                return
            with st.spinner(f"گام ۲: بررسی {len(records)} فریم با NudeNet و Falconsai ..."):
                run_stage2(records, float(nudenet_threshold), float(falconsai_threshold))
            st.session_state["records"] = records
            st.session_state["meta"] = meta
            st.session_state.pop("vlm_done", None)
        except Exception as exc:
            st.error(f"اجرای گام ۱/۲ با خطا مواجه شد: {exc}")
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    if "records" not in st.session_state:
        st.stop()

    records: list[FrameRecord] = st.session_state["records"]
    meta = st.session_state["meta"]
    flagged = [r for r in records if r.is_nsfw_stage2]

    st.write(f"**تعداد صحنه‌ها:** {meta['scene_count']} · **فریم میانی استخراج‌شده:** {len(records)} · **پرچم‌خورده در گام ۲:** {len(flagged)}")

    def stage2_label(r: FrameRecord) -> str:
        return f"scene {r.scene_index} | nd={r.nudenet_score:.2f} fc={r.falconsai_score:.2f}"

    st.subheader("همه‌ی فریم‌های میانی (گام ۱)")
    st.image(cv2.cvtColor(make_contact_sheet(records, label_fn=stage2_label), cv2.COLOR_BGR2RGB))

    with st.expander("🔍 دیباگ NudeNet: نمایش همه‌ی برچسب‌های خام تشخیص‌داده‌شده (حتی غیرصریح)"):
        st.caption(
            "اگر این جدول کاملاً خالی است، یعنی مدل روی این فریم‌ها هیچ چیز تشخیص نمی‌دهد "
            "(ممکن است هنوز مشکلی در پردازش تصویر باشد). اگر برچسب‌هایی مثل FACE_FEMALE یا "
            "*_COVERED می‌بینید ولی هیچ‌کدام *_EXPOSED نیست، یعنی مدل درست کار می‌کند و صرفاً "
            "عضو نمایانی در این فریم‌های خاص وجود ندارد."
        )
        debug_rows = []
        for r in records:
            for det in (r.nudenet_raw or []):
                debug_rows.append(
                    {
                        "scene": r.scene_index,
                        "label": det.get("class") or det.get("label"),
                        "score": round(float(det.get("score", 0.0)), 3),
                    }
                )
        if debug_rows:
            st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("هیچ تشخیصی (حتی غیرصریح) توسط NudeNet روی هیچ‌کدام از فریم‌ها ثبت نشده است.")

    if not flagged:
        st.success("هیچ فریمی توسط گام ۲ (NudeNet/Falconsai) به‌عنوان مشکوک علامت‌گذاری نشد؛ نیازی به ارسال به VLM نیست.")
        st.stop()

    st.subheader("فریم‌های پرچم‌خورده در گام ۲ (کاندیدهای ارسال به VLM)")
    st.image(cv2.cvtColor(make_contact_sheet(flagged, label_fn=stage2_label), cv2.COLOR_BGR2RGB))

    if st.button(f"گام ۳: ارسال {len(flagged)} فریم مشکوک به VLM"):
        if not api_key:
            st.error("برای گام ۳ باید OpenRouter API key وارد کنید.")
            st.stop()
        progress = st.progress(0)
        for i, record in enumerate(flagged):
            try:
                record.vlm_result = call_openrouter(record, api_key, model, timeout=int(timeout))
            except Exception as exc:
                record.vlm_result = {"is_nsfw": False, "confidence": 0.0, "categories": [], "reason": f"API error: {exc}"}
            progress.progress((i + 1) / len(flagged))
        st.session_state["records"] = records
        st.session_state["vlm_done"] = True

    if st.session_state.get("vlm_done"):
        results = [r for r in flagged if r.vlm_result is not None]
        rows = [
            {
                "scene": r.scene_index,
                "time (s)": round(r.timestamp, 2),
                "nudenet": round(r.nudenet_score, 3),
                "falconsai": round(r.falconsai_score, 3),
                **r.vlm_result,
            }
            for r in results
        ]
        st.subheader("نتایج نهایی گام ۳ (VLM)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        confirmed = [r for r in results if r.vlm_result.get("is_nsfw")]
        st.metric("فریم‌های نهایتاً تأیید NSFW توسط VLM", len(confirmed))
        if confirmed:
            st.image(
                cv2.cvtColor(
                    make_contact_sheet(
                        confirmed,
                        label_fn=lambda r: f"scene {r.scene_index} | conf={r.vlm_result['confidence']:.2f}",
                    ),
                    cv2.COLOR_BGR2RGB,
                ),
                caption="فریم‌های نهایتاً NSFW تأیید‌شده توسط VLM",
            )


if __name__ == "__main__":
    main()
