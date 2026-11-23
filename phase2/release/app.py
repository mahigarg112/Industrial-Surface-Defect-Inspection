
"""Minimal academic demonstration using only the verified K checkpoint."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from phase2.release.inference import (Inspector, PACKAGE, DISCLAIMER, decode_upload,
                                      render, inspection_report, report_bytes)

st.set_page_config(page_title="Steel surface inspection", layout="wide")


@st.cache_resource(show_spinner=False)
def get_inspector():
    return Inspector("auto", PACKAGE)


st.title("Steel surface inspection")
st.caption("Student research prototype - Four-class Severstal segmentation")
uploaded = st.file_uploader("Upload a steel image", type=["jpg", "jpeg", "png"], key="upload",
                            help="JPG/JPEG/PNG, up to 20 MiB and 4 million pixels.")
samples = sorted((Path(__file__).parent / "demo_inputs").glob("*.jpg"))
sample_name = None
if samples:
    with st.expander("Use an existing demo sample", expanded=False):
        sample_name = st.selectbox("Demo sample", [p.name for p in samples], index=None,
                                   placeholder="Choose a sample", key="sample")
        st.caption("These samples were used for verification; they are not unseen accuracy tests.")

try:
    with st.spinner("Loading the local model..."):
        inspector = get_inspector()
except Exception as exc:
    st.error(f"Model could not be loaded: {exc}")
    st.stop()

thresholds = inspector.locked_thresholds
with st.expander("Advanced: Explore Prediction Thresholds", expanded=False):
    exploratory = st.toggle("Adjust thresholds", value=False, key="adjust")
    st.caption("Exploratory thresholds have not been independently validated. Changes reuse cached probabilities; they do not retrain the model.")
    if exploratory:
        thresholds = tuple(st.slider(f"Class {c+1} threshold", 0.0, 1.0, round(value, 3), 0.005,
                                     key=f"threshold_{c+1}") for c, value in enumerate(thresholds))
    else:
        st.caption("The original locked thresholds are active.")

st.info("**Research Prototype \u2014 Not Validated for Industrial Use**\n\n" + DISCLAIMER)
with st.expander("About / Limitations", expanded=False):
    st.write("Trained on the Severstal steel dataset with four anonymized defect classes (Class 1-4). Not validated on other manufacturing domains.")
    st.write("Class 2 remains weak. False-positive predictions and missed defects are known limitations. Sigmoid outputs are not proven calibrated confidence.")
    st.write("The original K model remains selected. This prototype is not suitable for unsupervised industrial decisions. Accuracy for an uploaded image is unknown without ground truth.")

source_input = None
if uploaded is not None:
    source_input = (uploaded.getvalue(), uploaded.name)
elif sample_name is not None:
    path = Path(__file__).parent / "demo_inputs" / sample_name
    source_input = (path.read_bytes(), path.name)
else:
    source_input = st.session_state.get("sample_input")
if source_input is None:
    st.caption("Upload an image or choose a demo sample, then click Inspect Image.")
    st.stop()

try:
    decode_start = time.perf_counter()
    image, source = decode_upload(*source_input)
    decode_seconds = time.perf_counter() - decode_start
except ValueError as exc:
    st.session_state.pop("inspection", None)
    st.error(str(exc))
    st.stop()

if st.session_state.get("input_key") != source["sha256"]:
    st.session_state["input_key"] = source["sha256"]
    st.session_state.pop("inspection", None)
if (source["height"], source["width"]) != (256, 1600):
    st.caption("This input differs from the study's 1600 x 256 images. Masks retain the displayed orientation and dimensions; accuracy for this geometry is unverified.")

if st.button("Inspect Image", type="primary", key="inspect"):
    st.session_state.pop("inspection", None)
    try:
        with st.spinner("Inspecting image..."):
            probability, timing = inspector.probabilities(image)
            timing["decode_seconds"] = decode_seconds
            probability.setflags(write=False)
            st.session_state["inspection"] = {"probability": probability, "timing": timing}
            st.session_state["network_runs"] = st.session_state.get("network_runs", 0) + 1
    except Exception as exc:
        st.error(f"Inference failed: {exc}. Please try another valid image or restart the app.")

original_column, result_column = st.columns(2)
with original_column:
    st.subheader("Original image")
    st.image(image, width="stretch")
    st.caption(f"{source['width']} x {source['height']} pixels")

cached = st.session_state.get("inspection")
if cached is None:
    with result_column:
        st.subheader("Prediction overlay")
        st.caption("Click Inspect Image to generate predictions.")
else:
    try:
        render_start = time.perf_counter()
        view = render(image, cached["probability"], thresholds)
        timing = dict(cached["timing"])
        timing["view_update_seconds"] = time.perf_counter() - render_start
        timing["processing_seconds"] = timing["decode_seconds"] + timing["inference_seconds"] + timing["view_update_seconds"]
        timing["scope"] = "decode + inference/reconstruction + current rendering; excludes cold loading and browser transfer"
        report = inspection_report(source, timing, view, thresholds, inspector.locked_thresholds)
        with result_column:
            st.subheader("Prediction overlay")
            st.image(view["overlay"], width="stretch")
            st.caption("Class 1: red | Class 2: green | Class 3: blue | Class 4: yellow")
        if any(view["positive_pixels"]):
            st.write("Predicted defect regions detected. These predictions do not establish whether the product is acceptable.")
        else:
            st.write("No defect predicted at these thresholds. This does not guarantee that the image is defect-free.")
        st.caption(f"Processing time: {timing['processing_seconds']:.3f} seconds")
        if timing.get("fallback_reason"):
            st.caption(timing["fallback_reason"])
        st.subheader("Individual defect masks")
        st.caption("Affected area is the predicted positive-pixel percentage, not accuracy. Independent class masks can overlap; percentages need not sum to 100%.")
        for c, column in enumerate(st.columns(4)):
            with column:
                st.metric(f"Class {c+1} affected area", f"{view['area_percent'][c]:.3f}%")
                st.image(view["masks"][c].astype("uint8") * 255, width="stretch")
        a, b = st.columns(2)
        a.download_button("Download overlay PNG", data=view["overlay_png"], file_name="steel_overlay.png",
                          mime="image/png", key="download_overlay", on_click="ignore")
        b.download_button("Download inspection JSON", data=report_bytes(report), file_name="steel_inspection.json",
                          mime="application/json", key="download_json", on_click="ignore")
        with st.expander("Download individual masks", expanded=False):
            for c, column in enumerate(st.columns(4)):
                column.download_button(f"Download Class {c+1} mask PNG", data=view["mask_pngs"][c],
                                       file_name=f"steel_class_{c+1}.png", mime="image/png", key=f"download_mask_{c+1}", on_click="ignore")
    except Exception as exc:
        st.error(f"The result could not be rendered: {exc}")
