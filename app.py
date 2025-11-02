import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import io
import requests
import os
import torchvision.models as models
import torch.nn.functional as F

# ---------- Helpers ----------
@st.cache_data
def fetch_imagenet_labels():
    """Download ImageNet human-readable labels (1000 classes)."""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        labels = [l.strip() for l in r.text.splitlines()]
        if len(labels) == 1000:
            return labels
        else:
            return [f"class_{i}" for i in range(1000)]
    except Exception:
        return [f"class_{i}" for i in range(1000)]

@st.cache_resource
def load_pretrained_alexnet_device(device):
    """Load torchvision pretrained AlexNet (ImageNet)."""
    model = models.alexnet(pretrained=True)
    model.eval()
    model.to(device)
    return model

@st.cache_resource
def load_checkpoint_model(checkpoint_bytes, device):
    """
    Load user-supplied checkpoint (expects a dict with 'state_dict' or direct state_dict).
    Returns a model (AlexNet architecture) with loaded weights if possible.
    """
    model = models.alexnet(pretrained=False)
    model.to(device)
    model.eval()
    try:
        # load bytes into buffer
        buf = io.BytesIO(checkpoint_bytes)
        checkpoint = torch.load(buf, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # sometimes keys prefixed with 'module.' (from DataParallel) - fix common case
        new_state = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_key] = v
        model.load_state_dict(new_state, strict=False)
        return model
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        return None

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(pil_image):
    transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transforms(pil_image).unsqueeze(0)  # add batch dim

def predict(model, input_tensor, topk=5):
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        top_probs, top_idxs = probs.topk(topk, dim=1)
    return top_probs.cpu().numpy()[0], top_idxs.cpu().numpy()[0]

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AlexNet Demo", layout="centered")

st.title("AlexNet Image Classification — Streamlit Demo")
st.markdown(
    """
Upload an image to get predictions. Optionally upload a PyTorch checkpoint to replace the pretrained weights.
- If you don't upload a checkpoint, a torchvision **pretrained AlexNet** (ImageNet) will be used.
- Model expects photos; results will show top-K predicted labels and probabilities.
"""
)

# Sidebar: Options
st.sidebar.header("Options")
topk = st.sidebar.number_input("Top K predictions", min_value=1, max_value=10, value=5)
use_gpu = st.sidebar.checkbox("Prefer GPU (if available)", value=False)
device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
st.sidebar.write(f"Using device: **{device.type}**")

# Image uploader
uploaded_image = st.file_uploader("Upload an image (png/jpg)", type=["png","jpg","jpeg"])
# Checkpoint uploader
uploaded_ckpt = st.file_uploader("Optional: upload PyTorch checkpoint (.pt/.pth/.bin)", type=["pt","pth","bin"])
# Option to use torchvision pretrained
use_torchvision_pretrained = st.sidebar.checkbox("Use torchvision pretrained AlexNet (ImageNet)", value=True)

# Load labels
labels = fetch_imagenet_labels()

# Load model (deferred until needed)
model = None
model_info = ""
if uploaded_ckpt is not None and not use_torchvision_pretrained:
    ckpt_bytes = uploaded_ckpt.read()
    model = load_checkpoint_model(ckpt_bytes, device)
    model_info = f"User-supplied checkpoint loaded ({uploaded_ckpt.name})" if model else "Failed to load checkpoint."
else:
    # Load torchvision pretrained
    model = load_pretrained_alexnet_device(device)
    model_info = "torchvision AlexNet (pretrained on ImageNet)"

st.write("**Model:**", model_info)

# When image uploaded -> show and predict
if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert("RGB")
    except Exception as e:
        st.error(f"Could not read the uploaded image: {e}")
        st.stop()

    st.image(image, caption="Uploaded image", use_column_width=True)
    st.write("Preprocessing and predicting...")

    # Preprocess
    input_tensor = preprocess_image(image)

    if model is None:
        st.error("Model is not available. Check if checkpoint failed to load.")
    else:
        try:
            probs, idxs = predict(model, input_tensor, topk=topk)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Display results
        st.subheader("Top predictions")
        for i, (p, idx) in enumerate(zip(probs, idxs), start=1):
            label = labels[idx] if idx < len(labels) else f"class_{idx}"
            st.write(f"{i}. **{label}** — {p*100:.2f}% (class id: {idx})")

        # Raw logits/probs (expandable)
        with st.expander("Show raw probabilities (first 20 classes)"):
            import numpy as np
            probs_all = F.softmax(model(input_tensor.to(device)), dim=1).cpu().numpy()[0]
            top20_idx = np.argsort(-probs_all)[:20]
            rows = [{"class_id": int(int(i)), "label": (labels[int(i)] if int(i) < len(labels) else f"class_{i}"), "prob": float(probs_all[int(i)])} for i in top20_idx]
            st.table(rows)

else:
    st.info("Upload an image to get started — or try the example images below.")
    # Provide a couple of example images via URL (not required — optional)
    if st.button("Load example image (cat)"):
        example_url = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
        try:
            r = requests.get(example_url, timeout=10)
            r.raise_for_status()
            image = Image.open(io.BytesIO(r.content)).convert("RGB")
            st.image(image, caption="Example image (cat)", use_column_width=True)
            input_tensor = preprocess_image(image)
            probs, idxs = predict(model, input_tensor, topk=topk)
            st.subheader("Top predictions")
            for i, (p, idx) in enumerate(zip(probs, idxs), start=1):
                label = labels[idx] if idx < len(labels) else f"class_{idx}"
                st.write(f"{i}. **{label}** — {p*100:.2f}% (class id: {idx})")
        except Exception as e:
            st.error(f"Could not load example image: {e}")

st.markdown("""---""")
st.caption("App created from an AlexNet notebook. You can customize this script to load different architectures or label maps.")
