
import gradio as gr
import tensorflow as tf
import numpy as np
import json

from pathlib import Path
from PIL import Image, ImageOps
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load configuration and trained model
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

with open(OUTPUT_DIR / "app_config.json") as f:
    config = json.load(f)

classes = config["classes"]
image_size = tuple(config["image_size"])

model = tf.keras.models.load_model(
    OUTPUT_DIR / config["model"],
    compile=False
)

# Pest-specific information
pest_information = {
    "Aphids": {
        "signs": (
            "Small, soft-bodied insects that often "
            "cluster around stems and new growth."
        ),
        "action": (
            "Isolate the plant, inspect new growth and "
            "leaf undersides, and consider rinsing "
            "visible aphids off with water."
        )
    },

    "Mealy Bugs": {
        "signs": (
            "White, waxy or cotton-like insects often "
            "found around leaf joints and stems."
        ),
        "action": (
            "Isolate the plant, inspect leaf joints "
            "and manually remove visible insects "
            "where practical."
        )
    },

    "Spider Mites": {
        "signs": (
            "Tiny mites, fine webbing and small "
            "discoloured spots on leaves."
        ),
        "action": (
            "Isolate the plant, closely inspect leaf "
            "undersides and look for fine webbing. "
            "Consider rinsing affected foliage."
        )
    }
}

# Image preprocessing
def prepare_image(image_path):

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        image = image.resize(
            image_size,
            Image.Resampling.BILINEAR
        )

        image = np.asarray(image, dtype=np.float32)

    image = np.expand_dims(image, axis=0)

    return preprocess_input(image)


# Prediction and downstream guidance
def identify_pest(image_path):

    if image_path is None:
        return {}, "Please upload a plant photograph."

    image = prepare_image(image_path)

    probabilities = model.predict(
        image, verbose=0
    )[0]

    scores = {
        label: float(score)
        for label, score in zip(classes, probabilities)
    }

    predicted_index = np.argmax(probabilities)
    predicted_class = classes[predicted_index]

    information = pest_information[predicted_class]

    guidance = (
        f"Predicted pest: {predicted_class}\n\n"
        f"Identifying characteristics:\n"
        f"{information['signs']}\n\n"
        f"Suggested next steps:\n"
        f"{information['action']}\n\n"
        "Important: Verify the identifying "
        "characteristics before applying treatment. "
        "The model only recognizes three pest categories "
        "and cannot determine whether a plant is healthy."
    )

    return scores, guidance


# Application interface
demo = gr.Interface(
    fn=identify_pest,

    inputs=gr.Image(
        type="filepath",
        label="Upload a plant photograph"
    ),

    outputs=[
        gr.Label(
            num_top_classes=3,
            label="Predicted Pest — Model Scores"
        ),

        gr.Textbox(
            label="Identification and Suggested Next Steps",
            lines=12
        )
    ],

    title="Common Household Plant Pest Identifier",

    description=(
        "Upload a photograph of a suspected plant pest. "
        "This experimental machine learning model "
        "classifies aphids, mealybugs and spider mites. "
        "Model scores are not calibrated probabilities "
        "of correctness."
    )
)

if __name__ == "__main__":
    demo.launch()
