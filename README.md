# plant-pest-identification

# Common Household Plant Pest Detection

An image classification system developed for TME-6015:
AI/ML Workflow Design at the University of New Brunswick.

## Supported Classes

- Aphids
- Mealybugs
- Spider mites

## Dataset

The cleaned dataset contains 441 images:

| Class | Training | Validation | Testing | Total |
|---|---:|---:|---:|---:|
| Aphids | 104 | 23 | 22 | 149 |
| Mealybugs | 104 | 22 | 22 | 148 |
| Spider mites | 101 | 22 | 21 | 144 |
| Total | 309 | 67 | 65 | 441 |

The dataset is not publicly distributed through this
repository pending verification of image redistribution
permissions.

## Implemented Models

- VGG16 + Support Vector Machine (SVM)
- VGG16 + Dense Classifier (Unregularized Baseline)
- VGG16 + Dense Classifier (Dropout, Augmentation
  and Early Stopping)
- VGG16 + Dense Classifier (Dropout and Early Stopping)
- VGG16 with Frozen Convolutional Layers
- Fine-Tuned VGG16 (Block 5)

## Repository Structure

```text
plant-pest-classification/
├── README.md
├── requirements.txt
├── app.py
├── TME6015_Final_project.ipynb
└── outputs/
    ├── Trained models
    ├── Configuration files
    ├── Training histories
    ├── Evaluation metrics
    └── Performance visualizations
```

## Installation

Download or clone this repository.

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Training

Google Colab with a GPU runtime is recommended for
model training.

1. Open `TME6015_Final_project.ipynb` in Google Colab.
2. Navigate to Runtime > Change runtime type.
3. Select a T4 GPU.
4. Upload `plant_pests_cleaned.zip` when prompted,
   using an authorized copy of the dataset.
5. Run the notebook cells sequentially.

The notebook includes dataset preparation, feature
extraction, model training, hyperparameter tuning,
regularization, fine-tuning, and performance evaluation.

Training outputs are saved in the `outputs/` directory.

## Pretrained Models

The larger trained models are available under the
Releases section of this repository (v1.0.0):

- `vgg16_warmup.keras`
- `vgg16_finetuned.keras`

Download these files and place them in the `outputs/`
directory.

The fine-tuned VGG16 model is used by the downstream
application.

## Running the Application

Ensure the following files are present:

```text
app.py
requirements.txt
outputs/
├── app_config.json
└── vgg16_finetuned.keras
```

Launch the Gradio application:

```bash
python app.py
```

Open the URL displayed in the terminal.

Upload a photograph of a suspected plant pest to
generate a classification, model scores, identifying
characteristics, and suggested next steps.

Alternatively, the application can be launched directly
from the Google Colab notebook.

## Evaluation Results

The `outputs/` directory contains:

- Model comparison results
- Training and validation accuracy curves
- Training and validation loss curves
- Classification reports
- Confusion matrices
- Hyperparameter tuning results
- Dataset split manifest

## Limitations

The application is an experimental prototype supporting
only three pest categories. It cannot determine whether
a plant is healthy or identify unsupported pest species.

Model scores should not be interpreted as calibrated
probabilities of correctness.

## Technologies

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Pillow
- Gradio
- Google Colab
