# Truth Shield 

**Truth Shield** is a powerful binary content detection system designed to precisely classify images and text as either **Authentic (Real)** or **AI-Generated**. Utilizing advanced machine learning models and a robust multi-signal ensemble approach, this project helps users navigate the digital landscape with confidence by cutting through the noise of synthetically generated content.

## Key Features

*   **Binary Decision Architecture:** Unambiguous, straightforward rulings—content is classified simply and strictly as either **Real** or **AI Detectable**, without confusing "uncertain" states.
*   **Multi-Modal Analysis:**
    *   **Image Detection:** Leverages an EfficientNet-B4 backend, enhanced by a multi-signal ensemble approach encompassing face detection, frequency analysis, metadata inspection, and file-path heuristics.
    *   **Text Detection:** Powered by a state-of-the-art Transformer model specifically tuned for synthetic text patterns.
*   **Immersive User Interface:** Features a stunning "cyber-security" aesthetic complete with CRT scanlines, glitch text animations for the title, and a dynamic, multi-phase progress visualization representing the analysis.
*   **Adaptive Feedback Loop:** Includes a system to capture user feedback on correct or incorrect classifications, enabling continuous model retraining and adaptation to novel AI-generation techniques.
*   **FastAPI Backend:** Built on top of a highly performant, asynchronous API framework capable of efficiently handling both standard requests and multi-part data uploads.

## Technology Stack

*   **Backend:** FastAPI, Uvicorn, Python 3.x
*   **Machine Learning:** PyTorch, Transformers (Hugging Face), Scikit-Learn
*   **Image Processing:** OpenCV (`opencv-python-headless`), Pillow
*   **Frontend:** HTML5, CSS3 (Vanilla + Custom Animations), JavaScript (Integrated into the backend for seamless single-file deployment serving)

## Project Structure

```text
truth_shield/
│
├── app.py                      # Main FastAPI application & HTML/JS frontend code
├── requirements.txt            # Python dependencies
├── feedback.json               # Local database for user feedback loop
│
├── models/                     # ML Model architectures and loading logic
│   ├── detector.py             # Image content detection logic (EfficientNet-B4 ensemble)
│   └── text_detector.py        # Text content detection logic (Transformer model)
│
├── uploads/                    # Temporary storage for uploaded files being analyzed
├── trained_model/              # Weights and architectures for the trained models
├── dataset/                    # Training and validation dataset directories
│   ├── AiArtData/
│   └── RealArt/
│
└── utils/                      # (Scripts such as train_model.py, organize_data.py, etc.)
```

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/truth-shield.git
    cd truth-shield/truth_shield
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On MacOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Ensure you have `pip` up to date, then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    Start the FastAPI server utilizing `uvicorn`:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    ```
    
5.  **Access the Interface:**
    Open your web browser and navigate to `http://localhost:8000` to interact with the Truth Shield UI.

## Model Training (Optional)

If you wish to retrain the models with your own local dataset or utilize the adaptive feedback loop, use the provided scripts in the root directory:

*   `train_model.py`: Train a base model from scratch using the `dataset/` directory.
*   `train_final.py`: Fine-tune the latest version of your models.
*   `retrain_with_feedback.py`: Kick off a retraining process using the historical data captured in `feedback.json`.
*   `organize_data.py`: Helper script to manage your `AiArtData` and `RealArt` directories.

## Contributing

Contributions, issues, and feature requests are welcome! 
Feel free to check the https://github.com/Hardik006-ds/Truth_shield/issues. 
## License

This project is open-source and available under the terms of the MIT License.
