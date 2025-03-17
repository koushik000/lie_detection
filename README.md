# Lie Detection Using Facial Expressions and Voice Recognition

## Overview
This project aims to detect deception by analyzing facial expressions and vocal features. It leverages OpenFace for facial feature extraction and Librosa for audio feature extraction. The extracted features are processed and used to train an LSTM-based deep learning model for lie detection.

## Features
- **Facial Expression Analysis**: Uses OpenFace to extract facial action units and other key points.
- **Voice Feature Extraction**: Uses Librosa to extract Mel-frequency cepstral coefficients (MFCCs) and other audio features.
- **LSTM-Based Classification**: Trained on labeled truth and lie samples to classify inputs as either true or deceptive.
- **Dataset Handling**: Processes truth and lie files separately for training and testing.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- OpenCV
- OpenFace
- Librosa
- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn

### Steps to Install
```bash
# Clone the repository
git clone https://github.com/your-repo/lie-detection.git
cd lie-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Data Preparation
1. Place truth and lie video files in `data/truth/` and `data/lie/` respectively.
2. Run the preprocessing script to extract features:
   ```bash
   python extractfeatures.py
   ```
then
```bash
   python combineFeatures.py
   ```

### Training the Model
```bash
python createModel.py
```

### Testing the Model
```bash
python test.py --input test_video.mp4
```

## File Structure
```


## Future Enhancements
- **Real-time Lie Detection**
- **Improved Feature Engineering**
- **Integration with Other Physiological Signals**

## Contributors
- Mokshgna Krishna Koushik
- P Venkata Kanith Kumar
- S Rohith Reddy
- Preetham Reddy
## License
MIT License

