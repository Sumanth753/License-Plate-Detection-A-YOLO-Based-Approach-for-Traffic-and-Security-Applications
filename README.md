# License Plate Detection: A YOLO-Based Approach for Traffic and Security Applications

## Introduction
This project focuses on license plate detection using YOLO (You Only Look Once) for real-time vehicle identification. The system integrates image processing, database storage, and a user-friendly interface to automate vehicle authentication. The project is designed for applications in traffic management, security, and automated parking systems.

## Features
- **Real-time License Plate Detection:** Utilizes YOLO for high-speed and accurate plate recognition.
- **Data Extraction and Storage:** Extracts license plate numbers using OCR and stores them in an SQLite database.
- **User Interface:** Built with Tkinter for an interactive and intuitive user experience.
- **Database Management:** Enables querying, data export, and record management.
- **Batch Processing:** Supports multiple image/video inputs simultaneously.
- **Scalability:** Suitable for integration with traffic cameras and parking systems.

## Technologies Used
- **Deep Learning:** YOLO for object detection.
- **Python:** Backend processing using OpenCV, NumPy, and TensorFlow/PyTorch.
- **SQLite:** Lightweight database management.
- **Tkinter:** GUI for user interaction.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.x
- OpenCV
- NumPy
- TensorFlow/PyTorch
- Tesseract OCR
- SQLite3

### Steps to Install
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/License-Plate-Detection.git
   cd License-Plate-Detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   python main.py
   ```

## Usage
- **Upload an image/video** for detection.
- **View detected plates** and extracted information in the interface.
- **Search stored data** using license plate numbers.
- **Export records** to CSV or JSON.

## System Workflow
1. **Image/Video Input:** Accepts image files or live feed.
2. **Detection & Extraction:** YOLO identifies plates, and OCR extracts numbers.
3. **Database Integration:** Stores data in SQLite with timestamps.
4. **Report Generation:** Provides insights and analytics on detected plates.

## Results
- **Accuracy:** Achieves over 95% accuracy in controlled conditions.
- **Processing Speed:** Detects plates within 2 seconds per frame.
- **Improvement:** Outperforms existing models by 10% in accuracy and 15% in speed.

## Future Enhancements
- Support for multilingual and non-standard plates.
- Integration with IoT for real-time tracking.
- Dynamic learning for continuous accuracy improvement.

## Contributors
- **Sumanth S** - Presidency University
- **Chandrashekhar S** - Presidency University
- **Kiran DT** - Presidency University

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
For any queries, please reach out via GitHub issues or email.

---
**Acknowledgment:** This project was developed under the guidance of Prof. Muthukumar M and Prof. Vishwanath Y at Presidency University.
