# Oyster Classification Using YOLOv8

This project applies YOLOv8 models to classify oyster states—**Oyster-Closed**, **Oyster-Open**, and **Oyster-Indeterminate**—for use in sustainable aquaculture automation. Using annotated datasets and YOLOv8 variants (`n`, `s`, `m`), this project improves classification accuracy under challenging underwater conditions.

---

## 📁 File Manifest

Oyster_Orientation_Model/
├── REU_Oyster_2024_Improved-1/ # Legacy dataset version
├── REU_Oyster_2024_Improved-2/ # Current working dataset and config
│ └── data.yaml # Dataset config file
├── oyster_logs/ # YOLOv8 model logs and evaluation output
│ ├── model_yolov8n/
│ ├── model_yolov8s/
│ └── model_yolov8m/
│ ├── weights/ # Trained YOLOv8 weights (best.pt)
│ ├── confusion_matrix.png
│ ├── PR_curve.png
│ ├── results.csv
│ └── ...
├── yolov8n.pt # YOLOv8 Nano model
├── yolov8s.pt # YOLOv8 Small model
├── yolov8m.pt # YOLOv8 Medium model
├── yolov8x.pt # YOLOv8 Extra-Large model (not used)
├── OysterResearch.ipynb # Research notebook
├── Project Report.odt # Formal research report
├── scrapCode.txt # Archived/unstructured code
└── README.md # Project overview (this file)


---

## ⚙️ Configuration Instructions

Make sure the following file exists and is properly configured:

**REU_Oyster_2024_Improved-2/data.yaml**
```yaml
train: ../images/train
val: ../images/val
test: ../images/test
nc: 3
names: ['Oyster-Closed', 'Oyster-Open', 'Oyster-Indeterminate']

💻 Installation Instructions
You can get this repo by emailing ncorcoran1@gulls.salisbury.edu.
If it is ever public, then:
git clone <repo-url>
cd Oyster_Orientation_Model

Create and activate a virtual environment:

python3 -m venv my_virtual_env
source my_virtual_env/bin/activate

Install dependencies:
	Dependencies will be installed by navigating the .ipynb file
	you may pip install pandas torch torchvision torchaudio roboflow etc...

▶️Operating Instructions
Everything in the .ipynb file is in order and, if ran sequentially, will yield results.
It covers setup --> training --> validation/evaluation --> analysis

Evaluation results will include:

    Precision

    Recall

    mAP@0.5

    mAP@0.5:0.95

Output is printed to console and optionally exportable.
📫 Contact Information

    Nicholas Corcoran - ncorcoran1@gulls.salisbury.edu

    Dr. Yuanwei Jin – yjin@umes.edu

    Dr. Enyue Lu – ealu@salisbury.edu

👨‍🔬 Credits and Acknowledgments

This project was developed as part of the 2025 NSF REU Program in Machine Learning and Aquaculture at the University of Maryland Eastern Shore and Salisbury University.

Special thanks to:

    Dr. Yuanwei Jin

    Dr. Enyue Lu


🛠️ Known Issues / Bugs

    Dataset class imbalance may still impact final results.

    YOLOv8x was downloaded but not trained or validated.

    Minor class confusion between “Open” and “Indeterminate.”

    No support for video stream input (still-image classification only).

❓ Troubleshooting
Problem	Suggested Fix
Low accuracy	Rebalance the dataset and re-check label consistency.
CUDA OOM (Out of Memory) error	Switch to a smaller model (YOLOv8n), tweak settings (eg. downsize image resolution or batch size), 
and ensure that you are running on a powerful maching - a previous version of this project used google colab, which is a very
accessible resource to do machine learning with, and the free plan is not half bad :)
Validation outputs seem off	Make sure data.yaml points to the correct test set.
📅 Change Log

    Trained YOLOv8n, YOLOv8s, and YOLOv8m on rebalanced oyster dataset

    Applied Gaussian blur and grayscale augmentations

    Validated models with precision, recall, mAP scores

    Created detailed confusion matrices

    Packaged project structure and documentation

✅ License

This project is provided for educational and research purposes only.
If used in publicatios, please cite:

THIS WILL MOST LIKELY NOT GET PUBLISHED, ADD CITATION TO THE README IF IT DOES

---

Let me know if you'd like this turned into a downloadable file or if you want to host it in a GitHub repo.
n
