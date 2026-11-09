# Real vs AI-Generated Image Classification

Klasifikasi biner gambar **Real** vs **AI-Generated** memakai dua model yang
dibandingkan secara adil pada split test yang sama:

- **Model A: Custom CNN** (±3,8 MB), dipakai sebagai model default.
- **Model B: MobileNetV2** (transfer learning, ±19 MB).

Demo online: **[Hugging Face Space](https://huggingface.co/spaces/MichaelAndreasTjendra/AI-Image-Classification)**
(isi dengan URL Space Anda).

## Dataset

[Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset)
(`Label_A`: Real = 0, AI-Generated = 1). Gambar AI berasal dari Stable
Diffusion 2.1, Stable Diffusion XL, Stable Diffusion 3, DALL-E 3, dan
Midjourney v6; gambar Real berasal dari COCO.

Split dibuat **berbasis grup caption** agar gambar dari caption yang sama tidak
bocor antar split.

| Split      | Gambar | Real  | AI-Generated | Grup caption |
| ---------- | -----: | ----: | -----------: | -----------: |
| Train      | 42.018 | 7.003 |       35.015 |        4.320 |
| Validation |  8.994 | 1.499 |        7.495 |          926 |
| Test       | 44.988 | 7.498 |       37.490 |        4.629 |

Rasio kelas sekitar 1:5 (Real : AI-Generated) di semua split.

## Preprocessing

Gambar dikonversi ke RGB, di-*center-crop* menjadi persegi, lalu di-resize ke
224 × 224 (bilinear). Input model adalah piksel mentah 0-255; normalisasi ada
di dalam model (`1/255` untuk Custom CNN, `[-1, 1]` untuk MobileNetV2).
Augmentasi hanya dipakai saat training.

## Hasil pada test split (44.988 gambar)

Threshold dipilih pada **validation** (maksimum macro-F1) dan hanya dibaca saat
testing, tidak pernah dituning pada test.

| Model       | Skenario             | Accuracy | Macro-F1 | Balanced acc. | ROC-AUC | Recall Real | Recall AI |
| ----------- | -------------------- | -------: | -------: | ------------: | ------: | ----------: | --------: |
| Custom CNN  | threshold 0,50       |   94,72% |   90,10% |        88,50% |  97,44% |      79,17% |    97,83% |
| Custom CNN  | validation (0,42)    |   94,56% |   89,53% |        87,04% |  97,44% |      75,77% |    98,31% |
| MobileNetV2 | threshold 0,50       |   90,07% |   83,71% |        87,15% |  94,99% |      82,77% |    91,53% |
| MobileNetV2 | validation (0,29)    |   91,96% |   85,05% |        83,96% |  94,99% |      71,97% |    95,95% |

Custom CNN lebih baik di hampir semua metrik dan jauh lebih kecil, sehingga
dijadikan model default.

Akurasi per sumber gambar (threshold validation):

| Sumber                     | Custom CNN | MobileNetV2 |
| -------------------------- | ---------: | ----------: |
| Real                       |     75,77% |      71,97% |
| Stable Diffusion 2.1       |     98,43% |      95,44% |
| Stable Diffusion XL        |     99,57% |      96,16% |
| Stable Diffusion 3         |     97,64% |      94,91% |
| DALL-E 3                   |     97,01% |      97,40% |
| Midjourney v6              |     98,92% |      95,87% |

Label 0 = Real, label 1 = AI-Generated. Keluaran sigmoid adalah probabilitas
AI-Generated; gambar dianggap AI-Generated jika nilainya ≥ threshold. Skor ini
bukan tingkat kepastian dan belum dikalibrasi.

## Struktur repository

```
├── notebooks/
│   ├── eda_data_preparation.ipynb       # EDA, group split, konfigurasi
│   ├── model_a_custom_cnn.ipynb         # training & evaluasi Custom CNN
│   ├── model_b_mobilenetv2.ipynb        # training & evaluasi MobileNetV2
│   └── final_testing_comparison_new.ipynb  # perbandingan final di test split
├── artifacts/
│   ├── configs/      # deployment_config.json, preprocessing_config.json, label_mapping.json, dataset_manifest.json
│   ├── eda/          # distribusi label, split, resolusi
│   ├── evaluation/   # metrik, confusion matrix, ROC, Grad-CAM, error analysis
│   └── models/       # custom_cnn_best.keras, mobilenetv2_best.keras
├── requirements.txt  # dependensi notebook
└── README.md
```

Urutan notebook: EDA → Model A → Model B → final testing.

## Demo Hugging Face Space

Aplikasi Gradio (`app.py`) memuat kedua model, memilih threshold masing-masing
dari konfigurasi, dan menampilkan prediksi, skor P(AI-Generated), serta peta
**Grad-CAM** (area hangat = bagian gambar yang menaikkan skor AI-Generated).

File Space (repo terpisah di Hugging Face):

```
├── README.md          # berisi konfigurasi Space (YAML) di bagian atas
├── app.py
├── requirements.txt
└── models/
    ├── custom_cnn_best.keras
    └── mobilenetv2_best.keras   # >10 MB, di-track dengan Git LFS
```

Menjalankan secara lokal (dari folder Space):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Buka alamat yang dicetak Gradio (umumnya `http://127.0.0.1:7860`).

Deploy/update Space:

```bash
hf auth login --add-to-git-credential
git lfs install
git remote add space https://huggingface.co/spaces/<username>/<nama-space>
git add README.md app.py requirements.txt models/
git commit -m "Update Space"
git push space main
```

Token dibuat di <https://huggingface.co/settings/tokens> dengan izin tulis;
jangan menyimpannya di kode atau file konfigurasi. Mengganti nama file model
tidak mengubah URL Space; yang perlu disesuaikan hanya path di `app.py`
(`MODELS`) agar cocok dengan nama file di folder `models/`.

## Menjalankan notebook

Notebook dijalankan di Kaggle atau Colab (butuh GPU untuk training).

```bash
pip install -r requirements.txt
```

Notebook meng-clone repo ini, membaca `artifacts/`, dan menulis hasil
evaluasi kembali ke sana.

## Keterbatasan

- Dataset timpang (±5:1 AI-Generated : Real), sehingga recall kelas Real
  (72-83%) lebih rendah daripada recall AI-Generated; sebagian gambar Real
  dapat salah ditandai AI-Generated.
- Gambar Real berasal dari COCO. Foto dari kamera/HP, hasil edit, kompresi, atau
  generator lain bisa berperilaku berbeda dan lebih sering salah dikenali.
- Hasil test menggambarkan dataset dan group-based split yang dipakai, bukan
  jaminan performa di domain lain.
- Skor model sering sangat yakin (mendekati 0 atau 1) walau salah; Grad-CAM
  menjelaskan perilaku model, bukan bukti bahwa prediksi benar.
- Aplikasi hanya membedakan dua kelas; tidak menentukan generator maupun
  keaslian gambar secara forensik, dan tidak menggantikan pemeriksaan manusia.
