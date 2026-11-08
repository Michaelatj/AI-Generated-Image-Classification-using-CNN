---
title: Real vs AI-Generated Image Classifier
emoji: 🖼️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app/app.py
pinned: false
---

# Real vs AI-Generated Image Classification

Aplikasi Gradio ini membandingkan gambar masukan dengan dua kelas: **Real**
dan **AI-generated**. Model yang dipakai adalah **Custom CNN v3**, dipilih
berdasarkan hasil test pada `final_testing_comparison.ipynb` dan
`artifacts/evaluation/final_test_scenarios_v3.csv`.

Model menerima gambar RGB 224 × 224. Sebelum masuk ke model, gambar dikonversi
ke RGB, di-center-crop menjadi persegi, lalu di-resize dengan bilinear.
Normalisasi `1/255` sudah berada di dalam model. Augmentasi yang digunakan
saat training tidak diterapkan saat inference.

## Performa test

Pada test split group-based (44.988 gambar), threshold validation **0,55**
menghasilkan:

| Metrik              |  Hasil |
| ------------------- | -----: |
| Accuracy            | 94,71% |
| Macro-F1            | 90,21% |
| Balanced accuracy   | 89,16% |
| ROC-AUC             | 97,44% |
| Recall Real         | 80,85% |
| Recall AI-generated | 97,48% |

Label 0 berarti Real; label 1 berarti AI-generated. Nilai sigmoid model adalah
probabilitas AI-generated. Nilai tersebut diklasifikasikan sebagai
AI-generated jika sekurang-kurangnya 0,55, selain itu Real. Confidence score
adalah probabilitas kelas yang diprediksi dan bukan jaminan kebenaran.

## Menjalankan secara lokal

Pastikan model terpilih tersedia di
`artifacts/models/custom_cnn_v3_best.keras`, lalu jalankan:

```powershell
py -3.11 -m venv "$env:USERPROFILE\venvs\ai-image-classifier"
& "$env:USERPROFILE\venvs\ai-image-classifier\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
python app/app.py
```

Buka alamat lokal yang dicetak Gradio (umumnya `http://127.0.0.1:7860`).
Untuk penggunaan deployment, folder Space minimal harus berisi `app/app.py`,
`app/inference.py`, `artifacts/configs/` yang dipakai aplikasi,
`artifacts/models/custom_cnn_v3_best.keras`, `requirements.txt`, dan file ini.
Folder dataset, `eda`, `evaluation`, `splits`, notebook, serta model lain tidak
diperlukan untuk inference.

## Deploy ke Hugging Face Spaces

Folder `hf_space/` adalah payload terpisah berisi hanya aplikasi, konfigurasi
inference, dan model terpilih. Jalankan perintah berikut dari root project di
PowerShell. Tentukan username Hugging Face dan nama Space yang ingin digunakan:

```powershell
$env:HF_USERNAME = "username-anda"
$env:HF_SPACE_NAME = "real-vs-ai-image-classifier"
$env:HF_GIT_EMAIL = "email-anda@example.com"
hf auth login --add-to-git-credential
hf repos create "$($env:HF_USERNAME)/$($env:HF_SPACE_NAME)" --repo-type space --space-sdk gradio

Push-Location hf_space
git init -b main
git lfs install
git config user.name "$($env:HF_USERNAME)"
git config user.email "$($env:HF_GIT_EMAIL)"
git remote add space "https://huggingface.co/spaces/$($env:HF_USERNAME)/$($env:HF_SPACE_NAME)"
git add README.md requirements.txt .gitignore app/app.py app/inference.py `
  artifacts/configs/deployment_config_v3.json `
  artifacts/configs/preprocessing_config.json `
  artifacts/configs/label_mapping.json `
  artifacts/models/custom_cnn_v3_best.keras
git status --short
git commit -m "Deploy Real vs AI image classifier"
git push -u space main
Pop-Location
```

Buat access token di <https://huggingface.co/settings/tokens> dengan izin tulis
untuk membuat/memperbarui Space. `hf auth login` meminta token melalui prompt
aman; jangan menaruh token di source code, file konfigurasi, atau command yang
disimpan. Model yang dipilih berukuran 3,77 MB, di bawah ambang 10 MB, sehingga
tidak perlu Git LFS untuk file ini. `git lfs install` hanya menyiapkan Git LFS
lokal; tidak ada file model lain yang di-track atau di-push.

URL Space setelah dibuat akan berbentuk
`https://huggingface.co/spaces/<username>/<space-name>`. Jika build gagal,
buka tab **Logs** pada halaman Space, periksa build/runtime log, lalu bagikan
pesan error tanpa token atau credential.

## Keterbatasan

- Dataset memiliki ketimpangan kelas sekitar 5:1 (AI-generated:Real).
- Pada test, recall kelas Real lebih rendah daripada recall kelas AI-generated;
  sebagian gambar Real dapat salah ditandai AI-generated.
- Hasil test menggambarkan dataset dan group-based split yang digunakan, bukan
  jaminan performa pada semua kamera, generator, edit, atau domain gambar.
- Skor confidence belum dikalibrasi dan tidak menggantikan pemeriksaan manusia.
- Aplikasi hanya membedakan dua kelas; tidak menentukan generator atau
  keaslian gambar dengan kepastian forensik.
