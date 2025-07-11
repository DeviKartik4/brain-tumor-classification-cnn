# 🧠 Klasifikasi Tumor Otak: Menganalisis Citra MRI dengan Kekuatan CNN 🚀

Proyek ini merupakan eksplorasi dalam penerapan Convolutional Neural Networks (CNN) untuk tugas klasifikasi citra medis, khususnya mendeteksi keberadaan tumor otak dari citra MRI. Dengan memanfaatkan arsitektur CNN yang kuat, kami bertujuan untuk membangun model yang mampu membedakan citra otak dengan dan tanpa tumor secara akurat.

## 🛠️ Teknologi yang Digunakan

Proyek ini dibangun di atas fondasi ekosistem *Machine Learning* dan *Computer Vision* yang tangguh, menggunakan teknologi-teknologi berikut:

* **Python**: Bahasa pemrograman utama yang digunakan untuk seluruh implementasi kode.
* **TensorFlow/Keras**: Merupakan *framework Deep Learning* pilihan kami untuk membangun, melatih, dan mengevaluasi arsitektur model CNN. Keras sebagai API tingkat tinggi memungkinkan pengembangan model yang efisien.
* **OpenCV (Open Source Computer Vision Library)**: Digunakan secara ekstensif untuk tugas-tugas *Computer Vision*, seperti membaca citra, pra-pemrosesan (misalnya *resizing* dan konversi *grayscale*), serta manipulasi citra MRI lainnya.
* **NumPy**: Pustaka fundamental untuk komputasi numerik di Python, sangat penting untuk manipulasi *array* multi-dimensi dari data citra.
* **Matplotlib/Seaborn**: Digunakan untuk visualisasi data yang komprehensif, termasuk grafik *loss* dan *accuracy* selama pelatihan.
* **Scikit-learn (sklearn)**: Meskipun bukan *deep learning framework*, `scikit-learn` menyediakan berbagai alat untuk *preprocessing* data (misalnya `train_test_split`).
* **Google Colab**: Lingkungan *notebook* berbasis *cloud* yang digunakan untuk pengembangan proyek, menyediakan akses gratis ke *resource* komputasi (termasuk GPU) yang sangat mempercepat proses pelatihan model *Deep Learning*.

## 💡 Arsitektur Model (Convolutional Neural Network - CNN)

Arsitektur model yang diimplementasikan dalam proyek ini adalah `Sequential`, dengan detail lapisan sebagai berikut:

| Layer Type          | Fungsi                                    | Parameter Kunci                             |
| :------------------ | :---------------------------------------- | :------------------------------------------ |
| 🔵 Input Layer      | Menerima citra MRI yang dipra-proses      | `input_shape=(224, 224, 3)`                 |
| 🌀 `Conv2D`          | Mengekstraksi fitur lokal                 | 32 filter, kernel (3,3), `activation='relu'`, `padding='valid'` |
| 🔽 `MaxPooling2D`    | Mengurangi dimensi, fitur lebih kuat      | pool size (2,2)                             |
| 🧩 `Flatten`         | Meratakan *output* untuk lapisan *dense* | -                                           |
| 🔗 `Dense`            | Lapisan terhubung penuh                   | 256 neuron, `activation='relu'`             |
| 🚫 `Dropout`          | Mencegah *overfitting* | `rate=0.5`                                  |
| 🎯 `Output Layer`    | Output probabilitas klasifikasi biner     | 1 neuron, `activation='sigmoid'`            |


Model ini di-*compile* menggunakan `Adam` sebagai *optimizer* dan `binary_crossentropy` sebagai *loss function*, dengan `accuracy` sebagai metrik pemantauan kinerja.

## 🔄 Cara Kerja Sistem Klasifikasi

1.  **Pengambilan Data**: Citra MRI otak yang dibutuhkan diakses dan dimuat ke dalam lingkungan kerja. 
2.  **Preprocessing Citra**: Citra melalui beberapa tahap preprocessing. Ini termasuk *resizing* ke dimensi seragam (224x224 piksel), konversi ke *grayscale*, normalisasi nilai piksel (0-255 menjadi 0-1), dan augmentasi data (*e.g.*, rotasi, pergeseran, *flip*) untuk memperkaya variasi *training set*.
3.  **Pembagian Dataset**: Dataset dibagi menjadi set pelatihan, validasi, dan pengujian. Pembagian ini penting untuk melatih model secara efektif, memvalidasi kinerjanya selama pelatihan, dan melakukan evaluasi akhir secara objektif.
4.  **Pelatihan Model CNN**: Model CNN dilatih menggunakan data pelatihan yang telah dipra-proses. Selama *epoch* pelatihan, model belajar mengenali pola-pola spesifik yang terkait dengan tumor. Kinerja model dipantau pada *validation set*, dan *callbacks* seperti *Early Stopping* serta *ModelCheckpoint* digunakan untuk optimasi.
5.  **Evaluasi Akhir**: Setelah pelatihan selesai, model dievaluasi pada *test set* yang terpisah. Pada tahap ini, metrik kinerja utama dihitung dan disajikan, termasuk *Test Loss*, *Test Accuracy*.
6.  **Prediksi (Uji Coba dengan Gambar Eksternal)**: Model yang sudah terlatih dapat digunakan untuk memprediksi citra MRI otak baru. Citra eksternal ini juga akan melalui tahap pra-pemrosesan serupa sebelum diumpankan ke model. Model akan menghasilkan probabilitas biner yang menunjukkan keyakinannya terhadap adanya atau tidak adanya tumor.

## 📊 Hasil Prediksi 
Model ini menunjukkan kinerja yang menjanjikan dalam mengklasifikasikan citra MRI otak. Kinerja ini dapat diamati melalui grafik pelatihan yang menampilkan tren loss dan accuracy untuk training set dan validation set sepanjang epoch pelatihan. Grafik ini sangat membantu dalam memvisualisasikan konvergensi model dan mengidentifikasi potensi overfitting atau underfitting selama proses pembelajaran. Setelah pelatihan, model dievaluasi lebih lanjut pada test set untuk mengukur performa secara objektif. Hasil evaluasi ini mencakup evaluasi metrik kunci seperti Test Loss dan Test Accuracy. Berdasarkan output yang teramati, model berhasil mencapai accuracy: 0.8564 dan loss: 0.6342 pada salah satu step terakhir pelatihan.
