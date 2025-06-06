# Panduan Penggunaan Aplikasi Pengolahan Citra

Aplikasi ini adalah alat pengolahan citra berbasis web yang dibuat dengan Streamlit. Aplikasi ini menggabungkan fitur dari Tugas 1 hingga Tugas 7, mulai dari pemisahan kanal RGB hingga analisis tekstur. Panduan ini menjelaskan cara menggunakan setiap fitur dengan bahasa sederhana dan langkah-langkah yang mudah diikuti. Dokumentasi ini dapat diakses melalui halaman khusus di aplikasi, dengan setiap tugas ditampilkan dalam bagian terpisah (expander).

## Persiapan Awal

Sebelum menggunakan aplikasi, lakukan persiapan berikut:

1. **Instalasi Python dan Library**:
   - Pastikan Python 3.12 atau lebih baru terinstal.
   - **(Opsional) Gunakan Virtual Environment (venv)**:  
     Disarankan menggunakan *virtual environment* agar dependensi terisolasi dari sistem utama.  
     Jalankan perintah berikut di terminal:
     ```bash
     python -m venv venv
     source venv/bin/activate        # Untuk Linux/MacOS
     .\venv\Scripts\activate         # Untuk Windows
     ```
     Setelah aktif, instal library yang dibutuhkan:
     ```bash
     pip install -r requirements.txt
     ```
   - Jika tidak menggunakan virtual environment, cukup jalankan:
     ```bash
     pip install -r requirements.txt
     ```

2. **Struktur Folder Proyek**:
   - Pastikan folder proyek memiliki struktur berikut:
     ```
     PCD2025-Proyek-Kompilasi/
     ├── app.py
     ├── data/
     │   ├── cbir_dataset/        # Isi dengan gambar untuk CBIR
     │   ├── dataset/             # Untuk dataset wajah
     │   └── processed_dataset/   # Hasil proses wajah
     ├── modules/
     │   ├── ui/
     │   │   └── image_ui.py
     │   └── processing/
     │       ├── rgb_processor.py
     │       ├── advanced_image_processor.py
     │       ├── frequency_processor.py
     │       ├── face_processor.py
     │       ├── shape_processor.py
     │       ├── compression_processor.py
     │       └── cbir_processor.py
     ├── output/
     │   ├── compressed/          # Hasil kompresi
     │   └── ssim/                # Hasil metrik SSIM
     ```
   - Folder `data/cbir_dataset/` harus berisi gambar dengan nama seperti `kategori_nomor.jpg` (contoh: `forest_001.jpg`).
   - Folder `output/compressed/` dan `output/ssim/` akan menyimpan hasil kompresi.

3. **Menjalankan Aplikasi**:
   - Buka terminal di folder `PCD2025-Proyek-Kompilasi`.
   - Jalankan perintah:
     ```bash
     streamlit run app.py
     ```
   - Buka browser di `http://localhost:8501`.
   - Gunakan sidebar untuk memilih tugas atau klik tombol **Dokumentasi** untuk membaca panduan ini.

4. **Persiapan Tambahan**:
   - Siapkan kamera web untuk Tugas 4 (pengambilan wajah).
   - Siapkan file gambar (JPG, JPEG, PNG) untuk diunggah.
   - Pastikan folder `data/cbir_dataset/` berisi cukup gambar untuk Tugas 7 (CBIR).

## Cara Menggunakan Fitur

Berikut adalah panduan penggunaan untuk setiap fitur, dikelompokkan berdasarkan tugas. Setiap tugas memiliki langkah-langkah sederhana untuk membantu Anda memahami cara kerja aplikasi.

## Tugas 1 - RGB Split
**Deskripsi**: Mengunggah gambar dan menampilkan kanal warna Red, Green, dan Blue secara terpisah, beserta data array RGB.

**Langkah-langkah**:
1. Pada sidebar, pilih **Task 1: Image Upload & RGB Split**.
2. Klik **Browse files** untuk mengunggah gambar (JPG, JPEG, atau PNG).
3. Tunggu hingga gambar asli muncul di layar.
4. Lihat tiga gambar terpisah yang menunjukkan kanal Red, Green, dan Blue.
5. Gulir ke bawah untuk melihat data array RGB dalam format JSON.

## Tugas 2 - Arithmetic Operations
**Deskripsi**: Melakukan operasi penjumlahan, pengurangan, maksimum, minimum, atau invers pada gambar.

**Langkah-langkah**:
1. Pilih **Task 2: Arithmetic Operations** di sidebar.
2. Unggah gambar.
3. Pilih operasi dari dropdown: **add**, **subtract**, **max**, **min**, atau **inverse**.
4. Jika memilih operasi selain **inverse**, atur nilai (0–255) menggunakan slider.
5. Lihat gambar asli dan hasil operasi di layar.

## Tugas 2 - Logic Operations
**Deskripsi**: Melakukan operasi logika NOT, AND, atau XOR pada satu atau dua gambar.

**Langkah-langkah**:
1. Pilih **Task 2: Logic Operations**.
2. Unggah gambar pertama.
3. Pilih operasi: **not**, **and**, atau **xor**.
4. Jika memilih **and** atau **xor**, unggah gambar kedua.
5. Lihat gambar asli (dan gambar kedua jika ada) serta hasil operasi logika.

**Catatan**: Gambar harus memiliki ukuran yang sama untuk operasi **and** atau **xor**.

## Tugas 2 - Grayscale Conversion
**Deskripsi**: Mengubah gambar berwarna menjadi grayscale.

**Langkah-langkah**:
1. Pilih **Task 2: Grayscale Conversion**.
2. Unggah gambar berwarna.
3. Lihat gambar asli dan hasil konversi grayscale.

## Tugas 2 - Histogram Generation
**Deskripsi**: Menampilkan histogram intensitas untuk gambar grayscale dan kanal warna RGB.

**Langkah-langkah**:
1. Pilih **Task 2: Histogram Generation**.
2. Unggah gambar.
3. Lihat gambar asli, histogram grayscale, dan histogram warna (Red, Green, Blue).

## Tugas 2 - Histogram Equalization
**Deskripsi**: Meningkatkan kontras gambar dengan ekualisasi histogram.

**Langkah-langkah**:
1. Pilih **Task 2: Histogram Equalization**.
2. Unggah gambar.
3. Lihat gambar asli dan hasil ekualisasi histogram.

## Tugas 2 - Histogram Specification
**Deskripsi**: Menyesuaikan histogram gambar sumber agar mirip dengan histogram gambar referensi.

**Langkah-langkah**:
1. Pilih **Task 2: Histogram Specification**.
2. Unggah gambar sumber.
3. Unggah gambar referensi.
4. Lihat gambar sumber, referensi, dan hasil spesifikasi histogram.

**Catatan**: Gunakan gambar dengan ukuran dan tipe serupa untuk hasil optimal.

## Tugas 2 - Image Statistics
**Deskripsi**: Menampilkan rata-rata intensitas dan standar deviasi gambar.

**Langkah-langkah**:
1. Pilih **Task 2: Image Statistics**.
2. Unggah gambar.
3. Lihat gambar asli dan nilai statistik (rata-rata dan standar deviasi).

## Tugas 3 - Convolution & Frequency Operations
**Deskripsi**: Melakukan operasi konvolusi, padding, filter frekuensi, transformasi Fourier, atau pengurangan noise.

**Langkah-langkah**:
1. Pilih **Task 3: Convolution & Frequency Operations**.
2. Unggah gambar.
3. Pilih operasi dari dropdown:
   - **Convolution**: Pilih kernel (**average**, **sharpen**, **edge**).
   - **Padding**: Atur ukuran padding (1–100).
   - **Filter**: Pilih filter (**low**, **high**, **band**).
   - **Fourier**: Transformasi Fourier.
   - **Noise Reduction**: Kurangi noise.
4. Klik tombol **Process**.
5. Lihat gambar asli dan hasil operasi.

## Tugas 4 - Add Faces to Dataset
**Deskripsi**: Mengambil gambar wajah menggunakan kamera web dan menyimpannya ke dataset.

**Langkah-langkah**:
1. Pilih **Task 4: Add Faces to Dataset**.
2. Masukkan nama orang baru di kolom teks.
3. Klik **Tambahkan Wajah Baru**.
4. Hadapkan wajah ke kamera web hingga beberapa gambar diambil.
5. Lihat pratinjau gambar yang tersimpan.

**Catatan**: Pastikan kamera web aktif.

## Tugas 4 - Process Dataset Images
**Deskripsi**: Memproses gambar wajah di dataset dengan menambah noise, menghilangkan noise, atau menajamkan gambar.

**Langkah-langkah**:
1. Pilih **Task 4: Process Dataset Images**.
2. Pilih nama orang dari dropdown (jika dataset tidak kosong).
3. Pilih opsi: **Tambah Noise Salt and Pepper**, **Hilangkan Noise**, atau **Tajamkan Gambar**.
4. Klik **Proses Semua Gambar**.
5. Lihat pratinjau gambar hasil proses.

**Catatan**: Dataset disimpan di `data/dataset/`.

## Tugas 5 - Freeman Chain Code
**Deskripsi**: Menghasilkan kode rantai Freeman untuk kontur objek dalam gambar.

**Langkah-langkah**:
1. Pilih **Task 5: Freeman Chain Code**.
2. Unggah gambar.
3. Atur nilai threshold (0–255) untuk binarisasi.
4. Lihat plot kontur dan teks kode rantai Freeman.

**Catatan**: Gunakan gambar dengan kontras tinggi.

## Tugas 5 - Canny Edge Detection
**Deskripsi**: Mendeteksi tepi dalam gambar menggunakan algoritma Canny.

**Langkah-langkah**:
1. Pilih **Task 5: Canny Edge Detection**.
2. Unggah gambar.
3. Atur **Low Threshold** dan **High Threshold** (0–255).
4. Lihat hasil deteksi tepi.

## Tugas 5 - Integral Projection (Default)
**Deskripsi**: Menghasilkan proyeksi integral horizontal dan vertikal dari gambar.

**Langkah-langkah**:
1. Pilih **Task 5: Integral Projection (Default)**.
2. Centang **Gunakan Citra Teks Default** atau unggah gambar.
3. Lihat plot proyeksi integral.

## Tugas 5 - Integral Projection (Otsu)
**Deskripsi**: Menghasilkan proyeksi integral dengan threshold Otsu.

**Langkah-langkah**:
1. Pilih **Task 5: Integral Projection (Otsu)**.
2. Unggah gambar.
3. Lihat plot proyeksi integral.

## Tugas 6 - JPEG Compression
**Deskripsi**: Menganalisis kompresi JPEG dengan berbagai kualitas, termasuk ukuran file, PSNR, dan SSIM.

**Langkah-langkah**:
1. Pilih **Task 6: JPEG Compression**.
2. Unggah gambar.
3. Lihat:
   - Perbandingan gambar pada berbagai kualitas.
   - Grafik ukuran file vs. kualitas.
   - Grafik PSNR dan SSIM vs. kualitas.
   - Tabel metrik kompresi.

## Tugas 6 - PNG Compression
**Deskripsi**: Menganalisis kompresi PNG dan membandingkan ukuran file.

**Langkah-langkah**:
1. Pilih **Task 6: PNG Compression**.
2. Unggah gambar.
3. Lihat:
   - Perbandingan gambar.
   - Grafik ukuran file.
   - Tabel metrik kompresi.

**Catatan**: Hasil disimpan di `output/compressed/` dan `output/ssim/`.

## Tugas 7 - Content-Based Image Retrieval (CBIR)
**Deskripsi**: Mencari gambar serupa dari dataset berdasarkan warna, tekstur, atau kombinasi keduanya, dengan prediksi label menggunakan k-NN.

**Langkah-langkah**:
1. Pilih **Task 7: CBIR**.
2. Unggah gambar kueri.
3. Pilih tipe fitur: **Color**, **Texture**, atau **Combined**.
4. Lihat gambar kueri dan 6 gambar paling mirip beserta skor kemiripan.
5. Atur jumlah tetangga (k) untuk prediksi k-NN.
6. Lihat label prediksi.

**Catatan**: Pastikan `data/cbir_dataset/` berisi gambar.

## Tugas 7 - Color Space Analysis
**Deskripsi**: Menganalisis gambar dalam ruang warna RGB, XYZ, Lab, YCbCr, HSV, dan YIQ, serta membandingkan komponen luminansi.

**Langkah-langkah**:
1. Pilih **Task 7: Color Space Analysis**.
2. Unggah gambar.
3. Lihat:
   - Gambar asli RGB dan kanal RGB.
   - Gambar dalam ruang warna XYZ, Lab, YCbCr, HSV, YIQ, beserta kanal masing-masing.
   - Perbandingan luminansi (Y dari YCbCr, L dari Lab, Y dari YIQ, V dari HSV).

## Tugas 7 - Texture Analysis
**Deskripsi**: Menganalisis tekstur gambar menggunakan fitur statistik, GLCM, LBP, filter Gabor, dan energi tekstur Law.

**Langkah-langkah**:
1. Pilih **Task 7: Texture Analysis**.
2. Unggah gambar.
3. Lihat:
   - Gambar asli dan grayscale.
   - Fitur statistik (mean, variance, std dev).
   - Matriks GLCM dan tabel fitur.
   - Peta dan histogram LBP.
   - Hasil filter Gabor.
   - Peta energi tekstur Law.

## Troubleshooting

Jika mengalami masalah, coba solusi berikut:

- **Aplikasi tidak berjalan**:
  - Pastikan semua library terinstal (`pip install -r requirements.txt`).
  - Periksa struktur folder proyek.
  - Jalankan `streamlit run app.py` dari direktori proyek (hasil kloning repository).

- **Error saat mengunggah gambar**:
  - Pastikan format gambar adalah JPG, JPEG, atau PNG.
  - Coba gambar dengan ukuran lebih kecil.

- **CBIR tidak menampilkan hasil**:
  - Pastikan folder `data/cbir_dataset/` berisi gambar.

- **Kamera web tidak berfungsi (Tugas 4)**:
  - Periksa koneksi kamera web dan izinkan akses di browser.
  - Pastikan driver kamera terinstal.

- **Plot atau gambar tidak muncul**:
  - Refresh halaman browser.
  - Pastikan `matplotlib` dan `opencv-python` terinstal.
