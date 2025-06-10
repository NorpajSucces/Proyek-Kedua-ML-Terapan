# 📈 Laporan Proyek Sistem Rekomendasi

**Nama Proyek:** Sistem Rekomendasi Film Menggunakan Content Based Filtering

**Nama Anda:** Zhafran Pradistyatama Kuncoro

**Email:** zhafrankuncoro@gmail.com

# 1. Project Overview

## Latar Belakang
Rekomendasi film merupakan fitur utama dalam berbagai layanan streaming modern, seperti Netflix dan Disney+, untuk meningkatkan pengalaman pengguna. Dengan jumlah film yang terus bertambah, pengguna membutuhkan sistem yang dapat membantu mereka menemukan film yang sesuai dengan preferensi pribadi mereka.

Proyek ini bertujuan untuk membangun sistem rekomendasi film berbasis Content-Based Filtering yang hanya menggunakan informasi genre dari dataset movies.csv. Sistem ini akan memberikan rekomendasi berdasarkan kemiripan genre antar film.

# 2. Business Understanding
## Problem Statements
1. **Kesulitan Menemukan Film Baru yang Relevan** <br>
   Pengguna sering bingung dalam memilih film karena banyaknya pilihan yang tersedia, sehingga mereka membutuhkan sistem yang dapat memberikan rekomendasi berdasarkan preferensi mereka.

2. **Tidak Tersedianya Data Riwayat untuk Pengguna Baru (Cold-Start)** <br>Banyak pengguna baru yang belum memiliki histori menonton atau memberikan rating, sehingga pendekatan seperti collaborative filtering tidak dapat bekerja.


## Goals
1. Meningkatkan Pengalaman Pengguna dalam Menemukan Film <br>
Membangun sistem yang mampu memberikan rekomendasi film yang relevan dan dipersonalisasi kepada pengguna berdasarkan judul film yang pernah ditonton atau genre film yang mereka sukai untuk mengatasi kebingungan akibat banyaknya pilihan.

2. Mengatasi Masalah Cold-Start untuk Pengguna Baru <br>
Mengembangkan model rekomendasi yang dapat berfungsi secara efektif sejak awal bagi pengguna baru, tanpa memerlukan data riwayat tontonan atau rating sebelumnya.

## Soluation Statement
Pendekatan yang digunakan adalah Content-Based Filtering. Sistem akan:
* Melakukan One Hot Encoding pada kolom genres untuk mendapatkan representasi fitur film
* Menghitung cosine similarity antar film
* Memberikan rekomendasi berdasarkan:
  * Input judul film
  * Input genre pilihan pengguna

# 3. Data Understanding
## Sumber Data
Dataset ini diperoleh dari Kaggle:
[Movie recomendation pjct](https://www.kaggle.com/datasets/sayan0211/movie-recomendation-pjct/data)

## Kondisi Data
1. Nama File: `movies.csv`
* Jumlah Kolom: 3
* Jumlah Baris: 9742

3. Fitur yang Digunakan:
* `movieId`: ID unik film
* `title`: Judul film
* `genres`: Genre film dalam format string (contoh: "Action|Adventure")

3. Insight:
* Missing Value: Tidak terdapat missing value
* Duplikat: Terdapat 5 Duplikat pada Kolom tittle
* Jumlah Genres 
  ![download](https://github.com/user-attachments/assets/950d96d0-1194-4d4f-8570-40e714ca6bde)

# 4. Data Preparation

1. Drop Data duplikat
```
# Drop data duplikat yang ada di kolom 'title'
film = film.drop_duplicates('title')
len(film)
```
Output: 9737 

2. Memilih Kolom yang akan digunakan
```
# Memilih hanya kolom 'title' dan 'genres'
film_final = film[['title', 'genres']]

# Menampilkan 5 baris pertama dari subset
film_final.head()
```
output: <br>
![image](https://github.com/user-attachments/assets/b4cb05a1-97b8-4f6f-b79c-379878616a51)

3. Melakukan One Hot Encoding
* Tujuan: Tujuannya adalah untuk mengubah data kategori (seperti nama genre film: "Action", "Comedy", "Drama") menjadi format yang dapat diproses oleh algoritma machine learning. Banyak algoritma machine learning membutuhkan input berupa angka, bukan teks kategori.
* Cara Kerja: Untuk setiap nilai unik dalam sebuah kolom kategori, one-hot encoding akan membuat kolom baru. Jika sebuah baris memiliki nilai kategori tersebut, maka kolom baru yang sesuai akan diisi dengan angka 1, dan kolom-kolom lainnya akan diisi dengan 0.

Output hasil One Hot Encoding: <br> ![image](https://github.com/user-attachments/assets/390efe4b-aca4-4b07-b5c6-d89c70b68bd6) <br>
Secara keseluruhan, kode ini mempersiapkan data genre film untuk digunakan dalam model rekomendasi berbasis konten dengan mengubah representasi string genre menjadi representasi numerik dalam bentuk vektor one-hot encoding. Representasi vektor ini kemudian digunakan untuk menghitung kesamaan antar film berdasarkan genre mereka.

# 5. Modeling
`Content-Based Filtering` adalah teknik sistem rekomendasi yang menyarankan item kepada pengguna berdasarkan kemiripan konten atau atribut dari item-item yang pernah disukai oleh pengguna tersebut di masa lalu.
<br> <br>

* Mengambil vektor one-hot encoding genre untuk setiap film dari kolom genre_vector.
* Menggabungkan vektor-vektor ini menjadi satu array NumPy 2 dimensi.
* Mengkonversi array NumPy tersebut menjadi matriks jarang (sparse matrix) menggunakan format CSR.

```
# Konversi genre_vector ke sparse matrix untuk efisiensi
genre_matrix = np.array(list(film_final['genre_vector']))
sparse_genre_matrix = csr_matrix(genre_matrix)
```
<br>
Alasan utama melakukan konversi ke sparse matrix adalah efisiensi memori dan komputasi. Karena matriks genre Anda berisi banyak angka nol, menggunakan format sparse matrix akan mengurangi jumlah memori yang dibutuhkan untuk menyimpan data tersebut. Ini sangat penting untuk dataset yang lebih besar. Selain itu, operasi matematika (seperti perhitungan cosine similarity) pada sparse matrix seringkali lebih cepat dibandingkan pada matriks padat (dense matrix) biasa karena algoritma dapat memanfaatkan fakta bahwa banyak elemen adalah nol.

## 5.1 Rekomendasi Berdasarkan Input Film
Kode ini mendefinisikan sebuah fungsi Python bernama `recommend_movies` yang bertujuan untuk memberikan rekomendasi film berdasarkan kesamaan genre dengan film input.
<br>

Contoh output penggunaan dari fungsi `recommend_movies`. Dengan input `movie_title = "Jumanji (1995)"`
<br>

Outuput: <br>
![image](https://github.com/user-attachments/assets/ccba94c1-b136-4daf-bd44-5cc4b824d2c1)

## 5.2 Rekomendasi Berdasarkan Input Genre
Mendefinisikan fungsi Python lain bernama `recommend_movies_by_genres`. Fungsi ini dirancang untuk memberikan rekomendasi film berdasarkan genre yang Anda masukkan, bukan berdasarkan judul film tertentu. Ini memberikan fleksibilitas lain dalam cara pengguna bisa mendapatkan rekomendasi.

Contoh output penggunaan dari fungsi `recommend_movies_by_genres`. Dengan input `input_genres = "Adventure, Children, Fantasy"`
Output: <br>
![image](https://github.com/user-attachments/assets/dc93b307-6906-4bda-a507-2df5bc5e0bc7)

## Kelebihan dan Keterbatasan:
* Kelebihan:
  * Tidak memerlukan data pengguna (aman untuk pengguna baru)
  * Mudah diimplementasikan dan efisien
* Keterbatasan:
  * Rekomendasi terbatas pada genre yang mirip
  * Tidak mempertimbangkan popularitas atau kualitas film

# 6. Evaluation
Pengujian kuantitatif dilakukan untuk mengukur kinerja model secara objektif menggunakan metrik Precision@20 dan Recall@20. Pengujian ini dilakukan untuk kedua skenario rekomendasi: berdasarkan input judul film dan input genre. <br>
1. Rekomendasi pertama: Menghasilkan output yang memuaskan dan sesuai dengan input judul film
Pengujian dilakukan dengan mengambil contoh kasus pada film 'Jumanji (1995)'. Hasil yang didapatkan adalah sebagai berikut:

| Metrik | Nilai |
| :--- | :---: |
| Precision@20 | 95.0% |
| Recall@20 | 82.6% |

<br>

Analisis: Nilai presisi yang sangat tinggi (95.0%) menunjukkan akurasi model dalam memberikan rekomendasi yang genrenya paling serupa. Skor recall sebesar 82.6% juga menunjukkan bahwa model mampu menjangkau sebagian besar dari total film yang paling relevan di dalam dataset. <br>
![image](https://github.com/user-attachments/assets/a1133d60-2af3-49bc-b136-c14456cc969e)

2. Rekomendasi kedua: Menghasilkan output yang memuaskan dan sesuai dengan input genre
Pengujian dilakukan dengan memberikan input genre "Adventure, Children, Fantasy". Hasil yang didapatkan adalah: <br> <br>

| Metrik | Nilai |
| :--- | :---: |
| Precision@20 | 100.0% |
| Recall@20 | 16.1% |

<br> 

Analisis: Presisi 100% membuktikan bahwa sistem sangat andal dalam merekomendasikan film yang sesuai dengan kategori genre yang diminta. Nilai recall sebesar 16.1% merupakan hal yang wajar dalam sistem rekomendasi Top-K, yang mengindikasikan bahwa daftar 20 teratas menampilkan sebagian dari total keseluruhan film relevan yang ada.

![image](https://github.com/user-attachments/assets/79b2a4fc-4c9b-4c23-9542-7bf3dd124b83)


# 7. Kesimpulan

Sistem rekomendasi film berbasis Content-Based Filtering berhasil dikembangkan dengan menggunakan informasi genre dari dataset movies.csv. Sistem ini mampu memberikan rekomendasi berdasarkan judul film atau genre yang dipilih oleh pengguna.
Kedepannya, sistem dapat ditingkatkan dengan menggabungkan pendekatan Collaborative Filtering atau menerapkan Hybrid Model untuk menghasilkan rekomendasi yang lebih personal dan akurat.





