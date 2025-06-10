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
```
# Pisahkan genre menjadi list
film_final['genres'] = film_final['genres'].str.split('|')

# Identifikasi genre unik
all_genres = set()
for genres in film_final['genres']:
    all_genres.update(genres)
all_genres = sorted(list(all_genres))  # Urutkan untuk konsistensi

# Fungsi untuk membuat vektor one-hot encoding
def genres_to_vector(genres):
    return [1 if genre in genres else 0 for genre in all_genres]

# Terapkan one-hot encoding
film_final['genre_vector'] = film_final['genres'].apply(genres_to_vector)

film_final
```
Output: <br> ![image](https://github.com/user-attachments/assets/390efe4b-aca4-4b07-b5c6-d89c70b68bd6) <br>
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
```
# Fungsi untuk merekomendasikan film
def recommend_movies(movie_title, df, genre_matrix, top_n=5):
    try:
        # Cari indeks film yang dipilih
        movie_idx = df[df['title'] == movie_title].index[0]
        
        # Ambil vektor genre dari film yang dipilih
        movie_vector = genre_matrix[movie_idx]
        
        # Hitung cosine similarity dengan semua film
        similarities = cosine_similarity(movie_vector, genre_matrix)[0]
        
        # Dapatkan indeks film dengan kemiripan tertinggi
        similar_indices = similarities.argsort()[-top_n-1:-1][::-1]  # Ambil top_n, abaikan film itu sendiri
        
        # Kembalikan judul, genre, dan skor kemiripan
        recommendations = df.iloc[similar_indices][['title', 'genres']].copy()
        recommendations['similarity_score'] = similarities[similar_indices]
        
        return recommendations
    except IndexError:
        return f"Film '{movie_title}' tidak ditemukan di dataset."
```
```
# Contoh penggunaan
movie_title = "Jumanji (1995)"
print(f"Rekomendasi untuk '{movie_title}':")
recommendations = recommend_movies(movie_title, film_final, sparse_genre_matrix, top_n=10)
recommendations
```
<br>

Kode ini adalah contoh cara memanggil dan menggunakan fungsi `recommend_movies`.
<br>


Outuput: <br>
![image](https://github.com/user-attachments/assets/ccba94c1-b136-4daf-bd44-5cc4b824d2c1)

## 5.2 Rekomendasi Berdasarkan Input Genre
Kode ini mendefinisikan fungsi Python lain bernama `recommend_movies_by_genres`. Fungsi ini dirancang untuk memberikan rekomendasi film berdasarkan genre yang Anda masukkan, bukan berdasarkan judul film tertentu. Ini memberikan fleksibilitas lain dalam cara pengguna bisa mendapatkan rekomendasi.

```
# Fungsi rekomendasi Berdasarkan genre
def recommend_movies_by_genres(input_genres, df, genre_matrix, all_genres, top_n=5, exact_match=False):
    try:
        # Ubah input genre menjadi list (misalnya, "Action, Comedy" -> ["Action", "Comedy"])
        input_genres = [genre.strip() for genre in input_genres.split(',')]
        
        # Validasi genre
        invalid_genres = [genre for genre in input_genres if genre not in all_genres]
        if invalid_genres:
            return f"Genre tidak valid: {invalid_genres}. Genre yang tersedia: {all_genres}"
        
        # Buat vektor one-hot encoding untuk input genre
        input_vector = np.array([[1 if genre in input_genres else 0 for genre in all_genres]])
        
        # Hitung cosine similarity antara input dan semua film
        similarities = cosine_similarity(input_vector, genre_matrix)[0]
        
        # Jika exact_match=True, hanya rekomendasikan film yang mengandung semua genre input
        if exact_match:
            mask = df['genres'].apply(lambda x: all(genre in x for genre in input_genres))
            if not mask.any():
                return f"Tidak ada film yang mengandung semua genre: {input_genres}"
            similar_indices = similarities[mask].argsort()[-top_n:][::-1]
            recommendations = df[mask].iloc[similar_indices][['title', 'genres']].copy()
            recommendations['similarity_score'] = similarities[mask][similar_indices]
        else:
            # Ambil top_n film dengan kemiripan tertinggi
            similar_indices = similarities.argsort()[-top_n:][::-1]
            recommendations = df.iloc[similar_indices][['title', 'genres']].copy()
            recommendations['similarity_score'] = similarities[similar_indices]
        
        return recommendations
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"
```
```
# Rekomendasi Berdasarkan genre
input_genres = "Adventure, Children, Fantasy"
print(f"\nRekomendasi untuk genre: {input_genres}")
recommendations_by_genres = recommend_movies_by_genres(input_genres, film_final, sparse_genre_matrix, all_genres, top_n=10, exact_match=False)
recommendations_by_genres
```
Kode ini adalah contoh penggunaan dari fungsi `recommend_movies_by_genres`
Output: <br>
![image](https://github.com/user-attachments/assets/dc93b307-6906-4bda-a507-2df5bc5e0bc7)

## Kelebihan dan Keterbatasan:
* Kelebihan:
  * Tidak memerlukan data pengguna (aman untuk pengguna baru)
  * Mudah diimplementasikan dan efisien
* Keterbatasan:
  * Rekomendasi terbatas pada genre yang mirip
  * Tidak mempertimbangkan popularitas atau kualitas film

# 6. Evaluasi
* Rekomendasi pertama: Menghasilkan output yang memuaskan dan sesuai dengan input judul film
* Rekomendasi kedua: Menghasilkan output yang memuaskan dan sesuai dengan input genre
* Output yang dihasilkan dari rekomendasi 1 dan 2 sama

# 7. Kesimpulan

Sistem rekomendasi film berbasis Content-Based Filtering berhasil dikembangkan dengan menggunakan informasi genre dari dataset movies.csv. Sistem ini mampu memberikan rekomendasi berdasarkan judul film atau genre yang dipilih oleh pengguna.
Kedepannya, sistem dapat ditingkatkan dengan menggabungkan pendekatan Collaborative Filtering atau menerapkan Hybrid Model untuk menghasilkan rekomendasi yang lebih personal dan akurat.





