'''
    Deployment Project CVTech
    Orbit Future Academy - AI Jobs - MBKM Batch 6
    Tim Deployment
    2024
'''

# Import Module
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import cv2
import time
import os
import base64
from streamlit_option_menu import option_menu

# Judul Web page
st.set_page_config(page_title="CVTech", page_icon="🌽", layout="wide")

# Class names dan explanation
class_names=['Blight', 'Common Rust', 'Gray Leaf Spot', 'Sehat']
class_explanations={
    'Blight': [
        "Blight adalah penyakit yang disebabkan oleh jamur atau bakteri. Penyakit ini biasanya menyebabkan daun-daun tanaman jagung menjadi layu, berubah warna menjadi coklat atau hitam, dan kemudian mati. Blight dapat menyebar dengan cepat, terutama dalam kondisi lembab.",
        "Penanganan atau pengendalian Blight pada tanaman jagung dapat melibatkan beberapa langkah pencegahan dan tindakan pengendalian. Berikut adalah beberapa metode umum yang dapat dilakukan:",
        "1. Rotasi Tanaman: Praktik rotasi tanaman dapat membantu mengurangi risiko infeksi Blight. Dengan merotasi tanaman secara berkala, Anda dapat mengurangi jumlah patogen Blight yang bertahan di tanah.",
        "2. Pemilihan Varietas Tahan Penyakit: Memilih varietas jagung yang tahan terhadap Blight dapat membantu mengurangi risiko infeksi. Konsultasikan dengan penjual benih atau katalog varietas untuk mencari varietas yang memiliki ketahanan terhadap penyakit ini.",
        "3. Pengaturan Jarak Tanam: Menjaga jarak yang cukup antara tanaman jagung dapat membantu meningkatkan sirkulasi udara di antara tanaman, mengurangi kelembaban, dan mengurangi risiko penyebaran Blight.",
        "<a href='https://plantix.net/id/library/plant-diseases/100161/southern-leaf-blight-of-maize/'style='font-size: 20px; text-decoration: none; font-weight: bold;'>Klik untuk informasi selanjutnya</a>🍃🍃🍃"
    ],
    'Common Rust': [
        "Common Rust (Karatan Umum) adalah penyakit jamur yang biasanya ditemukan pada daun-daun tanaman jagung. Gejalanya termasuk munculnya bercak-bercak berwarna oranye atau coklat pada daun jagung. Penyakit ini dapat menyebabkan penurunan produksi jika tidak diatasi dengan tepat.",
        "Menangani Common Rust (Karatan Umum) pada tanaman jagung dapat dilakukan dengan beberapa langkah pencegahan dan pengendalian. Berikut adalah beberapa langkah efektif yang dapat diambil:",
        "1. Pilih Varietas Tahan Penyakit: Tanam varietas jagung yang memiliki ketahanan terhadap Common Rust. Banyak benih jagung modern telah dikembangkan untuk memiliki resistensi terhadap berbagai penyakit, termasuk Common Rust.",
        "2. Rotasi Tanaman: Lakukan rotasi tanaman dengan menanam tanaman selain jagung di lahan yang sama setiap tahun. Rotasi tanaman membantu mengurangi populasi patogen di tanah yang bisa menyebabkan infeksi pada musim berikutnya.",
        "3. Pemantauan Rutin: Pantau tanaman jagung secara rutin untuk mendeteksi gejala awal Common Rust. Identifikasi dan tangani infeksi sedini mungkin untuk mencegah penyebaran lebih lanjut.",
        "<a href='https://plantix.net/id/library/plant-diseases/100082/common-rust-of-maize/'style='font-size: 20px; text-decoration: none; font-weight: bold;'>Klik untuk informasi selanjutnya</a>🍃🍃🍃"
    ],
    'Gray Leaf Spot': [
        "Gray Leaf Spot disebabkan oleh jamur yang menginfeksi daun-daun tanaman jagung. Gejalanya berupa bercak-bercak abu-abu atau coklat kecil yang muncul di daun. Jika tidak dikendalikan, penyakit ini dapat menyebabkan penurunan hasil panen yang signifikan.",
        "Mengatasi Gray Leaf Spot (Bercak Daun Abu-abu) pada tanaman jagung melibatkan beberapa langkah pengendalian yang efektif. Berikut adalah beberapa langkah utama untuk menangani penyakit ini:",
        "1. Rotasi Tanaman: Lakukan rotasi tanaman dengan menanam tanaman selain jagung di lahan yang sama pada musim berikutnya untuk memutus siklus hidup patogen penyebab Gray Leaf Spot dan mengurangi inokulum patogen di tanah.",
        "2. Penggunaan Varietas Tahan Penyakit: Pilih dan tanam varietas jagung yang memiliki resistensi terhadap Gray Leaf Spot, sehingga varietas tahan penyakit memiliki daya tahan lebih baik terhadap infeksi dan mengurangi keparahan penyakit serta kerugian hasil panen.",
        "3. Sanitasi Lahan: Bersihkan sisa-sisa tanaman yang terinfeksi setelah panen dan musnahkan dengan cara dibakar atau dikubur untuk mengurangi sumber inokulum patogen di lapangan dan mengurangi kemungkinan infeksi di musim tanam berikutnya.",
        "<a href='https://plantix.net/id/library/plant-diseases/100107/grey-leaf-spot-of-maize/'style='font-size: 20px; text-decoration: none; font-weight: bold;'>Klik untuk informasi selanjutnya</a>🍃🍃🍃"
    ],
    'Sehat': [
        "Tanaman jagung yang berada dalam kondisi sehat tidak menunjukkan gejala penyakit apapun. Tanaman ini tumbuh dengan baik dan memiliki warna daun yang hijau. Perawatan yang baik, termasuk pemupukan dan pengairan yang tepat, membantu menjaga kesehatan tanaman. Pengamatan teratur dan tindakan pencegahan, seperti rotasi tanaman, pemilihan varietas tahan penyakit, sanitasi lahan, dan pengelolaan air yang baik, adalah kunci untuk menjaga tanaman tetap sehat dan mengurangi risiko infeksi penyakit seperti Gray Leaf Spot."
    ]
}

# Load Model
def load_model():
    custom_objects = {"Adam": tf.keras.optimizers.Adam}
    model = tf.keras.models.load_model('corn_disease_classification.h5', compile=False)
    # Konfigurasi ulang optimizer
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

# Fungsi untuk mengubah gambar ke base64
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.error(f"File not found: {image_path}")
        return ""

# Mendapatkan jalur direktori proyek
current_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(current_dir, "images")

# Mendapatkan jalur gambar
logo_left = os.path.join(image_dir, "logo_left.png")
logo_right = os.path.join(image_dir, "logo_right.png")

# Konversi gambar ke base64
logo_left_base64 = image_to_base64(logo_left)
logo_right_base64 = image_to_base64(logo_right)


# Fungsi untuk menentukan gaya caption berdasarkan tema
def get_caption_style():
    theme = st.get_option("theme.base")
    if theme == "dark":
        return "color: white;"
    elif theme == "light":
        return "color: black;"

caption_style = get_caption_style()

# Membuat HTML untuk ditampilkan
html_temp = f"""
    <style>
    .justified {{
        text-align: justify;
    }}
    .center {{
        text-align: center;
    }}
    
    .container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color:#228B22;
        padding: 5px;
        border-radius: 30px;
    }}
    </style>
    <div class="container">
        <img src="data:image/png;base64,{logo_left_base64}" style="width:100px; height:100px;">
        <h2 style="color:white; font-size:40px;">CVTech</h2>
        <img src="data:image/png;base64,{logo_right_base64}" style="width:100px; height:100px;">
    </div>
"""

# Tampilkan HTML di Streamlit
st.markdown(html_temp, unsafe_allow_html=True)



# SIDEBAR
# Fungsi untuk mendapatkan warna teks berdasarkan tema
def get_text_color():
    theme = st.get_option("theme.base")
    if theme == "dark":
        return "color: white;"
    elif theme == "light":
        return "color: black;"
    return "color: black;"  # Default

# Mengambil warna teks sesuai tema
text_color_style = get_text_color()

# Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Home", "Team", "Predict"],
        icons=["house", "people", "activity"],
        menu_icon="grid",
        default_index=0,
        styles={
            "container": {"padding": "5px 10px"},
            "icon": {"color": "text_color_style", "font-size": "15px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "color": "var(--nav-link-color, text_color_style)",
                "--hover-color": "rgba(0, 0, 0, 0)",
            },
            "nav-link-selected": {
                "background-color": "#228B22",
                "color": "white",  # Warna teks yang dipilih
            },
        }
    )
    
# Menampilkan text copyright
def display_copyright():
    theme = st.get_option("theme.base")
    if theme == "dark":
        text_color = "white"
    elif theme == "light":
        text_color = "black"

    st.sidebar.markdown(
        """
        <div style="position: fixed; bottom: 10px; left: 10px; text-align: center; font-size: 12px; color: {text_color};">
            Team 1 &copy;CVTech 2024
        </div>
        """,
        unsafe_allow_html=True
    )

# Display copyright text
display_copyright()

# Halaman Depan
if selected == "Home":
    st.markdown("""
    <h2 style="text-align: center; font-size: 32px;">Corn Vision Technology: Deteksi Dini, Panen Pasti!</h2>
    <div style="text-align: justify; font-size: 18px;">
    Pertanian merupakan tulang punggung bagi pemenuhan kebutuhan pangan global, namun petani sering kali dihadapkan pada tantangan yang kompleks. Tanaman jagung, salah satu tanaman pangan utama, tidak luput dari berbagai ancaman penyakit dan gangguan yang dapat mengancam hasil panen. Untuk mengatasi masalah ini, CVTech hadir sebagai solusi terdepan dengan menggunakan teknologi terkini dalam bidang Computer Vision.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Tentang Aplikasi
    st.markdown("""
    <h2 style="text-align: center; font-size: 32px;">Tentang Aplikasi</h2>
    <div style="text-align: center; font-size: 15px;">
    CVTech adalah aplikasi yang menggunakan teknologi Computer Vision untuk mendeteksi penyakit pada tanaman jagung. Dengan memanfaatkan model pembelajaran machine learning, aplikasi ini dapat mengenali berbagai penyakit jagung hanya dengan menganalisis gambar daun jagung.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .image-container {
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .caption {
        font-size: 13px; /* Ubah ukuran caption di sini */
        margin-top: 2px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Gambar dan Caption
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="image-container">
            <img src='data:image/png;base64,{image_to_base64("images/aman.png")}' alt='AMAN' width='150' style='display: block; margin: 5px auto;'>
            <div class="caption">Privasi dan perlindungan informasi Anda adalah prioritas utama.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="image-container">
            <img src='data:image/png;base64,{image_to_base64("images/akurat.png")}' alt='AKURAT' width='150' style='display: block; margin: 5px auto;'>
            <div class="caption">Akurasi yang tepat memungkinkan anda menemukan solusi dengan cepat.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="image-container">
            <img src='data:image/png;base64,{image_to_base64("images/mudah.png")}' alt='MUDAH' width='150' style='display: block; margin: 5px auto;'>
            <div class="caption">Dapat digunakan kapan saja dan dimana saja.
            </div>
        </div>
        """, unsafe_allow_html=True)



# Tim Pengembang
if selected == "Team":
    st.markdown("<h1 style='text-align: center;'>Anggota Tim</h1>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    .image-container {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
        padding-left: 0px;
        padding-right: 0px;
    }
    .profile-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-left: 10px;
        margin-right: 10px;
    }
    .profile-content img {
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .profile-links {
        display: flex;
        justify-content: center;
        gap: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Fungsi untuk menampilkan profil anggota
    def show_profile(image_path, name, univ, role, github_link, linkedin_link):
        st.markdown(f"""
        <div class="image-container">
            <div class="profile-content">
                <img src='data:image/png;base64,{image_to_base64(image_path)}' width='150'>
                <div><span style='font-size: 16px;'>{name}</span></div>
                <div>
                    <p style='margin: 0; font-size: 13px;'>{univ}</p>
                    <p style='margin: 0; font-size: 13px;'>{role}</p>
                </div>
                <div class="profile-links">
                    <a href="{github_link}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" width="25"></a> 
                    <a href="{linkedin_link}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" width="25"></a>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Menampilkan profile untuk setiap anggota tim
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_profile('images/photo_anggota1.png', 'Anggih Prasetio', 'Universitas Pendidikan Indonesia', 'Data Scientist', 'https://github.com/anggihprasss', 'https://www.linkedin.com/in/anggih-prasetio-a32218223/')
    with col2:
        show_profile('images/photo_anggota2.png', 'Asyifa Fauziyah', 'Institut Teknologi Garut', 'Data Engineer', 'https://github.com/Asyifauziyah990', 'https://www.linkedin.com/in/asyifa-fauziyah-7b93192b7/')
    with col3:
        show_profile('images/photo_anggota3.png', 'Muhammad Raihan', 'Institut Teknologi Garut', 'Full Stack Developer', 'https://github.com/Reyourbae', 'https://www.linkedin.com/in/muhammad-raihan-6680112a3/')
    with col4:
        show_profile('images/photo_anggota4.png', 'Nabila Fitriyani', 'Universitas Mayasari Bakti', 'Project Manager', 'https://github.com/nbilaft', 'https://www.linkedin.com/in/nabilafitriyani/')


# Sistem Deteksi
elif selected == "Predict":
    st.markdown("<h1 style='text-align: center;'>Sistem Prediksi Penyakit Jagung</h1>", unsafe_allow_html=True)
    file = st.file_uploader("",type=["jpg", "jpeg", "png"])
    st.set_option('deprecation.showfileUploaderEncoding', False)

    def import_and_predict(image_data, model):
        size = (256, 256)    
        image = ImageOps.fit(image_data, size)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      
        img_reshape = img[np.newaxis,...]    
        prediction = model.predict(img_reshape)
        return prediction

    if file is None:
        st.text("Silahkan Unggah Gambar")
    else:
        image = Image.open(file)
        st.image(image, width=300)
        if st.button("Prediksi"):
            predictions = import_and_predict(image, model)
            score = np.array(predictions[0])
            predicted_class = class_names[np.argmax(score)]
            prediction_score = np.max(score) * 100
            st.write("## {}".format(predicted_class))
            st.write("### Prediksi Score: {:.1f}%".format(prediction_score))
            st.title("Sistem memprediksi jagung dalam kondisi {}".format(predicted_class))
            st.write("Penjelasan:")
            explanation_html = "<div class='justified'>"
            for point in class_explanations[predicted_class]:
                explanation_html += "<p>{}</p>".format(point)
            explanation_html += "</div>"
            st.markdown(explanation_html, unsafe_allow_html=True)