import cv2 as cv
import numpy as np
import pytesseract
import re
import pandas as pd
from itertools import zip_longest
import streamlit as st
from PIL import Image
import io
import sys
import os



if sys.platform.startswith('win'):
    # Windows (Senin Bilgisayarın)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # Linux (Streamlit Cloud Sunucusu)
    # Sunucuda tesseract path'e otomatik eklenir, ayar yapmaya gerek yoktur.
    # Ancak bazen garanti olsun diye şu komut gerekebilir:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

imgInstance = cv.imread('photos/yoklama.jpg'),
#cv.imshow('image',img)

#contour_img = imgInstance.copy()

#- otomasyon süreci


def PreProcessing(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur_gaus = cv.GaussianBlur(gray,(3,3),1)
    canny = cv.Canny(blur_gaus, 60,150)
    canny_thick = np.ones((5,5), np.uint8)
    dial = cv.dilate(canny, canny_thick, iterations=2)
    erode = cv.erode(dial, canny_thick, iterations=1)
    return erode

def getContours(img, imgContour):
    biggest = np.array([])
    maxArea = 0
#   ------------------>  şekilleri algılar, kapalı mı?...
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        #contour Area == 4 nokta birleştiğinde oluşan alanları hesaplar
        area = cv.contourArea(cnt)
        #----------------------------
        #gürültüyü azaltacağız
        if area > 6000:
            #------> noktalar arası uzaklık hesaplıyoruz
            peri = cv.arcLength(cnt, True)
            #-----------------
            approx = cv.approxPolyDP(cnt, 0.04*peri, True)
            print(f"Area --> {area}, Edge --> {len(approx)}")

            cv.drawContours(imgContour, cnt, -1, (0, 0, 255), 4)
            #hem alan büyük hem 4 köşeliyse bunu seçiyoruz
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

                cv.drawContours(imgContour, cnt, -1, (0, 255, 0), 4)

    return biggest

def preProcessing4Letters(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur_median = cv.medianBlur(gray, 3)
    thresh = cv.adaptiveThreshold(blur_median, 255,
                                  cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY,
                                  11, 11)
    return thresh


def reorder(myPoints):
    # Gelen veri: (4, 1, 2) formatında. Bunu (4, 2) yapalım.
    myPoints = myPoints.reshape((4, 2))

    # Yeni sıralanmış noktaları tutacak kutu
    myPointsNew = np.zeros((4, 1, 2), np.int32)

    # --- Toplama Yöntemi (Sol-Üst ve Sağ-Alt için) ---
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # En Küçük Toplam -> Sol Üst
    myPointsNew[3] = myPoints[np.argmax(add)]  # En Büyük Toplam -> Sağ Alt

    # --- Çıkarma Yöntemi (Sağ-Üst ve Sol-Alt için) ---
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # En Küçük Fark -> Sağ Üst
    myPointsNew[2] = myPoints[np.argmax(diff)]  # En Büyük Fark -> Sol Alt

    return myPointsNew

def getWarp(img, biggest):
    widthImg = 480
    heightImg = 640

    #noktaları dizdik
    biggest = reorder(biggest)

    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])

    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv.warpPerspective(img, matrix, (widthImg, heightImg))

    imgcrop = imgOutput[10:imgOutput.shape[0]-10, 10:imgOutput.shape[1]-10]
    imgCropresize = cv.resize(imgcrop, (widthImg, heightImg))

    return imgCropresize
#----------------------------------------------------








#----------------------------Streamlit Arayüzü

st.set_page_config(page_title="Image Scanner", page_icon="📄")

st.title("📄 Smart Photo Reader")
st.subheader("Convert your Attendace File to Excell list")

# Dosya Yükleme
uploaded_files = st.file_uploader(
    "Select your files",
    type=['jpg', 'png', 'jpeg'],
    accept_multiple_files = True
)
tum_veriler_havuzu = []




with st.expander("ℹ️How to use?"):
    st.markdown("""
    **Step 1:** The edges of the paper must be visible  
    **Step 2:** Only PNG, JPG, JPEG files are included  
    **Step 3:** Your image should be vertical ↕️
    """)


st.markdown(
    """
    <div style="
    background-color:#072b10;
    color:#05253b;
    padding:10px;
    border-radius:8px;
    font-weight:bold;
    ">
    </div>
    """,
    unsafe_allow_html=True
)


# --- 3. ANA MANTIK (MAIN) ---
if uploaded_files:  # Eğer dosya yüklendiyse

    # İlerleme çubuğu ve durum metni
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Hatalı dosyaları raporlamak için liste
    hatali_dosyalar = []

    for i, file in enumerate(uploaded_files):

        # Kullanıcıya bilgi ver
        status_text.text(f"İşleniyor: {file.name} ({i + 1}/{len(uploaded_files)})")
        progress_bar.progress((i + 1) / len(uploaded_files))

        # --- GÜVENLİK BLOĞU BAŞLANGICI ---
        try:
            # 1. Resmi Okuma (Burası hata yapmaya müsait)
            image_pil = Image.open(file)
            img = np.array(image_pil)

            # Resim RGB değilse (örn: siyah beyazsa) dönüştürme hatasını önle
            if len(img.shape) == 3:
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            elif len(img.shape) == 2:  # Zaten griyse
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

            # 2. Görüntü İşleme
            imgThreshold = PreProcessing(img)
            imgContours = img.copy()
            biggest = getContours(imgThreshold, imgContours)

            if biggest.size != 0:
                imgWarped = getWarp(img, biggest)
                imgFinal = preProcessing4Letters(imgWarped)

                # 3. OCR İşlemi
                ocr_output = pytesseract.image_to_string(imgFinal, lang='tur')

                # 4. Veri Ayıklama (Regex)
                lines = [line.strip() for line in ocr_output.split('\n') if line.strip()]

                gecici_isimler = []
                gecici_numaralar = []

                for line in lines:
                    if line.isdigit() and len(line) > 3:
                        gecici_numaralar.append(line)
                    else:
                        temiz_isim = re.sub(r'^\d+\s+', '', line)
                        # Sadece anlamlı uzunluktaki isimleri al
                        if len(temiz_isim) > 2:
                            gecici_isimler.append(temiz_isim)

                # Verileri eşleştir
                eslesmis_veri = list(zip_longest(gecici_isimler, gecici_numaralar, fillvalue='-'))

                # Havuza Ekle
                for isim, numara in eslesmis_veri:
                    tum_veriler_havuzu.append({
                        "Kaynak Dosya": file.name,
                        "Ad Soyad": isim,
                        "Okul Numarası": numara
                    })

            else:
                # Kağıt bulunamadıysa uyarı ver ama durma
                st.warning(f"⚠️ {file.name}: Kağıt çerçevesi algılanamadı.")
                hatali_dosyalar.append(f"{file.name} (Kağıt Bulunamadı)")

        except Exception as e:
            # --- HATA YAKALAMA ANI ---
            # Bir dosya bozuksa buraya düşer, program çökmez, diğer dosyaya geçer.
            st.error(f"❌ {file.name} dosyasında hata oluştu: {str(e)}")
            hatali_dosyalar.append(f"{file.name} (Teknik Hata: {str(e)})")
            continue  # Döngüye devam et (Sıradaki dosyaya geç)

        # --- GÜVENLİK BLOĞU BİTİŞİ ---

    # --- FİNAL İŞLEMLER ---
    progress_bar.empty()  # Çubuğu temizle
    status_text.text("Completed!🏆")

    # Eğer en az 1 satır veri okuyabildiysek Excel'i oluştur
    if tum_veriler_havuzu:

        st.success(f"✅ Total {len(uploaded_files)} files have been scanned .")
        if hatali_dosyalar:
            with st.expander("Click to view the false files"):
                for hata in hatali_dosyalar:
                    st.write(hata)

        # DataFrame Oluştur
        df = pd.DataFrame(tum_veriler_havuzu)

        # Önizleme
        st.dataframe(df)

        # Excel Oluşturma (RAM'de)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Toplu Liste')

            # STİL KODLARIN
            workbook = writer.book
            worksheet = writer.sheets['Toplu Liste']

            header_format = workbook.add_format({
                'bold': True, 'text_wrap': True, 'valign': 'vcenter',
                'align': 'center', 'fg_color': '#D9E1F2', 'border': 1
            })

            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            worksheet.set_column('A:A', 25)  # Kaynak Dosya
            worksheet.set_column('B:B', 30)  # Ad Soyad
            worksheet.set_column('C:C', 20)  # Numara

        buffer.seek(0)

        st.download_button(
            label="📥 Birleştirilmiş Excel'i İndir",
            data=buffer,
            file_name="toplu_yoklama_listesi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error("Üzgünüm, yüklenen hiçbir dosyadan anlamlı veri çıkarılamadı.")


total_size = sum(file.size for file in uploaded_files) / 1024
st.metric("📦 Total Size: (MB)",
          f"{total_size / 1000:.2f}"
          )



st.markdown(
    """
    <div style="
    background-color:#072b10;
    color:#05253b;
    padding:10px;
    border-radius:8px;
    font-weight:bold;
    ">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='color:#0b2b07;'>Made by Tolga Bekiroğlu</h3>",
    unsafe_allow_html=True
)










