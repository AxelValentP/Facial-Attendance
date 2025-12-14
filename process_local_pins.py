import os
import shutil

# --- KONFIGURASI ---
# Nama folder hasil ekstrak manual Anda
SOURCE_FOLDER = "105_classes_pins_dataset" 
DEST_FOLDER = "research_dataset"

# Konfigurasi Sampel (Sama seperti sebelumnya)
MAX_PEOPLE = 20  
MIN_IMAGES = 15

def process_local_data():
    print(f"📂 Membaca data lokal dari: {SOURCE_FOLDER}")
    
    if not os.path.exists(SOURCE_FOLDER):
        print(f"❌ Error: Folder '{SOURCE_FOLDER}' tidak ditemukan!")
        print("Pastikan Anda sudah mengekstrak zip dari Kaggle ke dalam folder project ini.")
        return

    if not os.path.exists(DEST_FOLDER):
        os.makedirs(DEST_FOLDER)

    folders = os.listdir(SOURCE_FOLDER)
    count = 0

    print(f"🚀 Memproses pemindahan data ke '{DEST_FOLDER}'...")

    for folder_name in folders:
        # Path ke folder artis (misal: pins_Adriana Lima)
        src_path = os.path.join(SOURCE_FOLDER, folder_name)
        
        # Cek apakah ini folder dataset yang valid (awalan pins_)
        if os.path.isdir(src_path) and "pins_" in folder_name:
            images = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if len(images) >= MIN_IMAGES:
                # Bersihkan nama: 'pins_Adriana Lima' -> 'Adriana_Lima'
                clean_name = folder_name.replace("pins_", "").replace(" ", "_")
                
                # Tambahkan prefix PINS_
                final_name = f"PINS_{clean_name}"
                dest_path = os.path.join(DEST_FOLDER, final_name)

                # Hapus jika sudah ada (biar fresh)
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                
                # Copy folder
                shutil.copytree(src_path, dest_path)
                print(f"   [OK] {clean_name} ({len(images)} foto) -> Disalin.")
                
                count += 1
                if count >= MAX_PEOPLE:
                    print(f"⚠️ Batas {MAX_PEOPLE} orang tercapai. Berhenti.")
                    break

    print("-" * 50)
    print(f"🎉 SELESAI! {count} Selebriti dari data lokal telah dipindah ke '{DEST_FOLDER}'.")
    print("👉 Sekarang Anda bisa menjalankan 'flask run' dan masuk ke Tab Validasi.")

if __name__ == "__main__":
    process_local_data()