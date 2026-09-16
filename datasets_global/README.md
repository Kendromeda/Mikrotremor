# Global dataset handoff - microtremor/HVSR and Vs30

**Status: 16 September 2026.** Folder ini melanjutkan pencarian Indonesia dengan corpus dari luar Indonesia. Semua artefak asli disimpan di `raw/`; hasil ekstraksi tidak boleh dipakai untuk menimpa file sumber.

## Ringkasan

- Fokus utama adalah data tiga-komponen ambient noise atau kurva HVSR yang dapat dipasangkan dengan Vs30/profil Vs.
- Manifest machine-readable untuk handoff tersedia di [`source_manifest.csv`](source_manifest.csv).
- Dataset global dipisahkan dari `datasets/` agar evaluasi domain Indonesia dapat ditahan sebagai external test.
- Prioritas supervised saat ini: **USGS ARRA 2013 (California)** dan **Wellington DesignSafe**.
- Pescara/Manfredonia berguna sebagai kurva HVSR tanpa label Vs30. ReMi Archive berguna sebagai profil/label tanpa rekaman HVSR.
- USGS California 2024 dan SISMI Milan menyediakan raw active/passive survey, tetapi hasil Vs30 perlu diperoleh atau dihitung dari analisis surface-wave sebelum menjadi label training. Christchurch menambah 14 lokasi profile-only.
- Jejak penyimpanan lokal saat verifikasi: **1.160 file / 6.008.564.685 byte** (sekitar 5,60 GiB), termasuk ZIP asli dan hasil ekstraksi sehingga bukan ukuran data unik.

## Dataset berhasil diunduh

### 1. Pescara dan Manfredonia, Italia

- Sumber: [University of Bologna AMS Acta](https://amsacta.unibo.it/id/eprint/7295/)
- DOI: `10.6092/unibo/amsacta/7295`
- Lisensi: CC BY-SA 4.0
- File:
  - `source_01_italy_pescara_manfredonia/raw/Pescara_HV.zip` - 2,346,916 byte; SHA256 `7A9EE4C48ABB9C4BE0E41BCC362A1B5C1BDA4B5CB164D2F5C8E8117A189E49CB`
  - `source_01_italy_pescara_manfredonia/raw/Manfredonia_HV.zip` - 3,837,225 byte; SHA256 `AD5F57E6566DAB4A8DAADE9842CF030CA82CC8BB51307EC9FDAD56C3F919AED5`
  - `source_01_italy_pescara_manfredonia/raw/README.rtf`
- Isi terverifikasi: 84 file `.asc`. Setiap file berisi `Frequency[Hz]`, `H/V`, dan `STD`.
- Status: **kurva HVSR processed, tanpa target Vs30 independen**. Gunakan untuk pretraining/representation learning atau pengujian parser, bukan supervised target.

### 2. USGS California SSCM/SWM 2024

- Sumber: [USGS ScienceBase](https://www.sciencebase.gov/catalog/item/65f10f88d34e1403329842b8)
- DOI: `10.5066/P134MR2M`
- Akses: public USGS data release
- File utama: `source_02_usgs_california_sscm/raw/2024.SSCM.data.zip` - 174,938,994 byte; MD5 `3CB84A89590E60ED2AFCBFA1596ECF6F` (cocok dengan metadata ScienceBase).
- File pendukung: `README.docx`, `SSCM_deployment_record_2024.xlsx`, `SWM_deployment_record_2024.xlsx`.
- Isi terverifikasi: 52 file `.sg2` dari survey active/passive. Workbook mencatat active-source MASW dan passive-source microtremor, geometri array, koordinat, tanggal, dan station ID untuk 27 strong-motion stations.
- Catatan penting: workbook yang tersedia adalah **deployment record**, bukan tabel hasil Vs30. Data release bertujuan mengukur Vs30, tetapi outcome/profil perlu diperoleh dari publikasi atau diproses terpisah. Jangan membuat label dari nama file.

### 3. SISMI Milan, Italia

- Sumber: [INGV SISMI dataset](https://shake.mi.ingv.it/sismi/SISMI_dataset.html)
- DOI: `10.13127/sismi/dataset`
- File: `source_03_italy_sismi_milan/raw/SISMI_dataset.zip` - 1,754,021,352 byte; SHA256 `84FDB19283F4B1B3B21D3331C303C05733437A33213A9C7ED9C39605C3520347`.
- Deskripsi sumber: 33 single-station ambient-noise measurements, microtremor arrays pada 3 lokasi, dan MASW pada 2 lokasi. Passive data diberikan dalam SAC, active data dalam format Geometrics `.dat`; koordinat WGS84 disertakan.
- Isi ekstraksi terverifikasi: **354 file / 2.450.367.339 byte** yang terdiri dari 279 `.sac`, 60 `.sac-hd`, 5 `.csv`, 5 `.xls`, 4 `.dat`, dan 1 `.txt`. Folder single-station memuat `M001`-`M033` serta `MI35`-`MI36`; array mencakup 18 posisi Giuriati, 14 Parco Nord, dan 14 Parco Vettabbia.
- Uji baca ObsPy berhasil untuk kedua ekstensi waveform. Sampel Parco Nord memiliki 2.460.970 sampel pada 200 Hz (sekitar 12.305 detik), sedangkan sampel Giuriati memiliki 4.324.000 sampel pada 200 Hz (sekitar 21.620 detik). Karena metadata aktual berbeda dari ringkasan umum sumber, parser wajib membaca header setiap trace dan tidak boleh mengunci sampling rate secara global.
- Status: **raw multi-method corpus**. Nilai Vs30/profil harus ditelusuri dari paper SISMI atau dihitung dari MASW/array. Jangan menganggap semua 33 single-station sites memiliki label colocated.

### 4. Wellington, New Zealand - DesignSafe PRJ-2075

- Sumber: [Dynamic Characterization of Wellington](https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-2075)
- DOI: `10.17603/DS24M6J`
- Unduhan merupakan subset terarah dari public API DesignSafe: seluruh raw MiniSEED di folder MAM, seluruh folder `Vs Profiles`, site/field PDFs, dan KMZ. Raw MASW besar tidak disalin karena profil hasil inversinya sudah tersedia dan target utama adalah input ambient-noise.
- Isi terverifikasi: **435 file / 560,615,634 byte**:
  - 212 MiniSEED (515,705,856 byte)
  - 187 text profiles
  - 34 PDF
  - 2 KMZ
- Contoh MiniSEED: 3 channel `BHN/BHZ/BHE`, 100 Hz, 30 menit per trace.
- Dataset mencakup sekitar 15 reference locations. Profil Vs menyertakan ensemble 1,000 model serta median profiles untuk beberapa layering ratio; site report menyajikan Vs30 dan uncertainty.
- Status: **prioritas supervised tinggi**, dengan catatan label Vs berasal dari inversi surface-wave MAM/MASW. Simpan sebagai `surface_wave_inversion`, bukan borehole ground truth. Split harus per lokasi, bukan per sensor/array/file.

### 5. USGS ARRA 2013 - subset tiga site California

- Sumber: [USGS Open-File Report 2013-1102](https://pubs.usgs.gov/of/2013/1102/)
- Corpus penuh: 191 stations dan sekitar 106 GB. Untuk tahap awal diunduh tiga site yang mewakili rentang Vs30, ditambah laporan utama.
- File:
  - `CI.DRE.zip` - 125,006,964 byte; SHA256 `16E3EBAEA33C0B4A94BC47A44CEE80BF02BD2B3DB46F9B3D1440526EFCADDF07`; Vs30 **196 m/s**.
  - `CI.CCC.zip` - 59,555,713 byte; SHA256 `4D575B35D6A2FC0F469A0183F1E7E41D2DE1EC462C7B8DECE7EB0E242826674D`; Vs30 **432 m/s**.
  - `CI.CLC.zip` - 117,099,904 byte; SHA256 `9661AD373288DE2DE6DBF72B6B8DC4B879A9D37199C5A5768DB6CBAE4BA042EF`; Vs30 **1,464 m/s**.
  - `of2013-1102_text.pdf` - laporan/metodologi dan Table 3 station labels.
- Isi terverifikasi: masing-masing site archive memiliki enam raw HVSR `.txt` tiga kolom, raw MASW/seismic `.dat`, koordinat spreadsheet, site report PDF, field forms, dan foto. Total subset: 18 raw HVSR text + 127 `.dat`. `FILE FORMATS.docx` mendefinisikan kolom sebagai **vertical, north, east**, dalam counts, dengan sampling rate **200 Hz**. Durasi rekaman yang diperiksa sekitar 1,451-3,600 detik.
- Status: **prioritas supervised tertinggi**. Input HVSR dan nilai Vs30 terdokumentasi dalam studi yang sama, dan Vs30 dikembangkan dari multi-technique surface-wave/refraction—not dari korelasi HVSR. Perlu mengonfirmasi urutan komponen dan sampling interval melalui `FILE FORMATS.docx` sebelum parsing.

### 6. Refraction Microtremor Vs(z) Profile Archive

- Sumber: [Zenodo record 3951865](https://zenodo.org/records/3951865) dan [public archive](https://sites.google.com/view/vs-profile-archive)
- DOI: `10.5281/zenodo.3951865`
- Lisensi: CC BY 4.0
- File:
  - `source_06_zenodo_vs_profile_archive/raw/Vs(z) Profile Archive.pdf` - 13,551,770 byte; MD5 `D2DC57994B2332255C3775F17EAF7E17`.
  - `source_06_zenodo_vs_profile_archive/raw/ReMi-Vs30.csv` - daftar **428 Vs30** beserta station, koordinat, dan nama profile file.
- Status: **label/profile-only**. Berguna untuk target distribution, spatial coverage, dan auxiliary supervision, tetapi tidak dapat dipasangkan dengan mHVSR tanpa waveform/station match tambahan.

### 7. Christchurch deep Vs profiles - DesignSafe PRJ-1295

- Sumber: [DesignSafe PRJ-1295](https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-1295)
- DOI: `10.17603/DS21D4D`
- Isi terverifikasi: **28 file / 20,937,523 byte**, yaitu 14 site report PDF dan 14 file ensemble `best1000_Vs.txt` untuk Burnside Park, Cashmere High School, Christchurch Park, Fitzgerald, Garrick Park, Groynes, Hagley Park, Ilam Fields, Latimere Square, Porritt Park, QEII Park, Redwood Park, Riccarton High School, dan South New Brighton Park.
- Status: **profile-only**. Project ini tidak memuat raw microtremor pada folder publik yang diperiksa, jadi belum bisa menjadi pasangan mHVSR-Vs30 tanpa sumber waveform lain.

## Sumber besar yang belum diunduh penuh

1. **USGS ARRA 2013 full repository** - 190 site archives, sekitar 106 GB. Strategi berikutnya adalah memilih tambahan site berdasarkan Table 3, khususnya metode `A` (array microtremor) dan rentang Vs30 yang belum seimbang.
2. **USGS 2024 SWM data1/data2** - sekitar 1.50 GB di luar paket SSCM; berisi conventional large-array surface-wave data. Unduh bila diperlukan untuk reproduksi Vs profiles.
3. **Christchurch raw recordings** - PRJ-1295 yang sudah diunduh hanya berisi 14 profiles/reports, sementara thesis model asli menyebut 25 sites dan 422 recordings dari studi Christchurch. Raw waveform perlu dicari pada project atau DOI lain.
4. **Italy ITACA / Felicetta et al. 2023** - metadata dan site characterization tersedia, tetapi pasangan raw ambient-noise yang digunakan dalam thesis belum ditemukan sebagai satu archive publik.
5. **Japan K-NET/KiK-net** - site metadata dan borehole profiles kuat, tetapi ambient-noise extraction memerlukan workflow FDSN/continuous data dan verifikasi izin/format.

## Aturan penggunaan untuk model

1. Tambahkan kolom `country`, `study_id`, `site_id`, `sensor_id`, `recording_id`, `label_method`, `label_independence`, `license`, dan `source_url`.
2. Untuk USGS ARRA, semua enam file HVSR dalam satu site harus berada pada split yang sama.
3. Untuk Wellington, seluruh sensor dan array dari satu reference location harus berada pada split yang sama.
4. Pescara/Manfredonia tidak boleh diberi label dari peta Vs30 global tanpa menandainya sebagai weak/pseudo-label.
5. Pisahkan evaluasi Indonesia dari training global. Gunakan Indonesia sebagai fine-tuning/holdout dan laporkan performa per negara/studi.
6. Jangan mencampur `measured/multi-method`, `surface_wave_inversion`, `HVSR_inversion`, dan `proxy_raster` dalam satu target tanpa provenance atau weighting.
7. Hitung checksum dan simpan manifest setelah normalisasi. Raw file tidak boleh diubah.

## Langkah lanjutan untuk AI berikutnya

1. Parse 18 raw HVSR USGS ARRA, baca `FILE FORMATS.docx`, lalu proses dengan parameter yang sama seperti `mhvsr-vs30`.
2. Hitung Vs30 dari Wellington median/ensemble profiles dan bandingkan antar layering ratio untuk mendapatkan label uncertainty.
3. Ekstrak daftar site, koordinat, sampling rate, durasi, dan target ke manifest tabular tunggal.
4. Tambah site USGS ARRA secara stratified sampai distribusi Vs30 lebih seimbang.
5. Audit lisensi/terms DesignSafe yang tidak ditampilkan eksplisit sebagai field lisensi pada project page; tetap pertahankan DOI dan citation.
