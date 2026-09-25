# Mikrotremor — Alur Pengolahan MAM-SPAC dan HVSR untuk Profil Vs dan Vs30

Repositori ini berisi alur kerja berbasis Jupyter Notebook untuk mengolah data
**mikrotremor** menjadi **profil kecepatan gelombang geser (Vs) terhadap kedalaman**
dan **Vs30**, melalui dua jalur:

- **Jalur array (MAM – SPAC):** rekaman beberapa sensor serentak → kurva dispersi
  gelombang Rayleigh → inversi profil Vs (Notebook 01–05).
- **Jalur single station (HVSR):** satu kurva H/V dari file GEOPSY `.hv` → inversi
  profil Vs dengan Particle Swarm Optimization (Notebook 06).

Kode pendukung (pembaca data, SPAC, forward model, algoritma inversi) ada di paket
Python `mhvsr_vs30` di folder `src/`, dan diuji otomatis dengan `pytest`.

## Daftar Isi

- [Alur kerja](#alur-kerja)
- [Isi notebook](#isi-notebook)
- [Instalasi](#instalasi)
- [Cara menjalankan](#cara-menjalankan)
- [Struktur folder](#struktur-folder)
- [Dokumentasi](#dokumentasi)
- [Referensi utama](#referensi-utama)
- [Batasan ilmiah](#batasan-ilmiah)
- [Pengembangan dan pengujian](#pengembangan-dan-pengujian)
- [Kredit, sitasi, dan lisensi](#kredit-sitasi-dan-lisensi)

## Alur kerja

```
Jalur array (MAM):
  01 MiniSEED 3 komponen + koordinat ─► 02 SPAC & kurva dispersi ─► 03 inversi Vs
        │  (QC, geometri, HVSR awal)        (fit J0, picking)          (Differential Evolution)
        └──────────────────────────────────────────────────────────► 04 penyesuaian dengan HVSR
                                                                     ─► 05 pembanding Vs30 mHVSR

Jalur single station:
  file GEOPSY .hv ─► 06 inversi HVSR (model awal otomatis/manual + PSO) ─► profil Vs & Vs30 matematis
```

Setiap notebook MAM menyimpan hasil di `outputs/<SITE_ID>/<nomor>/`, dan notebook
berikutnya membacanya otomatis. Jalankan berurutan, dan ulangi dari notebook yang
inputnya berubah.

## Isi notebook

| Notebook | Isi | Input utama | Output utama |
|---|---|---|---|
| `01_data_preprocessing_qc.ipynb` | Inventaris & QC rekaman, geometri array, HVSR awal dari satu sensor | Folder MiniSEED (4 sensor × Z/N/E), koordinat sensor | `site_inputs.json`, QC waveform, `hvsr_observed.csv` |
| `02_mam_spac_dispersion.ipynb` | Koherensi SPAC per pasangan sensor, fit Bessel J₀, QC numerik, picking kurva dispersi (widget interaktif atau `MANUAL_ACCEPTED_PICKS`) | Kanal vertikal + output 01 | `dispersion_auto.csv`, `dispersion_final.csv`, `dispersion_qc.json` |
| `03_dispersion_inversion.ipynb` | Inversi kurva dispersi → profil Vs berlapis (forward Rayleigh `disba` + Differential Evolution, multi-seed) | Output 02, bounds ketebalan & Vs | Model terbaik, ensemble, grafik fit (`preview/` atau `final/`) |
| `04_hvsr_refinement_final_vs.ipynb` | Penyesuaian profil Vs terhadap bentuk HVSR sambil menjaga fit dispersi | Output 01 & 03 | Kandidat profil Vs akhir |
| `05_vs30_final_comparison.ipynb` | Pembanding Vs30 dari profil dengan prediksi model mHVSR | Output 01 & 04, `models/log_ANN_model.keras` | Prediksi & tabel perbandingan |
| `06_hvsr_single_site_pso_inversion.ipynb` | Inversi HVSR single station: forward body-wave S/P (Herak, 2008), Vp & ρ dari Brocher (2005), PSO multi-seed, ensemble model | File GEOPSY `.hv` | Profil Vs(z), ensemble, Vs30 matematis, `inversion_summary.json` |

Notebook tambahan:

- `low_dim_models.ipynb`, `high_dim_models.ipynb` — contoh prediksi Vs30 dari model
  mHVSR asli (lihat [Kredit](#kredit-sitasi-dan-lisensi)); memakai rekaman di `data/`.
- `Titik_1 (...).ipynb` — notebook eksplorasi; metode model awalnya sudah dipindahkan
  ke Notebook 06.

### Notebook 06: model awal otomatis atau manual

Di sel input Notebook 06, `INITIAL_MODEL_MODE` menentukan cara membentuk model awal
dan search space PSO:

- `"auto"` (bawaan): batas lapisan diambil dari puncak kontras kurva H/V, lalu frekuensi
  diubah ke kedalaman dengan profil empiris `Vs(z) = V0·(1+z)^x`. Setiap titik ukur
  mendapat jumlah lapisan dan bounds sendiri. Pengaturan: `POWER_LAW_V0_M_S`,
  `POWER_LAW_EXPONENT`, `AUTO_INITIAL_MODEL`.
- `"manual"`: bounds dari `THICKNESS_BOUNDS_M` dan `VS_BOUNDS_M_S`, model awal
  opsional dari `MANUAL_INITIAL_MODEL`.
- `BOUND_OVERRIDES` mengganti bounds lapisan tertentu pada kedua mode.

## Instalasi

Butuh **Python 3.12**. Cara yang disarankan memakai [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Kendromeda/Mikrotremor.git
cd Mikrotremor
uv sync                 # membuat .venv dari pyproject.toml + uv.lock
uv run jupyter lab      # buka notebook
```

Alternatif dengan `pip` (di virtual environment Python 3.12):

```bash
pip install -e .        # memasang paket mhvsr_vs30 dari folder src/ (mode editable)
pip install jupyterlab
```

Widget picking kurva dispersi di Notebook 02 membutuhkan `ipywidgets` dan `plotly`
yang aktif di JupyterLab. Jika klik di grafik tidak berfungsi, gunakan dropdown
frekuensi di widget atau isi `MANUAL_ACCEPTED_PICKS` di sel input Notebook 02.

## Cara menjalankan

1. Jalankan notebook dari **folder utama repositori**.
2. Isi **sel input pengguna** di bagian atas setiap notebook. Data dan parameter
   hanya diubah di sel itu, bukan di file keluaran.
   - **01:** `SITE_ID`, `MINISEED_DIRECTORY`, `COORDINATES_M`, `COORDINATE_CRS`,
     `ARRAY_GEOMETRY`, `HVSR_STATION`, pengaturan HVSR.
   - **02:** `SITE_ID` (sama dengan 01), parameter SPAC, pick yang sudah direview,
     status verifikasi (`GEOMETRY_VERIFIED`, dll.) beserta catatan buktinya.
   - **03:** bounds ketebalan & Vs, rasio Poisson, densitas, anggaran optimasi.
   - **06:** `HV_FILE`, band frekuensi inversi, mode model awal, Q, parameter PSO.
3. Jalankan dari sel pertama sampai akhir (**Restart Kernel & Run All** setelah mengubah
   input atau memperbarui kode di `src/`).
4. Notebook 03 berjalan dalam mode **preview** sampai pick dispersi sudah direview dan
   keempat status verifikasi di Notebook 02 bernilai `True`. Ini disengaja agar hasil
   yang belum diverifikasi tidak dianggap final.

Data rekaman lapangan tidak disimpan di repositori. Letakkan di folder lokal dan
arahkan path-nya di sel input.

## Struktur folder

```
Mikrotremor/
├── 01_…06_*.ipynb          # notebook alur kerja
├── src/mhvsr_vs30/         # paket Python
│   ├── mam/                #   SPAC, picking, inversi dispersi, geometri array
│   ├── hvsr_inversion.py   #   forward HVSR body-wave, PSO, ensemble, Vs30
│   ├── hvsr_initial_model.py #  model awal & bounds otomatis dari kurva H/V
│   ├── preprocessing/      #   windowing, smoothing, penolakan window, QC SESAME
│   ├── io/, manifest/      #   pembaca data mentah & manifest dataset
│   └── training/           #   model tabular/baseline
├── tests/                  # unit, integrasi, dan regresi (pytest)
├── configs/                # konfigurasi site, dataset, dan preprocessing
├── models/                 # bobot model mHVSR (dipakai Notebook 05 & notebook lama)
├── datasets/, datasets_global/  # katalog sumber data (checksum, lisensi, DOI)
├── artifacts/              # manifest & laporan hasil pipeline dataset
├── data/                   # contoh rekaman untuk notebook lama
└── docs/                   # dokumentasi & panduan
```

## Dokumentasi

- [`docs/Panduan_Alur_Inversi_Mikrotremor_MAM_HVSR.docx`](docs/Panduan_Alur_Inversi_Mikrotremor_MAM_HVSR.docx)
  — panduan belajar (Bahasa Indonesia): dasar teori, kurva dispersi, algoritma inversi
  (Differential Evolution & PSO), dan parameter yang perlu diubah di Notebook 01–03 dan 06.
- [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) — pipeline manifest dataset dan
  pembaca data mentah (`uv run mhvsr-vs30 …`).
- [`docs/MAM_SPAC_VS_PLAN.md`](docs/MAM_SPAC_VS_PLAN.md) — rencana dan batas metode MAM-SPAC.
- [`datasets/README.md`](datasets/README.md), [`datasets_global/README.md`](datasets_global/README.md)
  — daftar sumber data beserta lisensi dan DOI.

## Referensi utama

| Notebook | Referensi | DOI |
|---|---|---|
| 01–03 | Hayashi et al. (2022), *Microtremor array method using spatial autocorrelation analysis of Rayleigh-wave data*, J. Seismology | [10.1007/s10950-021-10051-y](https://doi.org/10.1007/s10950-021-10051-y) |
| 04 | Zor et al. (2010), Geophysical Journal International 182(3) | [10.1111/j.1365-246X.2010.04710.x](https://doi.org/10.1111/j.1365-246X.2010.04710.x) |
| 05 | Sharma Wagle et al. (2026), BSSA — model mHVSR → Vs30 | [10.1785/0120250128](https://doi.org/10.1785/0120250128) |
| 06 | Zaenudin et al. (2024), Earthquake Science — alur inversi HVSR dengan PSO | [10.1016/j.eqs.2024.04.004](https://doi.org/10.1016/j.eqs.2024.04.004) |
| 06 | Herak (2008), *ModelHVSR*, Computers & Geosciences — forward model H/V | [10.1016/j.cageo.2007.07.009](https://doi.org/10.1016/j.cageo.2007.07.009) |
| 06 | Brocher (2005), BSSA — relasi Vp dan densitas dari Vs | [10.1785/0120050017](https://doi.org/10.1785/0120050017) |

## Batasan ilmiah

- Inversi kurva HVSR saja bersifat **tidak unik**; kombinasi Vs dan ketebalan yang
  berbeda dapat menghasilkan kurva yang hampir sama. Gunakan informasi independen
  (bor, SPAC/MASW, geologi) untuk membatasi bounds.
- Kedalaman yang dapat diselidiki MAM dibatasi ukuran array. Sebagai pedoman umum,
  panjang gelombang andal ≈ 2–4× jarak sensor terjauh, dengan kedalaman ≈ λ/2.
- Sebaran ensemble adalah sebaran hasil pencarian optimizer, **bukan** interval
  ketidakpastian statistik.
- Vs30 yang dihitung dari model hanya bermakna bila data benar-benar sensitif sampai
  kedalaman 30 m.

## Pengembangan dan pengujian

```bash
uv run ruff check src tests                    # lint
uv run mypy src/mhvsr_vs30                     # pemeriksaan tipe
uv run pytest -m "not requires_real_data"      # test cepat tanpa dataset eksternal
```

CI GitHub Actions menjalankan ketiga pemeriksaan di atas beserta syarat coverage dan
`uv build` pada setiap pull request.

## Kredit, sitasi, dan lisensi

Repositori ini dikembangkan dari proyek **mHVSR-Vs30** oleh Kushal Sharma Wagle,
Joseph P. Vantassel, Adrian Rodriguez-Marek, dan B. Anbazhagan. Model prediksi Vs30
di folder `models/` dan notebook `low_dim_models.ipynb` / `high_dim_models.ipynb`
berasal dari proyek tersebut. Jika Anda memakai model atau alat tersebut, mohon sitasi:

> Sharma Wagle, K., Vantassel, J.P., Rodriguez-Marek, A., and Anbazhagan, B. (2026). "A Set of Data-Driven Models to Predict VS30 from the Horizontal-to-Vertical Spectral Ratio of Microtremors". Bulletin of the Seismological Society of America. [https://doi.org/10.1785/0120250128](https://doi.org/10.1785/0120250128)

Dirilis di bawah **GNU General Public License v3.0**, lihat [`LICENSE.txt`](LICENSE.txt).
