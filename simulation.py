import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import heapq
import math
import time
import warnings
from PIL import Image
import scipy.ndimage as ndimage
import cv2  

warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed() 

IZGARA_BOYUTU = 500
BASLANGIC_POZ = (20, 20)
HEDEF_POZ = [475, 475] 
GUVENLIK_PAYI = 5 
TOPLAM_A_YILDIZ_ADIMLARI = 0 
TOPLAM_A_YILDIZ_SURESI = 0 

GORSEL_Z_OLCEK = 3.0 

def goruntuden_dunya_olustur(gorsel_yolu, boyut):
    try:
        img_pil = Image.open(gorsel_yolu).convert('L')
    except FileNotFoundError:
        print(f"Hata: '{gorsel_yolu}' dosyası bulunamadı. Lütfen görselin kodla aynı klasörde olduğundan emin olun.")
        exit()
        
    img_pil = img_pil.resize((boyut, boyut), Image.Resampling.LANCZOS)
    img_arr = np.array(img_pil, dtype=np.float32) / 255.0 

    # 1. VARSAYILAN YÜKSEKLİK VE MİNİK PÜRÜZLER
    VARSAYILAN_Z = 10.0
    z_haritasi = np.full((boyut, boyut), VARSAYILAN_Z, dtype=np.float32)
    izgara_2b = np.zeros((boyut, boyut), dtype=int)
    
    puruz = np.random.randn(boyut, boyut) * 0.15
    z_haritasi += ndimage.gaussian_filter(puruz, sigma=1.0)

    # 2. TAM SİYAH ALANLARI KRATER OLARAK ALGILAMA (GERÇEKÇİ KASE FORMU)
    siyah_maske = img_arr < 0.1 
    dist_transform = ndimage.distance_transform_edt(siyah_maske)
    max_dist = np.max(dist_transform) if np.max(dist_transform) > 0 else 1
    
    norm_dist = dist_transform / max_dist
    derinlik_carpanı = 30.0 
    derinlik = (norm_dist ** 1.8) * derinlik_carpanı 
    z_haritasi -= derinlik

    # Kraterin etrafını (çeperini) tespit et ve engel yap
    siyah_maske_uint8 = siyah_maske.astype(np.uint8)
    ceper_maske = cv2.dilate(siyah_maske_uint8, np.ones((5,5), np.uint8)) - siyah_maske_uint8
    izgara_2b[ceper_maske == 1] = 1
    z_haritasi[ceper_maske == 1] += 0.4 

    # 3. YÜKSEK RENK DEĞİŞİMLERİNDEN MİNİK TEPECİKLER OLUŞTURMA
    gy, gx = np.gradient(img_arr)
    gradyan_buyuklugu = np.sqrt(gx**2 + gy**2)
    
    genisletilmis_siyah = cv2.dilate(siyah_maske_uint8, np.ones((9,9), np.uint8))
    gradyan_buyuklugu[genisletilmis_siyah == 1] = 0
    
    y_g, x_g = np.ogrid[0:boyut, 0:boyut]
    tepecik_engelleme_yaricapi = 60
    bas_mesafe_tepecik = (x_g - BASLANGIC_POZ[0])**2 + (y_g - BASLANGIC_POZ[1])**2
    hed_mesafe_tepecik = (x_g - HEDEF_POZ[0])**2 + (y_g - HEDEF_POZ[1])**2
    gradyan_buyuklugu[bas_mesafe_tepecik < tepecik_engelleme_yaricapi**2] = 0
    gradyan_buyuklugu[hed_mesafe_tepecik < tepecik_engelleme_yaricapi**2] = 0

    yuksek_gradyan_maskesi = gradyan_buyuklugu > 0.09 
    z_haritasi[yuksek_gradyan_maskesi] += 1.2 

    z_haritasi = ndimage.gaussian_filter(z_haritasi, sigma=1.2)

    # 4. GÜVENLİ BAŞLANGIÇ VE BİTİŞ ALANI 
    guvenli_yaricap = 45
    bas_mesafe = (x_g - BASLANGIC_POZ[0])**2 + (y_g - BASLANGIC_POZ[1])**2
    izgara_2b[bas_mesafe < guvenli_yaricap**2] = 0
    z_haritasi[bas_mesafe < (guvenli_yaricap*0.7)**2] = VARSAYILAN_Z 
    
    hed_mesafe = (x_g - HEDEF_POZ[0])**2 + (y_g - HEDEF_POZ[1])**2
    izgara_2b[hed_mesafe < guvenli_yaricap**2] = 0
    z_haritasi[hed_mesafe < (guvenli_yaricap*0.7)**2] = VARSAYILAN_Z 

    return izgara_2b, z_haritasi

def a_yildiz_3b_farkindalikli(bilinen_harita, yukseklik_haritasi, baslangic, hedef, sezgisel_agirlik, gecerli_yon=(0,0)):
    global TOPLAM_A_YILDIZ_ADIMLARI, TOPLAM_A_YILDIZ_SURESI
    baslangic_zamani = time.time()
    I_BOYUTU = IZGARA_BOYUTU
    bas_d, hed_d = (int(baslangic[0]), int(baslangic[1])), (int(hedef[0]), int(hedef[1]))
    bitise_mesafe = math.dist(bas_d, hed_d)
    dinamik_pay = GUVENLIK_PAYI if bitise_mesafe > 20 else 2
    
    dolgulu = np.pad(bilinen_harita, dinamik_pay, mode='constant', constant_values=1)
    genisletilmis_harita = np.zeros_like(bilinen_harita)
    for dy in range(-dinamik_pay, dinamik_pay + 1):
        for dx in range(-dinamik_pay, dinamik_pay + 1):
            genisletilmis_harita |= dolgulu[dinamik_pay + dy : dinamik_pay + dy + I_BOYUTU, 
                                  dinamik_pay + dx : dinamik_pay + dx + I_BOYUTU]

    g_skoru = np.full((I_BOYUTU, I_BOYUTU), np.inf, dtype=np.float32)
    g_skoru[bas_d[1], bas_d[0]] = 0
    acik_liste = [(0, bas_d[0], bas_d[1], gecerli_yon[0], gecerli_yon[1])]
    geldigi_yer = {}
    
    adimlar = 40000
    
    en_yakin_dugum = (bas_d[0], bas_d[1], gecerli_yon[0], gecerli_yon[1])
    min_mesafe = bitise_mesafe
    
    # --- YENİ FİZİKSEL EŞİKLER ---
    MAX_AŞILABILIR_EGIM = 1.0 
    
    while acik_liste and adimlar > 0:
        adimlar -= 1
        _, cx, cy, pdx, pdy = heapq.heappop(acik_liste)
        suanki_mesafe = math.dist((cx, cy), hed_d)
        if suanki_mesafe < min_mesafe: min_mesafe, en_yakin_dugum = suanki_mesafe, (cx, cy, pdx, pdy)
        if suanki_mesafe < 2.5: break
        
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < I_BOYUTU and 0 <= ny < I_BOYUTU:
                if genisletilmis_harita[ny, nx] == 1: 
                    continue
                
                cp = 0.0 if (dx == pdx and dy == pdy) else 0.2
                
                z1 = yukseklik_haritasi[cy, cx]
                z2 = yukseklik_haritasi[ny, nx]
                dz_mutlak = abs(z2 - z1) 
                
                if dz_mutlak > MAX_AŞILABILIR_EGIM:
                    continue
                
                egim_cezasi = (dz_mutlak ** 2) * 400.0
                
                yeni_g = g_skoru[cy, cx] + (1.414 if dx!=0 and dy!=0 else 1.0) + egim_cezasi + cp
                
                if yeni_g < g_skoru[ny, nx]:
                    g_skoru[ny, nx] = yeni_g
                    geldigi_yer[(nx, ny, dx, dy)] = (cx, cy, pdx, pdy)
                    f = yeni_g + (math.dist((nx, ny), hed_d) * sezgisel_agirlik)
                    heapq.heappush(acik_liste, (f, nx, ny, dx, dy))
                    
    hesaplama_suresi_ms = (time.time() - baslangic_zamani) * 1000
    TOPLAM_A_YILDIZ_ADIMLARI += (40000 - adimlar)
    TOPLAM_A_YILDIZ_SURESI += hesaplama_suresi_ms
    yol = []
    gecerli = en_yakin_dugum
    while gecerli in geldigi_yer:
        yol.append((gecerli[0], gecerli[1])); gecerli = geldigi_yer[gecerli]
    return yol[::-1] if yol else None

def cevreyi_tara_vektorize(poz, gercek_harita, gercek_yukseklikler, hafiza, yukseklik_hafizasi, kesfedilen, isin_sayisi=50, max_menzil=120):
    x, y = poz
    acilar = np.linspace(0, 2*np.pi, isin_sayisi, endpoint=False)
    mesafeler = np.arange(0, max_menzil, 1.2)
    isin_sonuclari = []
    for aci in acilar:
        dx, dy = math.cos(aci), math.sin(aci)
        cx_v, cy_v = (x + mesafeler * dx).astype(int), (y + mesafeler * dy).astype(int)
        gecerli_mi = (cx_v >= 0) & (cx_v < IZGARA_BOYUTU) & (cy_v >= 0) & (cy_v < IZGARA_BOYUTU)
        v_ix, v_iy = cx_v[gecerli_mi], cy_v[gecerli_mi]
        
        if len(v_ix) == 0: continue
            
        carpmalar = gercek_harita[v_iy, v_ix] == 1
        if np.any(carpmalar):
            carpma_indeksi = np.argmax(carpmalar)
            v_ix = v_ix[:carpma_indeksi+1]
            v_iy = v_iy[:carpma_indeksi+1]
            hafiza[v_iy[-1], v_ix[-1]] = 1 
            c_noktasi = (x + mesafeler[gecerli_mi][carpma_indeksi]*dx, y + mesafeler[gecerli_mi][carpma_indeksi]*dy)
        else:
            c_noktasi = (x + max_menzil*dx, y + max_menzil*dy)
            
        yukseklik_hafizasi[v_iy, v_ix] = gercek_yukseklikler[v_iy, v_ix]
        kesfedilen[v_iy, v_ix] = True 
        isin_sonuclari.append(((x, y), c_noktasi))
        
    return isin_sonuclari

# --- ANA ÇALIŞTIRMA ---

print("Dünya oluşturuluyor, lütfen bekleyin...")
gercek_izgara, gercek_z_haritasi = goruntuden_dunya_olustur("image.jpg", IZGARA_BOYUTU)
print("Dünya oluşturuldu.")

arac_kesfedilen = np.zeros((IZGARA_BOYUTU, IZGARA_BOYUTU), dtype=bool)
arac_hafizasi = np.zeros((IZGARA_BOYUTU, IZGARA_BOYUTU), dtype=int)
arac_yukseklik_hafizasi = np.zeros((IZGARA_BOYUTU, IZGARA_BOYUTU), dtype=np.float32)
arac_yukseklik_hafizasi.fill(10.0) 

arac_poz = np.array([BASLANGIC_POZ[0]*1.0, BASLANGIC_POZ[1]*1.0], dtype=np.float64)
gecmis_yol = [tuple(arac_poz)]
gorev_tamamlandi = False

Z_MIN, Z_MAX = np.min(gercek_z_haritasi), np.max(gercek_z_haritasi)

cizici = pv.Plotter(title="Sancak 26 - Otonom Rover Simülasyonu", window_size=(1600, 1000))
cizici.set_background('#111111')

x_a, y_a = np.meshgrid(np.arange(IZGARA_BOYUTU), np.arange(IZGARA_BOYUTU))

arazi_agi = pv.StructuredGrid(x_a.astype(np.float32), y_a.astype(np.float32), gercek_z_haritasi * GORSEL_Z_OLCEK)
cizici.add_mesh(arazi_agi, cmap='terrain', lighting=True, opacity=0.5, show_edges=False, name='gercek_arazi', clim=[Z_MIN * GORSEL_Z_OLCEK, Z_MAX * GORSEL_Z_OLCEK])

kesfedilen_arazi_agi = pv.StructuredGrid(x_a.astype(np.float32), y_a.astype(np.float32), arac_yukseklik_hafizasi * GORSEL_Z_OLCEK)
cizici.add_mesh(kesfedilen_arazi_agi, cmap='terrain', lighting=True, opacity=1.0, show_edges=False, name='kesfedilen_arazi', clim=[Z_MIN * GORSEL_Z_OLCEK, Z_MAX * GORSEL_Z_OLCEK])

try:
    arac_m = pv.read("C://Users//lolxk//Desktop//polarispy//24883_MER_static.obj")
    arac_m.rotate_x(90, inplace=True); arac_m.rotate_z(90, inplace=True); arac_m.scale([0.04, 0.04, 0.04], inplace=True)
    arac_aktoru = cizici.add_mesh(arac_m, color='silver', name='rover')
except:
    arac_aktoru = cizici.add_mesh(pv.Box(bounds=(-1.5,1.5,-1.2,1.2,0,1.5)), color='red', name='rover') 

hedef_hz = gercek_z_haritasi[int(HEDEF_POZ[1]), int(HEDEF_POZ[0])]
hedef_aktoru = cizici.add_mesh(pv.Sphere(radius=8, center=(HEDEF_POZ[0], HEDEF_POZ[1], (hedef_hz * GORSEL_Z_OLCEK) + 5)), color='gold', ambient=0.5, name='hedef')

plt.ion()
fig, (ax_radar, ax_enerji) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.5]}) 
fig.tight_layout(pad=3.0)
fig.canvas.manager.set_window_title('Rover Kontrol Paneli')

ax_radar.axis('off')
ax_radar.set_title("Lokal Radar Görünümü (Keşif & Yol)", color='white', fontsize=12)
ax_radar.set_facecolor('#1a1a1a')

ax_enerji.set_title("Tahmini Anlık Güç Tüketimi (dz maliyeti)", color='white', fontsize=12)
ax_enerji.set_facecolor('#1a1a1a')
ax_enerji.tick_params(axis='x', colors='white')
ax_enerji.tick_params(axis='y', colors='white')
ax_enerji.set_ylabel("Güç Birimi", color='white')
ax_enerji.set_xlabel("Son 100 Adım", color='white')
enerji_cizgisi, = ax_enerji.plot([], [], 'r-', linewidth=2) 
enerji_gecmisi = [0] * 100 

def tiklama_olayi(olay):
    global HEDEF_POZ, gecerli_plan, gorev_tamamlandi
    if olay.inaxes == ax_radar and olay.xdata is not None:
        yeni_x, yeni_y = olay.xdata, olay.ydata
        if 0 <= yeni_x < IZGARA_BOYUTU and 0 <= yeni_y < IZGARA_BOYUTU:
            HEDEF_POZ[0], HEDEF_POZ[1] = yeni_x, yeni_y
            h_hz = gercek_z_haritasi[int(HEDEF_POZ[1]), int(HEDEF_POZ[0])]
            hedef_aktoru.position = (HEDEF_POZ[0], HEDEF_POZ[1], (h_hz * GORSEL_Z_OLCEK) + 5)
            gecerli_plan, gorev_tamamlandi = [], False 

fig.canvas.mpl_connect('button_press_event', tiklama_olayi)

gecerli_plan = []
sim_hizi = 0.01 
son_radar_guncelleme = 0

print("Simülasyon başlıyor...")
cizici.show(interactive_update=True)

try:
    while cizici.render_window and plt.fignum_exists(fig.number):
        current_time_ms = time.time() * 1000
        
        if not gorev_tamamlandi:
            hedefe_mesafe = math.dist(arac_poz, HEDEF_POZ)
            
            aktif_isinlar = cevreyi_tara_vektorize(arac_poz, gercek_izgara, gercek_z_haritasi, arac_hafizasi, arac_yukseklik_hafizasi, arac_kesfedilen)
            
            cizici.add_mesh(pv.StructuredGrid(x_a.astype(np.float32), y_a.astype(np.float32), arac_yukseklik_hafizasi * GORSEL_Z_OLCEK), 
                             cmap='terrain', lighting=True, name='kesfedilen_arazi', show_edges=False, clim=[Z_MIN * GORSEL_Z_OLCEK, Z_MAX * GORSEL_Z_OLCEK])

            if hedefe_mesafe > 4.0:
                plan_guvenli_mi = True
                if gecerli_plan:
                    kontrol_menzili = 15
                    check_nodes = gecerli_plan[:kontrol_menzili]
                    d_pay = 2 
                    
                    for n in check_nodes:
                        nx_i, ny_i = int(n[0]), int(n[1])
                        if 0 <= nx_i < IZGARA_BOYUTU and 0 <= ny_i < IZGARA_BOYUTU:
                            y_min, y_max = max(0, ny_i-d_pay), min(IZGARA_BOYUTU, ny_i+d_pay+1)
                            x_min, x_max = max(0, nx_i-d_pay), min(IZGARA_BOYUTU, nx_i+d_pay+1)
                            if np.any(arac_hafizasi[y_min:y_max, x_min:x_max] == 1):
                                plan_guvenli_mi = False
                                break
                else:
                    plan_guvenli_mi = False

                if not plan_guvenli_mi:
                    gecerli_plan = a_yildiz_3b_farkindalikli(arac_hafizasi, arac_yukseklik_hafizasi, arac_poz, HEDEF_POZ, 1.5)
                
                if gecerli_plan:
                    siradaki_dugum = np.array(gecerli_plan.pop(0))
                    
                    eski_z = gercek_z_haritasi[int(arac_poz[1]), int(arac_poz[0])]
                    suanki_z = gercek_z_haritasi[int(siradaki_dugum[1]), int(siradaki_dugum[0])]
                    
                    hedef_aci = np.degrees(np.arctan2(siradaki_dugum[1]-arac_poz[1], siradaki_dugum[0]-arac_poz[0]))
                    
                    arac_aktoru.position = (siradaki_dugum[0], siradaki_dugum[1], (suanki_z * GORSEL_Z_OLCEK) + 0.5) 
                    arac_aktoru.orientation = (0, 0, hedef_aci)
                    
                    cam_pos = [siradaki_dugum[0] - 80, siradaki_dugum[1] - 80, (suanki_z * GORSEL_Z_OLCEK) + 120]
                    cam_focal = [siradaki_dugum[0], siradaki_dugum[1], (suanki_z * GORSEL_Z_OLCEK)]
                    
                    cizici.camera.position = cam_pos
                    cizici.camera.focal_point = cam_focal
                    
                    dz_mutlak = abs(suanki_z - eski_z)
                    enerji_maliyeti = 5 + (dz_mutlak ** 2) * 150.0  
                    enerji_gecmisi.append(enerji_maliyeti)
                    
                    arac_poz = siradaki_dugum
                    gecmis_yol.append(tuple(arac_poz))
                    
                else:
                    enerji_gecmisi.append(0)
                    time.sleep(0.1) 
                
            else:
                gorev_tamamlandi = True
                enerji_gecmisi.append(0)
                gecerli_plan = []

        if current_time_ms - son_radar_guncelleme > 100:
            son_radar_guncelleme = current_time_ms
            
            ax_radar.clear()
            ax_radar.axis('off')
            ax_radar.set_facecolor('#1a1a1a')
            ax_radar.set_title("Lokal Radar Görünümü (Keşif & Yol)", color='white', fontsize=12)
            
            bilinen_maske = arac_kesfedilen.astype(float)
            viz_z = arac_yukseklik_hafizasi * bilinen_maske
            # --- GÜNCELLEME: cmap='bone' yerine cmap='terrain' eklendi ---
            ax_radar.imshow(viz_z, origin='lower', cmap='terrain', alpha=0.8, vmin=Z_MIN, vmax=Z_MAX)
            
            ax_radar.contour(arac_hafizasi, levels=[0.5], colors='#ff3333', linewidths=2, origin='lower') 
            
            if not gorev_tamamlandi:
                for i_b, i_s in aktif_isinlar:
                    ax_radar.plot([i_b[0], i_s[0]], [i_b[1], i_s[1]], color='#00ff00', linewidth=0.5, alpha=0.2) 
            
            if len(gecmis_yol) > 2:
                h_noktalar = np.array(gecmis_yol)
                ax_radar.plot(h_noktalar[:,0], h_noktalar[:,1], color='#3399ff', linewidth=2, label='Geçmiş Yol') 
                
            if gecerli_plan and len(gecerli_plan) > 2:
                p_noktalar = np.array(gecerli_plan)
                ax_radar.plot(p_noktalar[:,0], p_noktalar[:,1], 'y--', linewidth=1.5, label='Aktif Plan')

            ax_radar.plot(arac_poz[0], arac_poz[1], 'rs', markersize=6, label='Rover')
            ax_radar.plot(HEDEF_POZ[0], HEDEF_POZ[1], 'gx', markersize=10, markeredgewidth=3, label='Hedef')
            
            if len(enerji_gecmisi) > 100: enerji_gecmisi.pop(0)
            enerji_cizgisi.set_data(range(len(enerji_gecmisi)), enerji_gecmisi)
            ax_enerji.set_xlim(0, 100)
            current_max_e = max(enerji_gecmisi) if enerji_gecmisi else 10
            ax_enerji.set_ylim(0, max(50, current_max_e + 10))
            
            plt.pause(0.001) 

        cizici.update()
        time.sleep(sim_hizi)
            
except KeyboardInterrupt:
    print("Simülasyon kullanıcı tarafından durduruldu.")

print("Simülasyon sonlandırıldı.")
cizici.close()
plt.close('all')
