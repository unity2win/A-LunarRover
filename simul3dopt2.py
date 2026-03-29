import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import heapq
import math
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed() 

IZGARA_BOYUTU = 500
BASLANGIC_POZ = (20, 20)
HEDEF_POZ = [475, 475] 
GUVENLIK_PAYI = 5 
TOPLAM_A_YILDIZ_ADIMLARI = 0 
TOPLAM_A_YILDIZ_SURESI = 0 

def karmasik_pikselli_dunya_olustur(boyut):
    izgara_2b = np.zeros((boyut, boyut), dtype=int)
    z_haritasi = np.zeros((boyut, boyut), dtype=np.float32)
    x, y = np.meshgrid(np.arange(boyut), np.arange(boyut))
    olcek_faktoru = (boyut / 60) ** 2 
    krater_sayisi = np.random.randint(70, 110) 
    for _ in range(krater_sayisi):
        mx, my = np.random.uniform(40, boyut-40), np.random.uniform(40, boyut-40)
        sapma = np.random.uniform(5.0, 15.0) 
        mesafe_kare = (x - mx)**2 + (y - my)**2
        maske = np.exp(-mesafe_kare / (2 * sapma**2)) > 0.4
        izgara_2b[maske] = 1
        z_haritasi[maske] = -5.0 * np.exp(-mesafe_kare[maske] / (2 * sapma**2))
    tumsek_sayisi = int(180 * olcek_faktoru * 0.1)
    for _ in range(tumsek_sayisi):
        tx, ty = np.random.uniform(0, boyut), np.random.uniform(0, boyut)
        sapma_t = np.random.uniform(3.0, 10.0); genlik = np.random.uniform(0.5, 2.5)
        mesafe_t = (x - tx)**2 + (y - ty)**2
        tumsek_degeri = genlik * np.exp(-mesafe_t / (2 * sapma_t**2))
        z_haritasi[izgara_2b == 0] += tumsek_degeri[izgara_2b == 0]
    izgara_2b[0:50, 0:50] = 0; z_haritasi[0:50, 0:50] = 0.1
    izgara_2b[boyut-50:boyut, boyut-50:boyut] = 0; z_haritasi[boyut-50:boyut, boyut-50:boyut] = 0.1
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
    
    adimlar, max_adim = 0, 40000
    
    en_yakin_dugum = (bas_d[0], bas_d[1], gecerli_yon[0], gecerli_yon[1])
    min_mesafe = bitise_mesafe
    
    while acik_liste and adimlar < max_adim:
        adimlar += 1
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
                dz = max(0, yukseklik_haritasi[ny, nx] - yukseklik_haritasi[cy, cx])
                yeni_g = g_skoru[cy, cx] + (1.414 if dx!=0 and dy!=0 else 1.0) + (dz * 9.0) + cp
                
                if yeni_g < g_skoru[ny, nx]:
                    g_skoru[ny, nx] = yeni_g
                    geldigi_yer[(nx, ny, dx, dy)] = (cx, cy, pdx, pdy)
                    f = yeni_g + (math.dist((nx, ny), hed_d) * sezgisel_agirlik)
                    heapq.heappush(acik_liste, (f, nx, ny, dx, dy))
                    
    hesaplama_suresi_ms = (time.time() - baslangic_zamani) * 1000
    TOPLAM_A_YILDIZ_ADIMLARI += adimlar
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

gercek_izgara, gercek_z_haritasi = karmasik_pikselli_dunya_olustur(IZGARA_BOYUTU)

arac_kesfedilen = np.zeros((IZGARA_BOYUTU, IZGARA_BOYUTU), dtype=bool)
arac_hafizasi = np.zeros((IZGARA_BOYUTU, IZGARA_BOYUTU), dtype=int)
arac_yukseklik_hafizasi = np.zeros((IZGARA_BOYUTU, IZGARA_BOYUTU), dtype=np.float32)
arac_poz = np.array([BASLANGIC_POZ[0]*1.0, BASLANGIC_POZ[1]*1.0], dtype=np.float64)
gecmis_yol = [tuple(arac_poz)]
gorev_tamamlandi = False

Z_MIN, Z_MAX = np.min(gercek_z_haritasi), np.max(gercek_z_haritasi)

cizici = pv.Plotter(title="3D", window_size=(2000, 1200))
x_a, y_a = np.meshgrid(np.arange(IZGARA_BOYUTU), np.arange(IZGARA_BOYUTU))
arazi_agi = pv.StructuredGrid(x_a.astype(np.float32), y_a.astype(np.float32), gercek_z_haritasi)
cizici.add_mesh(arazi_agi, cmap='terrain', lighting=True)

try:
    arac_m = pv.read(r"C:\Users\lolxk\Desktop\polarispy\24883_MER_static.obj")
    arac_m.rotate_x(90, inplace=True); arac_m.rotate_z(90, inplace=True); arac_m.scale([0.04, 0.04, 0.04], inplace=True)
    arac_aktoru = cizici.add_mesh(arac_m, color='silver')
except: arac_aktoru = cizici.add_mesh(pv.Box(bounds=(-1.25,1.25,-1,1,0,1.25)), color='red') 

hedef_aktoru = cizici.add_mesh(pv.Sphere(radius=10, center=(0,0,0)), color='gold')
hedef_aktoru.position = (HEDEF_POZ[0], HEDEF_POZ[1], gercek_z_haritasi[int(HEDEF_POZ[1]), int(HEDEF_POZ[0])])

plt.ion()

fig, (ax_radar, ax_enerji) = plt.subplots(1, 2, figsize=(20, 8)) 
fig.tight_layout()

ax_radar.axis('off')
ax_enerji.axis('off')

enerji_cizgisi, = ax_enerji.plot([], [], 'r-', linewidth=4) 
enerji_gecmisi = [0] * 100 

def tiklama_olayi(olay):
    global HEDEF_POZ, gecerli_plan, gorev_tamamlandi
    if olay.inaxes == ax_radar and olay.xdata is not None:
        HEDEF_POZ[0], HEDEF_POZ[1] = olay.xdata, olay.ydata
        hedef_aktoru.position = (HEDEF_POZ[0], HEDEF_POZ[1], gercek_z_haritasi[int(HEDEF_POZ[1]), int(HEDEF_POZ[0])])
        gecerli_plan, gorev_tamamlandi = [], False 

fig.canvas.mpl_connect('button_press_event', tiklama_olayi)

gecerli_plan = []
TAN_DEGERI = math.tan(math.radians(75))
cizici.show(interactive_update=True)

gecerli_tur = 1 
tur_agirliklari = {1: 1.0, 2: 2.0}

try:
    while cizici.render_window:
        if not gorev_tamamlandi:
            hedefe_mesafe = math.dist(arac_poz, HEDEF_POZ)
            
            if hedefe_mesafe > 3.0:
                aktif_isinlar = cevreyi_tara_vektorize(arac_poz, gercek_izgara, gercek_z_haritasi, arac_hafizasi, arac_yukseklik_hafizasi, arac_kesfedilen)
                plan_guvenli_mi = True
                if gecerli_plan:
                    kontrol_payi = GUVENLIK_PAYI if hedefe_mesafe > 20 else 1
                    for n in gecerli_plan[:15]:
                        y_min, y_max = max(0, int(n[1])-kontrol_payi), min(IZGARA_BOYUTU, int(n[1])+kontrol_payi+1)
                        x_min, x_max = max(0, int(n[0])-kontrol_payi), min(IZGARA_BOYUTU, int(n[0])+kontrol_payi+1)
                        if np.any(arac_hafizasi[y_min:y_max, x_min:x_max] == 1): plan_guvenli_mi = False; break
                else: plan_guvenli_mi = False
                
                if not plan_guvenli_mi:
                    gecerli_plan = a_yildiz_3b_farkindalikli(arac_hafizasi, arac_yukseklik_hafizasi, arac_poz, HEDEF_POZ, tur_agirliklari[gecerli_tur])
                
                if gecerli_plan:
                    siradaki_dugum = np.array(gecerli_plan.pop(0))
                    hedef_aci = np.degrees(np.arctan2(siradaki_dugum[1]-arac_poz[1], siradaki_dugum[0]-arac_poz[0]))
                    suanki_z = gercek_z_haritasi[int(siradaki_dugum[1]), int(siradaki_dugum[0])]
                    arac_aktoru.position = (siradaki_dugum[0], siradaki_dugum[1], suanki_z)
                    arac_aktoru.orientation = (0, 0, hedef_aci)
                    
                    cizici.camera.position = (siradaki_dugum[0]-105, siradaki_dugum[1]-105, suanki_z+135)
                    cizici.camera.focal_point = (siradaki_dugum[0], siradaki_dugum[1], suanki_z)
                    enerji_gecmisi.append(10 + max(0, (suanki_z - gercek_z_haritasi[int(arac_poz[1]), int(arac_poz[0])]) * 60))
                    arac_poz = siradaki_dugum
                    gecmis_yol.append(tuple(arac_poz))
                    
                    if len(gecmis_yol) % 5 == 0:
                        ax_radar.clear()
                        
                        ax_radar.axis('off') 
                        
                        ax_radar.set_facecolor('#1a1a1a')
                        
                        bilinen_z_haritasi = np.ma.masked_where(~arac_kesfedilen, arac_yukseklik_hafizasi)
                        ax_radar.imshow(bilinen_z_haritasi, origin='lower', cmap='YlOrBr', alpha=0.9, vmin=Z_MIN, vmax=Z_MAX)
                        
                        ax_radar.contour(arac_hafizasi, levels=[0.5], colors='red', linewidths=4, origin='lower') 
                        
                        for i_b, i_s in aktif_isinlar: ax_radar.plot([i_b[0], i_s[0]], [i_b[1], i_s[1]], color='yellow', linewidth=1.0, alpha=0.3) 
                        
                        h_noktalar = np.array(gecmis_yol[-800:])
                        
                        ax_radar.plot(h_noktalar[:,0], h_noktalar[:,1], 'g-', linewidth=4) 
                        
                        ax_radar.scatter(HEDEF_POZ[0], HEDEF_POZ[1], c='gold', marker='X', s=400) 
                        
                else: enerji_gecmisi.append(0)
                
            else:
                if gecerli_tur == 1:
                    
                    time.sleep(3)
                    
                    gecerli_tur = 2
                    arac_hafizasi.fill(0) 
                    arac_yukseklik_hafizasi.fill(0)
                    arac_kesfedilen.fill(False) 
                    arac_poz = np.array([BASLANGIC_POZ[0]*1.0, BASLANGIC_POZ[1]*1.0], dtype=np.float64) 
                    gecmis_yol = [tuple(arac_poz)]
                    enerji_gecmisi = [0] * 100 
                    gecerli_plan = []
                    
                    TOPLAM_A_YILDIZ_ADIMLARI = 0
                    TOPLAM_A_YILDIZ_SURESI = 0
                    
                else:
                    gorev_tamamlandi = True

        if len(enerji_gecmisi) > 100: enerji_gecmisi.pop(0)
        enerji_cizgisi.set_data(range(len(enerji_gecmisi)), enerji_gecmisi)
        ax_enerji.set_xlim(0, 100)
        ax_enerji.set_ylim(0, max(enerji_gecmisi) + 20 if len(enerji_gecmisi)>0 else 100)
        
        plt.pause(0.00001)
        cizici.update()
        
        if gorev_tamamlandi: 
            enerji_gecmisi.append(0)
            time.sleep(0.05)
            
except KeyboardInterrupt: pass
cizici.close()