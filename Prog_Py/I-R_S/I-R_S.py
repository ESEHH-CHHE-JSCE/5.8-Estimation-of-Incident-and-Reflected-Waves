'''
I-R_S.py      入反射波分離推定プログラム 
                                                   Yoshihiko IDE 19 May., 2023
                  Created with reference to I-R_S.f90 made by Masaru YAMASHIRO
'''
import settings

# --- User setting ---
Dfile = settings.Dfile # データファイル名
n = settings.n # FFTデータ数
dt = settings.dt # サンプリング間隔(sec.)
NW = settings.NW # スムージングフィルター回数(Hanning Window)
DEP = settings.DEP # 水深(m)
DL = settings.DL # 波高計間隔(m)

# ---

import numpy as np
import pandas as pd

### function ###
# --- Cal Wave Number ---
def WAVENUMBER(nf, Ffreq, DEP):
    WNUM = []
    for i in range(nf):
        D = ((4. * np.pi ** 2 * DEP) / 9.8) * Ffreq[i] ** 2
        if D < 1.:
            XX = np.sqrt(D)/(1.-D/6.)
            WNUM.append(XX)
        else:
            XX = D*(D+1/(1+D*(.6522+D*(.4622+D**2*(.0864+.0675*D)))))
            WNUM.append(np.sqrt(XX))
    return np.array(WNUM)

def SMOOTH(WKORG, N):
    WK = np.zeros_like(WKORG)
    for J in range(2, N-2):
        WK[J] = 0.25*WKORG[J-1]+0.5*WKORG[J]+0.25*WKORG[J+1]
    for J in range(2, N-2):
        WKORG[J] = WK[J]
    return WKORG


# --- Cal Separation ---
FMIN, FMAX = 0.05, 0.45 # 分離推定有効周波数範囲
def SEPARATION(AK, BK, WNUM, nf, DEP, DDL, Ffreq, NW, tl):
    FAMPI = np.zeros(nf)
    FAMPR = np.zeros(nf)

    for j in range(0, nf):
        KW = WNUM[j] / DEP
        if KW != 0.:
            L = 2. * np.pi / KW
            FM = DDL / L
        else:
            FM = 0.        

        if FM < FMIN or FM > FMAX:
            FAMPI[j] = 0.; FAMPR[j] = 0.
        else:
            AK1,AK0,BK0,BK1 = AK[1][j],AK[0][j],BK[0][j],BK[1][j]
            AAI1 = (AK1 - AK0 * np.cos(KW * DDL) - BK0 * np.sin(KW * DDL)) ** 2
            AAI2 = (BK1 + AK0 * np.sin(KW * DDL) - BK0 * np.cos(KW * DDL)) ** 2
            AAR1 = (AK1 - AK0 * np.cos(KW * DDL) + BK0 * np.sin(KW * DDL)) ** 2
            AAR2 = (BK1 - AK0 * np.sin(KW * DDL) - BK0 * np.cos(KW * DDL)) ** 2
            FAMPI[j] = np.sqrt(AAI1 + AAI2) / (2. * abs(np.sin(KW * DDL)))
            FAMPR[j] = np.sqrt(AAR1 + AAR2) / (2. * abs(np.sin(KW * DDL)))

    # Power Spec. (入射波)
    WORK = 0.5 * tl * FAMPI ** 2
    for iw in range(NW): # 平滑化
        x = SMOOTH(WORK, nf)
    PspecI = x
    # Power Spec. (反射波)
    WORK = 0.5 * tl * FAMPR ** 2
    for iw in range(NW): # 平滑化
        x = SMOOTH(WORK, nf)
    PspecR = x
    
    return PspecI, PspecR


### main ###
# ---- Read time series data ------
df = pd.read_table(f'../Dat/{Dfile}', header=None, sep='\s+')
df.columns=['CH1', 'CH2']
df.head()
df = df.tail(n) # FFT用のデータは時系列の後ろから取る
print(df)
tl = n * dt # FFT データ長
nf = int(n/2) + 1 # FFT 周波数成分数

xr = df.iloc[:, :] - np.sum(df.iloc[:, :])/n
xi = np.zeros(n)

F = np.fft.fft(np.array(xr).T) # 高速フーリエ変換

Ffreq = np.array(range(nf)) / (n * dt) # 周波数(HZ)
AK = +2. * F.real[:, :nf] / n # フーリエ係数の格納（分離推定用）
BK = -2. * F.imag[:, :nf] / n # フーリエ係数の格納（分離推定用）

# --- 入反射分離 ---
WNUM = WAVENUMBER(nf, Ffreq, DEP)
PspecI, PspecR = SEPARATION(AK, BK, WNUM, nf, DEP, DL, Ffreq, NW, tl)

# --- 出力 ---
arr = np.column_stack([Ffreq, PspecI, PspecR])
df = pd.DataFrame(arr, columns=['Freq.', 'S(f)in', 'S(f)re'])
df = df.set_index('Freq.')
df.to_csv(f'../Out/Spec_{Dfile}.csv', float_format="%.5f")
