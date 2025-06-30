import os
from freeqdsk import geqdsk
import matplotlib.pyplot as plt
import numpy as np
base_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(base_dir, "g026000.003900")

def flgetter(data):
    with open(filepath, "r") as g:
        gdata=geqdsk.read(g)

    # === 2. 격자 정보 및 psi 불러오기 ===
    psi = gdata.psi.T  # psi shape: (NZ, NR) → transpose to (NR, NZ)
    nr, nz = psi.shape
    rmin = gdata.rleft
    zmid = gdata.zmid
    rmax = rmin + gdata.rdim
    zmax = zmid + gdata.zdim

    # 격자 생성
    R = np.linspace(rmin, rmax, nr)
    Z = np.linspace(zmid - gdata.zdim / 2, zmax, nz)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')

    # === 3. psi로부터 B_R, B_Z 계산 (중앙 차분 사용) ===
    # 중앙차분법을 사용하여 도함수 계산
    dpsi_dR = np.zeros_like(psi)
    dpsi_dZ = np.zeros_like(psi)

    # 내부 포인트만 중앙차분 (가장자리는 전/후방차분 사용)
    dpsi_dR[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dR)
    dpsi_dZ[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * dZ)

    # 경계는 전방/후방차분
    dpsi_dR[0, :] = (psi[1, :] - psi[0, :]) / dR
    dpsi_dR[-1, :] = (psi[-1, :] - psi[-2, :]) / dR
    dpsi_dZ[:, 0] = (psi[:, 1] - psi[:, 0]) / dZ
    dpsi_dZ[:, -1] = (psi[:, -1] - psi[:, -2]) / dZ

    # === 4. 자기장 성분 계산 ===
    BR = -dpsi_dZ / RR
    BZ =  dpsi_dR / RR

    # B_phi 계산: F = R * B_phi, F는 ψ의 함수 (freeqdsk에서는 fpol로 제공됨)
    # → 보간을 위해 fpol(ψ)와 ψ grid를 매칭시켜야 함

    # 정상화된 psi
    psi_norm = (psi - gdata.simagx) / (gdata.sibdry - gdata.simagx)
    fpol = np.interp(psi_norm.flatten(), np.linspace(0, 1, len(gdata.fpol)), gdata.fpol)
    fpol = fpol.reshape(psi.shape)
    Bphi = fpol / RR
    return BR, Bphi, BZ, RR, ZZ