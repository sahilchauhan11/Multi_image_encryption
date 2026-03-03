import numpy as np
import hashlib
from modules import hilbert
from modules import fractal
from modules import cicsml
from modules import image_utils


def derive_key_parts(user_key: str):
    key_bytes = hashlib.sha384(user_key.encode()).digest()
    Keys = []
    for i in range(12):
        seg = key_bytes[4 * i:4 * (i + 1)]
        Keys.append(int.from_bytes(seg, "big", signed=False))
    return Keys

def scale_chaos(D, L):
    D = np.asarray(D[:L], dtype=np.float64)
    D = np.mod(np.floor(D * 1e14), 256)
    return D.astype(np.uint8)
 
def generate_chaos_sequences(user_key, length):
    chaos = cicsml.generate_chaos_with_key(user_key, length=8 * length)

    if isinstance(chaos, tuple) and len(chaos) == 8:
        return chaos

    chaos = np.asarray(chaos).flatten()
    return np.split(chaos, 8)



def synchronized_disorder_diffusion_decrypt(CR_mat, CG_mat, CB_mat,
                                            IC1, IC2, Fmat,
                                            D1, D2, D3, D4, D5, D6, D7, D8):

    M, N = CR_mat.shape
    L = M * N

    # Flatten column-first
    CR = CR_mat.reshape(-1, order='F').astype(np.uint8)
    CG = CG_mat.reshape(-1, order='F').astype(np.uint8)
    CB = CB_mat.reshape(-1, order='F').astype(np.uint8)

    IC1 = np.asarray(IC1, dtype=np.int64)
    IC2 = np.asarray(IC2, dtype=np.int64)

    # Prepare chaotic sequences
    D1 = scale_chaos(D1, L)
    D2 = scale_chaos(D2, L)
    D3 = scale_chaos(D3, L)
    D4 = scale_chaos(D4, L)
    D5 = scale_chaos(D5, L)
    D6 = scale_chaos(D6, L)
    D7 = scale_chaos(D7, L)
    D8 = scale_chaos(D8, L)

    d7_const = D7[L - 1]
    d8_const = D8[L - 1]
    print("D1 unique (dec):", np.unique(D1))
    # ---------- REVERSE SECOND STAGE ----------
    TR_perm = np.zeros(L, dtype=np.uint8)
    TG_perm = np.zeros(L, dtype=np.uint8)
    TB_perm = np.zeros(L, dtype=np.uint8)

    # F matrix
    F = np.asarray(Fmat, dtype=np.int64)
    a, b = F[0]
    c, d = F[1]

    mod_val = max(L - 1, 1)
    idxs = np.arange(L, dtype=np.int64)
    k_all = (c * idxs + d) % mod_val

    for n in range(L):

        if n == 0:
            TR_perm[n] = CR[n] ^ d7_const ^ d8_const ^ D2[0]
            TG_perm[n] = CG[n] ^ d7_const ^ d8_const ^ D4[0]
            TB_perm[n] = CB[n] ^ d7_const ^ d8_const ^ D6[0]

        elif n == 1:
            TR_perm[n] = CR[n] ^ d7_const ^ CR[n - 1] ^ D2[0]
            TG_perm[n] = CG[n] ^ d7_const ^ CG[n - 1] ^ D4[0]
            TB_perm[n] = CB[n] ^ d7_const ^ CB[n - 1] ^ D6[0]

        else:
            k = k_all[n]
            TR_perm[n] = CR[n] ^ CR[n - 2] ^ CR[n - 1] ^ D2[k]
            TG_perm[n] = CG[n] ^ CG[n - 2] ^ CG[n - 1] ^ D4[k]
            TB_perm[n] = CB[n] ^ CB[n - 2] ^ CB[n - 1] ^ D6[k]

    # Undo IC2 permutation
    TR = np.zeros(L, dtype=np.uint8)
    TG = np.zeros(L, dtype=np.uint8)
    TB = np.zeros(L, dtype=np.uint8)

    for n in range(L):
        TR[IC2[n]] = TR_perm[n]
        TG[IC2[n]] = TG_perm[n]
        TB[IC2[n]] = TB_perm[n]

    # ---------- REVERSE FIRST STAGE ----------
    v1_perm = np.zeros(L, dtype=np.uint8)
    v2_perm = np.zeros(L, dtype=np.uint8)
    v3_perm = np.zeros(L, dtype=np.uint8)

    for n in range(L):

        if n == 0:
            v1_perm[n] = TR[n] ^ d7_const ^ d8_const ^ D1[0]
            v2_perm[n] = TG[n] ^ d7_const ^ d8_const ^ D3[0]
            v3_perm[n] = TB[n] ^ d7_const ^ d8_const ^ D5[0]

        elif n == 1:
            v1_perm[n] = TR[n] ^ d7_const ^ TR[n - 1] ^ D1[0]
            v2_perm[n] = TG[n] ^ d7_const ^ TG[n - 1] ^ D3[0]
            v3_perm[n] = TB[n] ^ d7_const ^ TB[n - 1] ^ D5[0]

        else:
            j = n - 1
            v1_perm[n] = TR[n] ^ TR[n - 2] ^ TR[n - 1] ^ D1[j]
            v2_perm[n] = TG[n] ^ TR[n - 2] ^ TG[n - 1] ^ D3[j]
            v3_perm[n] = TB[n] ^ TB[n - 2] ^ TB[n - 1] ^ D5[j]

    # Undo IC1 permutation
    v1 = np.zeros(L, dtype=np.uint8)
    v2 = np.zeros(L, dtype=np.uint8)
    v3 = np.zeros(L, dtype=np.uint8)

    for n in range(L):
        v1[IC1[n]] = v1_perm[n]
        v2[IC1[n]] = v2_perm[n]
        v3[IC1[n]] = v3_perm[n]

    # Reshape back
    I1 = v1.reshape(M, N, order='F')
    I2 = v2.reshape(M, N, order='F')
    I3 = v3.reshape(M, N, order='F')

    return I1, I2, I3



def decrypt_three_images(C, user_key: str, MAP1, MAP2, MAP3):

    H, W = C.shape[0], C.shape[1]

    Keys = derive_key_parts(user_key)
    FM, IC1, IC2, Fmat = fractal.build_fractal_matrix(H, W, Keys)
    D1, D2, D3, D4, D5, D6, D7, D8 = generate_chaos_sequences(user_key, H * W)

    CR = C[:, :, 0].astype(np.uint8)
    CG = C[:, :, 1].astype(np.uint8)
    CB = C[:, :, 2].astype(np.uint8)

    I1_idx, I2_idx, I3_idx = synchronized_disorder_diffusion_decrypt(
        CR, CG, CB,
        IC1, IC2, Fmat,
        D1, D2, D3, D4, D5, D6, D7, D8
    )

    P1_dec = image_utils.indexed_to_rgb(I1_idx, MAP1)
    P2_dec = image_utils.indexed_to_rgb(I2_idx, MAP2)
    P3_dec = image_utils.indexed_to_rgb(I3_idx, MAP3)

    return P1_dec, P2_dec, P3_dec
