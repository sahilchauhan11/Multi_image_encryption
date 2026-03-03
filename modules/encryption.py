import numpy as np
import hashlib
from modules import hilbert
from modules import fractal
from modules import cicsml




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



def synchronized_disorder_diffusion(I1, I2, I3,
                                    IC1, IC2, Fmat,
                                    D1, D2, D3, D4, D5, D6, D7, D8):
    M, N = I1.shape
    L = M * N

    v1 = I1.reshape(-1, order='F').astype(np.uint8)
    v2 = I2.reshape(-1, order='F').astype(np.uint8)
    v3 = I3.reshape(-1, order='F').astype(np.uint8)

    IC1 = np.asarray(IC1, dtype=np.int64)
    IC2 = np.asarray(IC2, dtype=np.int64)

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

    print("D1 unique:", np.unique(D1))
    print("D2 unique:", np.unique(D2))
    print("D7 unique:", np.unique(D7))
    TR = np.zeros(L, dtype=np.uint8)
    TG = np.zeros(L, dtype=np.uint8)
    TB = np.zeros(L, dtype=np.uint8)
    
    for n in range(L):
        i = IC1[n]

        if n == 0:
            TR[n] = v1[i] ^ d7_const ^ d8_const ^ D1[0]
            TG[n] = v2[i] ^ d7_const ^ d8_const ^ D3[0]
            TB[n] = v3[i] ^ d7_const ^ d8_const ^ D5[0]

        elif n == 1:
            TR[n] = v1[i] ^ d7_const ^ TR[n - 1] ^ D1[0]
            TG[n] = v2[i] ^ d7_const ^ TG[n - 1] ^ D3[0]
            TB[n] = v3[i] ^ d7_const ^ TB[n - 1] ^ D5[0]

        else:
            j = n - 1        
            
            TR[n] = v1[i] ^ TR[n - 2] ^ TR[n - 1] ^ D1[j]
          
            TG[n] = v2[i] ^ TR[n - 2] ^ TG[n - 1] ^ D3[j]
          
            TB[n] = v3[i] ^ TB[n - 2] ^ TB[n - 1] ^ D5[j]

    F = np.asarray(Fmat, dtype=np.int64)
    a, b = F[0]
    c, d = F[1]

    mod_val = max(L - 1, 1)
    idxs = np.arange(L, dtype=np.int64)


    j_all = (a * idxs + b) % mod_val
    k_all = (c * idxs + d) % mod_val

    CR = np.zeros(L, dtype=np.uint8)
    CG = np.zeros(L, dtype=np.uint8)
    CB = np.zeros(L, dtype=np.uint8)

    for n in range(L):
        i = IC2[n]
        j = j_all[n]
        k = k_all[n]

        if n == 0:
            CR[n] = TR[i] ^ d7_const ^ d8_const ^ D2[0]
            CG[n] = TG[i] ^ d7_const ^ d8_const ^ D4[0]
            CB[n] = TB[i] ^ d7_const ^ d8_const ^ D6[0]

        elif n == 1:
            CR[n] = TR[i] ^ d7_const ^ CR[n - 1] ^ D2[0]
            CG[n] = TG[i] ^ d7_const ^ CG[n - 1] ^ D4[0]
            CB[n] = TB[i] ^ d7_const ^ CB[n - 1] ^ D6[0]

        else:
            CR[n] = TR[i] ^ CR[n - 2] ^ CR[n - 1] ^ D2[k]
            CG[n] = TG[i] ^ CG[n - 2] ^ CG[n - 1] ^ D4[k]
            CB[n] = TB[i] ^ CB[n - 2] ^ CB[n - 1] ^ D6[k]

    CR_mat = CR.reshape(M, N, order='F')
    CG_mat = CG.reshape(M, N, order='F')
    CB_mat = CB.reshape(M, N, order='F')
    
    
    return CR_mat, CG_mat, CB_mat


def encrypt_three_images(I1, I2, I3, user_key: str):
    H, W = I1.shape

    Keys = derive_key_parts(user_key)
    FM, IC1, IC2, Fmat = fractal.build_fractal_matrix(H, W, Keys)
    
    D1, D2, D3, D4, D5, D6, D7, D8 = generate_chaos_sequences(user_key, H * W)

    CR, CG, CB = synchronized_disorder_diffusion(
        I1, I2, I3, IC1, IC2, Fmat,
        D1, D2, D3, D4, D5, D6, D7, D8
    )
    C = np.stack([CR, CG, CB], axis=-1).astype(np.uint8)
    return C