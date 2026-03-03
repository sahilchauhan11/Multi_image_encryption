import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from modules import encryption
import time


# ===============================
# BASIC METRICS
# ===============================

def entropy(channel):
    hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())



def correlation(channel, direction="horizontal"):
    if direction == "horizontal":
        x = channel[:, :-1].flatten()
        y = channel[:, 1:].flatten()
    elif direction == "vertical":
        x = channel[:-1, :].flatten()
        y = channel[1:, :].flatten()
    else:  # diagonal
        x = channel[:-1, :-1].flatten()
        y = channel[1:, 1:].flatten()
    return float(pearsonr(x, y)[0])


def npcr(img1, img2):
    assert img1.shape == img2.shape
    diff = img1 != img2
    return diff.sum() * 100.0 / diff.size


def uaci(img1, img2):
    assert img1.shape == img2.shape
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    return diff.mean() * 100.0 / 255.0


def chi_square(channel):
    hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
    expected = channel.size / 256.0
    return float(((hist - expected) ** 2 / expected).sum())


def mse(img1, img2):
    return float(np.mean((img1.astype(float) - img2.astype(float)) ** 2))


def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / m))


# ===============================
# MAIN ANALYSIS FUNCTION
# ===============================

def run_full_analysis(
    I1_index, I2_index, I3_index,
    cipher_image,
    P1_quant, P2_quant, P3_quant,
    P1_dec, P2_dec, P3_dec,
    user_key
):
    print("\n==============================")
    print(" ENCRYPTION PERFORMANCE TEST ")
    print("==============================")

    # 1) ENTROPY ANALYSIS (cipher RGB)
    print("\n--- Information Entropy (Cipher RGB) ---")
    for i, ch_name in enumerate(["R", "G", "B"]):
        print(f"{ch_name} channel entropy:",
              entropy(cipher_image[:, :, i]))

    # 2) CORRELATION ANALYSIS (cipher RGB)
    print("\n--- Correlation (Cipher Image) ---")
    for i, ch_name in enumerate(["R", "G", "B"]):
        ch = cipher_image[:, :, i]
        print(f"\n{ch_name} Channel:")
        print("Horizontal:", correlation(ch, "horizontal"))
        print("Vertical:",   correlation(ch, "vertical"))
        print("Diagonal:",   correlation(ch, "diagonal"))

    # 3) KEY SENSITIVITY (NPCR / UACI between two ciphers)
    print("\n--- Key Sensitivity Test ---")
    user_key_modified = "Bskdkaka"

    cipher_mod = encryption.encrypt_three_images(
        I1_index, I2_index, I3_index, user_key_modified
    )

    for i, ch_name in enumerate(["R", "G", "B"]):
        print(f"\n{ch_name} Channel:")
        print("NPCR:", npcr(cipher_image[:, :, i], cipher_mod[:, :, i]))
        print("UACI:", uaci(cipher_image[:, :, i], cipher_mod[:, :, i]))

    # 4) CHI-SQUARE TEST (cipher RGB)
    print("\n--- Chi-Square Test (Cipher RGB) ---")
    for i, ch_name in enumerate(["R", "G", "B"]):
        chi = chi_square(cipher_image[:, :, i])
        print(f"{ch_name} channel χ²:", chi)

    # 5) ENCRYPTION QUALITY (Plain vs Cipher)
    print("\n--- Encryption Quality (Plain vs Cipher) ---")
    for plain, name in [(P1_quant, "Img1"),
                        (P2_quant, "Img2"),
                        (P3_quant, "Img3")]:
        print(f"\n{name}:")
        print("MSE:", mse(plain, cipher_image))
        print("PSNR:", psnr(plain, cipher_image))
        print("SSIM:", ssim(plain, cipher_image,
                             channel_axis=2, data_range=255))

    # 6) DECRYPTION QUALITY (Plain vs Decrypted)
    print("\n--- Decryption Quality ---")
    for name, orig, dec in [
        ("Img1", P1_quant, P1_dec),
        ("Img2", P2_quant, P2_dec),
        ("Img3", P3_quant, P3_dec),
    ]:
        print(f"\n{name}:")
        print("MSE:", mse(orig, dec))
        print("PSNR:", psnr(orig, dec))
        print("SSIM:", ssim(orig, dec,
                             channel_axis=2, data_range=255))

   # 7) HISTOGRAM PLOT (cipher RGB)
    print("\n--- Generating Histogram Plots ---")
    colors = ["r", "g", "b"]
    plt.figure(figsize=(12, 4))
    for i, ch_name in enumerate(["R", "G", "B"]):
        plt.subplot(1, 3, i + 1)
        plt.hist(cipher_image[:, :, i].flatten(),
                 bins=256, color=colors[i])
        plt.title(f"Cipher {ch_name}")
    plt.tight_layout()
    plt.savefig("cipher_histogram.png")
    plt.close()
    print("Histogram saved as cipher_histogram.png")

    # 8) EXECUTION TIME TEST
    print("\n--- Execution Time Test ---")
    start = time.time()
    _ = encryption.encrypt_three_images(
        I1_index, I2_index, I3_index, user_key
    )
    enc_time = time.time() - start
    print("Encryption Time:", enc_time, "seconds")

    print("\n==============================")
    print(" ANALYSIS COMPLETE ")
    print("==============================") 