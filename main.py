import numpy as np
import config
from modules import image_utils, encryption, decryption
from modules import analysis
def main():

    img1 = config.INPUT_PATH + "img1.png"
    img2 = config.INPUT_PATH + "img2.png"
    img3 = config.INPUT_PATH + "img3.png"

    P1_rgb, P2_rgb, P3_rgb = image_utils.prepare_images(img1, img2, img3)

    (I1_index, MAP1), (I2_index, MAP2), (I3_index, MAP3) = \
        image_utils.prepare_indexed_images(P1_rgb, P2_rgb, P3_rgb)

    user_key = "sssskksksa"

    cipher_image = encryption.encrypt_three_images(
        I1_index, I2_index, I3_index, user_key
    )
    
    image_utils.save_image(cipher_image, "final_cipher.png")
    print("[SUCCESS] Encryption completed.")

    P1_dec, P2_dec, P3_dec = decryption.decrypt_three_images(
        cipher_image, user_key, MAP1, MAP2, MAP3
    )
    print("Original R sum:", np.sum(P1_rgb[:, :, 0]))
    print("Decrypted R sum:", np.sum(P1_dec[:, :, 0]))

    image_utils.save_image(P1_dec, "dec1.png")
    image_utils.save_image(P2_dec, "dec2.png")
    image_utils.save_image(P3_dec, "dec3.png")
    print("[SUCCESS] Decryption completed.")
    
    P1_quantized = image_utils.indexed_to_rgb(I1_index, MAP1)
    P2_quantized = image_utils.indexed_to_rgb(I2_index, MAP2)
    P3_quantized = image_utils.indexed_to_rgb(I3_index, MAP3)

    if (np.array_equal(P1_quantized, P1_dec) and
        np.array_equal(P2_quantized, P2_dec) and
        np.array_equal(P3_quantized, P3_dec)):
        print("✅ PERFECT REVERSIBILITY CONFIRMED")
        
    analysis.run_full_analysis(
        I1_index, I2_index, I3_index,
        cipher_image,
        P1_quantized, P2_quantized, P3_quantized,
        P1_dec, P2_dec, P3_dec,
        user_key
        )
if __name__ == "__main__":
    main()