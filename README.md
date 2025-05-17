#🧬 Secure-transmission-of-confidential-image


This project implements a secure **image encryption and decryption technique** using a combination of **logistic map-based chaotic sequences** and **DNA computing**. It is designed in MATLAB and simulates a multi-layer security scheme to protect color images against unauthorized access.

---

## 📌 Overview

The encryption process uses:
- **Logistic map** to generate a chaotic sequence (used for diffusion and key generation)
- **DNA encoding and XOR operations** to secure pixel data
- **Confusion (shuffling)** and **diffusion** stages for robust transformation
- **Reversible logic** to accurately reconstruct the original image during decryption

The image goes through:
1. **DNA Encoding** using custom rules
2. **Chaotic key generation**
3. **Pixel-wise XOR using DNA rules**
4. **Confusion and Diffusion**
5. **Reverse operations** to decrypt
