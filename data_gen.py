import cv2
import numpy as np
import os
import random

# =========================
# CONFIG
# =========================
REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"

os.makedirs(FAKE_DIR, exist_ok=True)

# =========================
# HELPER FUNCTIONS
# =========================

def generate_fake_number():
    return " ".join([str(random.randint(1000, 9999)) for _ in range(3)])


# =========================
# FRAUD ATTACK FUNCTIONS
# =========================

# 1. Aadhaar Number Tampering
def tamper_number(img):
    h, w, _ = img.shape

    # Assume number region (adjust if needed)
    x, y = int(w * 0.4), int(h * 0.75)

    fake_number = generate_fake_number()

    cv2.putText(
        img,
        fake_number,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2
    )
    return img


# 2. Font Mismatch
def font_attack(img):
    h, w, _ = img.shape
    x, y = int(w * 0.4), int(h * 0.75)

    fake_number = generate_fake_number()

    cv2.putText(
        img,
        fake_number,
        (x, y),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 0, 0),
        2
    )
    return img


# 3. Blur Attack
def blur_attack(img):
    return cv2.GaussianBlur(img, (21, 21), 0)


# 4. Noise Injection
def noise_attack(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)


# 5. Rotation Attack
def rotate_attack(img):
    h, w = img.shape[:2]
    angle = random.randint(-10, 10)

    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))


# 6. Copy-Paste Attack
def copy_paste_attack(img, donor_img):
    h, w, _ = img.shape

    x1, y1 = int(w * 0.2), int(h * 0.5)
    x2, y2 = int(w * 0.7), int(h * 0.65)

    patch = donor_img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = patch

    return img


# =========================
# MAIN GENERATOR
# =========================

def generate_fake_dataset():
    files = os.listdir(REAL_DIR)

    for i, file in enumerate(files):
        img_path = os.path.join(REAL_DIR, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        fake = img.copy()

        # Choose random attacks (1 to 3 attacks per image)
        attacks = [
            tamper_number,
            font_attack,
            blur_attack,
            noise_attack,
            rotate_attack
        ]

        num_attacks = random.randint(1, 3)

        for _ in range(num_attacks):
            attack = random.choice(attacks)
            fake = attack(fake)

        # Copy-paste attack (optional)
        if random.random() > 0.7:
            donor_file = random.choice(files)
            donor_img = cv2.imread(os.path.join(REAL_DIR, donor_file))

            if donor_img is not None:
                fake = copy_paste_attack(fake, donor_img)

        # Save fake image
        save_path = os.path.join(FAKE_DIR, f"fake_{i}.jpg")
        cv2.imwrite(save_path, fake)

        print(f"Generated: {save_path}")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    generate_fake_dataset()