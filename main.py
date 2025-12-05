#branch alex
#dataset /home/alex23/.cache/kagglehub/datasets/omkargurav/face-mask-dataset/versions/1/data/without_mask
#Path to dataset: /home/alex23/.cache/kagglehub/datasets/anku5hk/5-faces-dataset/versions/1/Five_Faces
#Path to dataset: /home/alex23/.cache/kagglehub/datasets/freak2209/face-data/versions/1/Custom_Data/images/train

import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np


# 1. Init models
mtcnn = MTCNN(image_size=160)
facenet = InceptionResnetV1(pretrained="vggface2").eval()



# Utility: write result to results.txt
def write_result(text):
    with open("results.txt", "a") as f:
        f.write(text + "\n")


# 2. Extract embedding from ONE image
def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")

    print(f"\nLoaded: {image_path}, size={img.size}")
    write_result(f"Loaded: {image_path}, size={img.size}")

    face = mtcnn(img)

    if face is None:
        print("⚠ No face found.")
        write_result("⚠ No face found.")
        return None

    with torch.no_grad():
        emb = facenet(face.unsqueeze(0))

    return emb.squeeze(0)


# 3. Build PERSON DATABASE
def build_database(root="persons"):
    db = {}

    for person_name in os.listdir(root):
        person_folder = os.path.join(root, person_name)
        if not os.path.isdir(person_folder):
            continue

        print(f"\n🔵 Loading person: {person_name}")
        write_result(f"\nLoading person: {person_name}")

        person_embeddings = []

        for file in os.listdir(person_folder):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_folder, file)
            emb = get_embedding(img_path)

            if emb is not None:
                person_embeddings.append(emb)

        if person_embeddings:
            db[person_name] = torch.stack(person_embeddings)
            print(f" → {person_name}: {len(person_embeddings)} embeddings saved.")
            write_result(f" → {person_name}: {len(person_embeddings)} embeddings saved.")

    print(f"\n DATABASE READY: {list(db.keys())}")
    write_result(f"DATABASE READY: {list(db.keys())}")
    return db


# 4. Compare face with all persons
def recognize_face(embedding, database):
    best_name = None
    best_distance = float("inf")

    for name, emb_list in database.items():
        distances = torch.norm(emb_list - embedding, dim=1)
        dist = torch.min(distances).item()

        if dist < best_distance:
            best_distance = dist
            best_name = name

    return best_name, best_distance


# 5. Save unknown person
def save_unknown_person(embedding, image_path, database):
    # Generate new name
    existing_unknowns = [p for p in database.keys() if p.startswith("unknown_")]
    next_id = len(existing_unknowns) + 1
    new_name = f"unknown_{next_id}"

    # Create folder
    new_folder = os.path.join("persons", new_name)
    os.makedirs(new_folder, exist_ok=True)

    # Copy the image for record
    img = Image.open(image_path)
    img.save(os.path.join(new_folder, "1.jpg"))

    # Add to database
    database[new_name] = embedding.unsqueeze(0)

    print(f"→ New person saved as {new_name}")
    write_result(f"→ New person saved as {new_name}")


# 6. Process faces folder with -, +, ** and auto-add new persons
def process_faces(database, folder="faces", threshold=1.0):
    print("\n🔍 Checking faces in:", folder)
    write_result("\nChecking faces...")

    for file in os.listdir(folder):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(folder, file)

        print("\n===============================")
        print(f"Processing: {file}")
        write_result(f"\nProcessing: {file}")

        emb = get_embedding(img_path)

        if emb is None:
            print("- NO PERSON DETECTED")
            write_result("- NO PERSON DETECTED")
            continue

        name, dist = recognize_face(emb, database)

        if dist >= threshold:
            print(f"+ UNKNOWN PERSON (dist={dist:.3f})")
            write_result(f"+ UNKNOWN PERSON (dist={dist:.3f})")

            save_unknown_person(emb, img_path, database)
        else:
            print(f"** PERSON IDENTIFIED: {name} (dist={dist:.3f})")
            write_result(f"** PERSON IDENTIFIED: {name} (dist={dist:.3f})")


# MAIN
if __name__ == "__main__":
    database = build_database("persons")
    process_faces(database, "faces", threshold=1.0)
