# super_program.py
# branch alex

# Exemple de dataset-uri (doar comentarii - nu afecteaza codul):
# /home/alex23/.cache/kagglehub/datasets/omkargurav/face-mask-dataset/versions/1/data/without_mask
# /home/alex23/.cache/kagglehub/datasets/anku5hk/5-faces-dataset/versions/1/Five_Faces
# /home/alex23/.cache/kagglehub/datasets/freak2209/face-data/versions/1/Custom_Data/images/train

import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import Dataset, DataLoader

RESULTS_FILE = "results.txt"


# ============================================================
# 0. LOG SYSTEM
# ============================================================
def write_result(text: str):
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# ============================================================
# 1. INIT DEVICE + MODELS (best of both worlds)
# ============================================================
workers = 0 if os.name == "nt" else 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
write_result(f"Using device: {device}")

# MTCNN cu setarile mai robuste din program2
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=10,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device,
)

# FaceNet (InceptionResnetV1) din program1+2
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


# ============================================================
# 2. DATASET pentru persoane (best from program2, adaptat la structura din program1)
#    persons/
#       Alex/  (mai multe .jpg/.png/.jpeg)
#       Maria/
#       unknown_1/
# ============================================================
class PersonsDataset(Dataset):
    """
    Dataset care parcurge structura:
        root/
            person_name1/
                img1.jpg
                img2.png
            person_name2/
                ...
    si intoarce: (image_path, person_name)
    """

    def __init__(self, root_folder: str):
        self.samples = []
        self.root_folder = root_folder

        exts = (".jpg", ".jpeg", ".png")

        if not os.path.isdir(root_folder):
            os.makedirs(root_folder, exist_ok=True)

        for person_name in sorted(os.listdir(root_folder)):
            person_folder = os.path.join(root_folder, person_name)
            if not os.path.isdir(person_folder):
                continue

            for fname in sorted(os.listdir(person_folder)):
                if not fname.lower().endswith(exts):
                    continue
                img_path = os.path.join(person_folder, fname)
                self.samples.append((img_path, person_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, person_name = self.samples[idx]
        # NU deschidem imaginea aici ca sa folosim get_embedding pentru logging + detectie fata
        return img_path, person_name


def collate_fn_one(batch):
    """
    DataLoader by default intoarce lista de elemente.
    Noi vrem direct primul (pentru batch_size=1).
    batch = [(path, name)]
    => return (path, name)
    """
    return batch[0]


# ============================================================
# 3. EXTRACT EMBEDDING (din program1, usor tunat)
# ============================================================
def get_embedding(image_path: str):
    try:
        img = Image.open(image_path).convert("RGB")  # fixes RGBA PNG
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        write_result(f"Error opening {image_path}: {e}")
        return None

    print(f"\nLoaded: {image_path}, size={img.size}")
    write_result(f"\nLoaded: {image_path}, size={img.size}")

    face = mtcnn(img)

    if face is None:
        print("⚠ No face found.")
        write_result("⚠ No face found.")
        return None

    face = face.to(device)

    with torch.no_grad():
        emb = facenet(face.unsqueeze(0))

    return emb.squeeze(0).cpu()  # torch.Tensor (512,)


# ============================================================
# 4. BUILD DATABASE FROM persons/ folosind PersonsDataset + DataLoader
# ============================================================
def build_database(root: str = "persons"):
    """
    Creeaza un dicționar:
        db = {
          "Alex": tensor(N,512),
          "Maria": tensor(M,512),
          "unknown_1": tensor(K,512),
          ...
        }
    folosind dataset + dataloader (performant, ordonat).
    """
    db = {}

    if not os.path.isdir(root):
        print(f"⚠ Folder '{root}' not found, creating empty one.")
        write_result(f"⚠ Folder '{root}' not found, creating empty one.")
        os.makedirs(root, exist_ok=True)
        return db

    dataset = PersonsDataset(root)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn_one,
    )

    # colectăm embedding-urile într-un dict temporar: nume -> list[embedding]
    temp = {}

    for img_path, person_name in loader:
        # img_path: str
        # person_name: str
        print(f"\n🔵 Loading person image for: {person_name}")
        write_result(f"\nLoading person image for: {person_name}")

        emb = get_embedding(img_path)
        if emb is None:
            continue

        if person_name not in temp:
            temp[person_name] = []
        temp[person_name].append(emb)

    # convertim listele in tensor (N,512)
    for name, emb_list in temp.items():
        if len(emb_list) == 0:
            continue
        db[name] = torch.stack(emb_list, dim=0)
        print(f" → {name}: {len(emb_list)} embeddings saved.")
        write_result(f" → {name}: {len(emb_list)} embeddings saved.")

    print(f"\nDATABASE READY: {list(db.keys())}")
    write_result(f"DATABASE READY: {list(db.keys())}")
    return db


# ============================================================
# 5. FACE RECOGNITION (din program1)
# ============================================================
def recognize_face(embedding: torch.Tensor, database: dict):
    if not database:
        return None, float("inf")

    best_name = None
    best_dist = float("inf")

    for name, emb_list in database.items():
        # emb_list: (N,512)
        distances = torch.norm(emb_list - embedding, dim=1)
        dist = distances.min().item()

        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name, best_dist


# ============================================================
# 6. SAVE / MERGE UNKNOWN PERSON (din program1)
# ============================================================
def save_unknown_person(
    embedding: torch.Tensor,
    image_path: str,
    database: dict,
    threshold_unknown: float = 0.9,
):
    img = Image.open(image_path).convert("RGB")

    # matching doar cu unknown_* (cum era in program1, "unknown merge")
    best_person = None
    best_dist = float("inf")

    for name, embeds in database.items():
        distances = torch.norm(embeds - embedding, dim=1)
        dist = distances.min().item()

        if dist < best_dist:
            best_dist = dist
            best_person = name

    # 1) Daca seamana cu un unknown_* existent suficient de mult → merge
    if best_person is not None and best_person.startswith("unknown_") and best_dist < threshold_unknown:
        print(f"→ Same UNKNOWN person: {best_person} (dist={best_dist:.3f})")
        write_result(f"→ Same UNKNOWN person: {best_person} (dist={best_dist:.3f})")

        folder = os.path.join("persons", best_person)
        os.makedirs(folder, exist_ok=True)

        count = len(
            [
                f
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        img.save(os.path.join(folder, f"{count + 1}.jpg"))

        database[best_person] = torch.cat(
            [database[best_person], embedding.unsqueeze(0)], dim=0
        )

        return best_person

    # 2) Altfel → cream unknown_{n+1}
    existing_unknowns = [name for name in database if name.startswith("unknown_")]
    next_id = len(existing_unknowns) + 1
    new_name = f"unknown_{next_id}"

    folder = os.path.join("persons", new_name)
    os.makedirs(folder, exist_ok=True)

    img.save(os.path.join(folder, "1.jpg"))

    database[new_name] = embedding.unsqueeze(0)

    print(f"→ New UNKNOWN person created: {new_name}")
    write_result(f"→ New UNKNOWN person created: {new_name}")

    return new_name


# ============================================================
# 7. PROCESS FACES (din program1, dar rămâne simplu cu os.listdir)
# ============================================================
def process_faces(
    database: dict,
    folder: str = "faces",
    known_threshold: float = 1.0,
    unknown_merge_threshold: float = 0.9,
):
    print("\n🔍 Checking faces in:", folder)
    write_result("\nChecking faces...")

    if not os.path.isdir(folder):
        print(f"⚠ Folder '{folder}' not found.")
        write_result(f"⚠ Folder '{folder}' not found.")
        return

    for file in sorted(os.listdir(folder)):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
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

        # DB empty -> create unknown direct
        if name is None:
            print(f"+ UNKNOWN PERSON (no DB)")
            write_result(f"+ UNKNOWN PERSON (no DB)")
            save_unknown_person(emb, img_path, database, unknown_merge_threshold)
            continue

        # PERSON IDENTIFIED (cunoscută, nu unknown_) -> salvăm poza si embedding-ul
        if not name.startswith("unknown_") and dist < known_threshold:
            print(f"** PERSON IDENTIFIED: {name} (dist={dist:.3f})")
            write_result(f"** PERSON IDENTIFIED: {name} (dist={dist:.3f})")

            person_folder = os.path.join("persons", name)
            os.makedirs(person_folder, exist_ok=True)

            count = len(
                [
                    f
                    for f in os.listdir(person_folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )

            img = Image.open(img_path).convert("RGB")
            img.save(os.path.join(person_folder, f"{count + 1}.jpg"))

            database[name] = torch.cat(
                [database[name], emb.unsqueeze(0)], dim=0
            )

            continue

        # UNKNOWN PERSON
        print(f"+ UNKNOWN PERSON (best match={name}, dist={dist:.3f})")
        write_result(f"+ UNKNOWN PERSON (best match={name}, dist={dist:.3f})")

        save_unknown_person(emb, img_path, database, unknown_merge_threshold)


# ============================================================
# 8. MAIN
# ============================================================
if __name__ == "__main__":
    # resetam logul
    open(RESULTS_FILE, "w", encoding="utf-8").close()

    print("Building database from 'persons/' ...")
    write_result("Starting run...\n")

    db = build_database("persons")

    print("\nProcessing faces from 'faces/' ...")
    process_faces(db, "faces", known_threshold=1.0, unknown_merge_threshold=0.9)

    print("\n✅ Done. Check results.txt for full log.")
    write_result("\nDone.")
