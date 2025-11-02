import cv2
import os
import sqlite3
import numpy as np
import datetime
import tkinter as tk
from tkinter import Label, Button, simpledialog, messagebox
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Recognition with Database")
        self.window.geometry("850x750")
        self.window.configure(bg="#2c2f33")

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        # Paths and models
        self.face_folder = "face_database"
        os.makedirs(self.face_folder, exist_ok=True)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Database setup
        self.conn = sqlite3.connect("faces.db")
        self.create_table()

        # UI Components
        self.label = Label(self.window, bg="#23272a")
        self.label.pack(padx=10, pady=10)

        self.status_label = Label(self.window, text="Status: Idle", fg="white", bg="#2c2f33", font=("Arial", 14))
        self.status_label.pack(pady=5)

        self.face_count_label = Label(self.window, text=f"Stored Faces: {self.get_face_count()}",
                                      fg="white", bg="#2c2f33", font=("Arial", 14))
        self.face_count_label.pack(pady=5)

        Button(window, text="Start Recognition", command=self.start_recognition, width=20, bg="#7289da", fg="white", font=("Arial", 12)).pack(pady=5)
        Button(window, text="Add New Face", command=self.add_new_face, width=20, bg="#43b581", fg="white", font=("Arial", 12)).pack(pady=5)
        Button(window, text="Delete Stored Faces", command=self.delete_faces, width=20, bg="#ff5555", fg="white", font=("Arial", 12)).pack(pady=5)
        Button(window, text="Switch Camera", command=self.switch_camera, width=20, bg="#faa61a", fg="white", font=("Arial", 12)).pack(pady=5)
        Button(window, text="Stop", command=self.stop, width=20, bg="#99aab5", fg="black", font=("Arial", 12)).pack(pady=5)
        Button(window, text="Quit", command=self.quit_app, width=20, bg="#f04747", fg="white", font=("Arial", 12)).pack(pady=5)

        self.tracking = False
        self.current_camera = 0
        self.trained = False
        self.labels = {}

        # Try to load any saved model
        self.load_recognizer()

    # ---------------- DATABASE ----------------
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            date_added TEXT NOT NULL,
                            recognition_count INTEGER DEFAULT 0
                          )''')
        self.conn.commit()

    def add_to_database(self, name):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO faces (name, date_added, recognition_count) VALUES (?, ?, ?)",
                       (name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0))
        self.conn.commit()

    def update_recognition_count(self, name):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE faces SET recognition_count = recognition_count + 1 WHERE name = ?", (name,))
        self.conn.commit()

    def get_face_count(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM faces")
        return cursor.fetchone()[0]

    # ---------------- TRAINING ----------------
    def load_recognizer(self):
        if os.path.exists("trainer.yml"):
            self.recognizer.read("trainer.yml")
            self.trained = True
            self.load_labels()

    def load_labels(self):
        label_path = os.path.join(self.face_folder, "labels.txt")
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    name, id_ = line.strip().split(":")
                    self.labels[int(id_)] = name

    def save_labels(self):
        label_path = os.path.join(self.face_folder, "labels.txt")
        with open(label_path, "w") as f:
            for id_, name in self.labels.items():
                f.write(f"{name}:{id_}\n")

    def train_recognizer(self):
        face_samples = []
        ids = []
        label_files = [f for f in os.listdir(self.face_folder) if f.endswith('.jpg') or f.endswith('.png')]
        for i, file in enumerate(label_files):
            path = os.path.join(self.face_folder, file)
            img = Image.open(path).convert('L')
            img_np = np.array(img, 'uint8')
            id_ = i + 1
            face_samples.append(img_np)
            ids.append(id_)
            name = os.path.splitext(file)[0]
            self.labels[id_] = name

        if face_samples:
            self.recognizer.train(face_samples, np.array(ids))
            self.recognizer.save("trainer.yml")
            self.save_labels()
            self.trained = True

    # ---------------- ADD FACE ----------------
    def add_new_face(self):
        name = simpledialog.askstring("Add Face", "Enter user name:")
        if not name:
            messagebox.showinfo("Cancelled", "No name entered.")
            return

        self.add_to_database(name)
        count = 0
        while count < 50:
            ret, frame = self.vid.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                count += 1
                face_img = gray[y:y+h, x:x+w]
                filename = os.path.join(self.face_folder, f"{name}_{count}.png" or f"{name}_{count}.jpg")
                cv2.imwrite(filename, face_img)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Capturing {count}/50", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("Capturing Faces", frame)
                cv2.waitKey(100)
            if count >= 50:
                break

        cv2.destroyWindow("Capturing Faces")
        self.train_recognizer()
        self.face_count_label.config(text=f"Stored Faces: {self.get_face_count()}")
        messagebox.showinfo("Done", f"Face data saved for {name}")

    # ---------------- DELETE FACES ----------------
    def delete_faces(self):
        choice = messagebox.askquestion(
            "Delete Faces",
            "Do you want to delete a specific person?\n\nClick 'Yes' to delete one person or 'No' to delete all."
        )

        cursor = self.conn.cursor()

        if choice == "yes":
            cursor.execute("SELECT name FROM faces")
            names = [row[0] for row in cursor.fetchall()]

            if not names:
                messagebox.showinfo("No Records", "No faces found in the database.")
                return

            name = simpledialog.askstring("Delete Person", f"Enter the name to delete:\n\nAvailable: {', '.join(names)}")
            if not name:
                messagebox.showinfo("Cancelled", "No name entered.")
                return

            if name not in names:
                messagebox.showerror("Not Found", f"No record found for '{name}'.")
                return

            preview_path = None
            for file in os.listdir(self.face_folder):
                if file.startswith(name) and file.endswith(".png") or file.endswith(".jpg"):
                    preview_path = os.path.join(self.face_folder, file)
                    break

            if preview_path and os.path.exists(preview_path):
                preview_win = tk.Toplevel(self.window)
                preview_win.title(f"Confirm Delete: {name}")
                preview_win.configure(bg="#2c2f33")

                img = cv2.imread(preview_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img.thumbnail((250, 250))
                img_tk = ImageTk.PhotoImage(img)

                lbl = tk.Label(preview_win, text=f"Delete {name}?", font=("Arial", 14), bg="#2c2f33", fg="white")
                lbl.pack(pady=10)
                tk.Label(preview_win, image=img_tk, bg="#2c2f33").pack(pady=5)
                lbl_img = img_tk  # Prevent GC

                def confirm_delete():
                    cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
                    self.conn.commit()

                    deleted_count = 0
                    for file in os.listdir(self.face_folder):
                        if file.startswith(name) and file.endswith(".png") or file.endswith(".jpg"):
                            os.remove(os.path.join(self.face_folder, file))
                            deleted_count += 1

                    preview_win.destroy()
                    messagebox.showinfo("Deleted", f"Removed {deleted_count} image(s) and record for {name}.")
                    self.face_count_label.config(text=f"Stored Faces: {self.get_face_count()}")
                    self.status_label.config(text=f"Status: Deleted {name}")

                def cancel_delete():
                    preview_win.destroy()
                    messagebox.showinfo("Cancelled", "Deletion cancelled.")

                tk.Button(preview_win, text="Delete", command=confirm_delete, bg="#ff5555", fg="white", width=12).pack(pady=5)
                tk.Button(preview_win, text="Cancel", command=cancel_delete, width=12).pack(pady=5)

                preview_win.mainloop()

            else:
                messagebox.showwarning("No Image", f"No stored image found for {name}.")

        else:
            confirm = messagebox.askyesno("Confirm Delete", "Are you sure you want to delete ALL stored faces?")
            if confirm:
                cursor.execute("DELETE FROM faces")
                self.conn.commit()

                for file in os.listdir(self.face_folder):
                    if file.endswith(".png") or file.endswith(".jpg"):
                        os.remove(os.path.join(self.face_folder, file))

                if os.path.exists("trainer.yml"):
                    os.remove("trainer.yml")
                label_path = os.path.join(self.face_folder, "labels.txt")
                if os.path.exists(label_path):
                    os.remove(label_path)

                self.trained = False
                self.labels.clear()
                messagebox.showinfo("Deleted", "All stored faces and records have been removed.")

            self.face_count_label.config(text=f"Stored Faces: {self.get_face_count()}")
            self.status_label.config(text="Status: Faces updated")

    # ---------------- RECOGNITION ----------------
    def start_recognition(self):
        if not self.trained:
            messagebox.showwarning("Not Trained", "Please add a face first!")
            return
        self.tracking = True
        self.status_label.config(text="Status: Recognizing Faces")
        self.update_frame()

    def stop(self):
        self.tracking = False
        self.status_label.config(text="Status: Stopped")

    def switch_camera(self):
        self.tracking = False
        self.vid.release()
        self.current_camera = 1 - self.current_camera
        self.vid = cv2.VideoCapture(self.current_camera)
        messagebox.showinfo("Camera Switched", f"Switched to camera {self.current_camera}")
        self.start_recognition()

    def update_frame(self):
        if self.tracking:
            ret, frame = self.vid.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    id_, conf = self.recognizer.predict(gray[y:y+h, x:x+w])
                    if conf < 80:
                        name = self.labels.get(id_, "Unknown")
                        color = (0, 255, 0)
                        if name != "Unknown":
                            self.update_recognition_count(name)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)
            self.window.after(10, self.update_frame)

    def quit_app(self):
        self.tracking = False
        self.vid.release()
        self.conn.close()
        cv2.destroyAllWindows()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
