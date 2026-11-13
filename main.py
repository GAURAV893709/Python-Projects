import re
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
from datetime import datetime
from pathlib import Path

import cv2


class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1150x720")
        self.root.minsize(950, 600)
        self.root.configure(bg="#F3F6FA")

        # All registered face images will be stored inside this project folder.
        self.project_dir = Path(__file__).resolve().parent
        self.dataset_dir = self.project_dir / "dataset"
        self.dataset_dir.mkdir(exist_ok=True)
        self.attendance_dir = self.project_dir / "attendance"
        self.attendance_dir.mkdir(exist_ok=True)

        self.attendance_file = self.attendance_dir / "attendance.csv"

        cascade_path = Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))

        if self.face_cascade.empty():
            messagebox.showerror(
                "OpenCV Error",
                f"Could not load Haar Cascade file:\n{cascade_path}"
            )
            self.root.destroy()
            return

        self.create_styles()
        self.create_header()
        self.create_sidebar()
        self.create_dashboard()

    def create_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            rowheight=32,
            font=("Segoe UI", 10),
            background="white",
            fieldbackground="white"
        )
        style.configure(
            "Treeview.Heading",
            font=("Segoe UI", 10, "bold"),
            background="#E8EEF7",
            foreground="#17365D"
        )

    def create_header(self):
        header = tk.Frame(self.root, bg="#17365D", height=75)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="FACE RECOGNITION ATTENDANCE SYSTEM",
            font=("Segoe UI", 17, "bold"),
            bg="#17365D",
            fg="white"
        ).pack(side="left", padx=25, pady=20)

        self.clock = tk.Label(
            header, font=("Segoe UI", 10), bg="#17365D", fg="white"
        )
        self.clock.pack(side="right", padx=25)
        self.update_clock()

    def update_clock(self):
        now = datetime.now().strftime("%d-%m-%Y  |  %I:%M:%S %p")
        self.clock.config(text=now)
        self.root.after(1000, self.update_clock)

    def create_sidebar(self):
        sidebar = tk.Frame(self.root, bg="#102A43", width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="MAIN MENU", font=("Segoe UI", 11, "bold"),
            bg="#102A43", fg="#A9C3DD"
        ).pack(anchor="w", padx=20, pady=(30, 15))

        menu_items = [
            ("Dashboard", self.show_dashboard),
            ("Register Face", self.register_face),
            ("Train Model", self.train_model),
            ("Live Recognition", self.live_recognition),
            ("Attendance Records", self.show_attendance_records),
            ("Export Attendance", self.export_attendance),
        ]
        

        for name, command in menu_items:
            tk.Button(
                sidebar,
                text="   " + name,
                command=command,
                font=("Segoe UI", 10),
                bg="#102A43",
                fg="white",
                activebackground="#28527A",
                activeforeground="white",
                relief="flat",
                anchor="w",
                padx=12,
                pady=13,
                cursor="hand2"
            ).pack(fill="x", padx=8, pady=2)

        tk.Label(
            sidebar, text="BCA DATA SCIENCE PROJECT",
            font=("Segoe UI", 8, "bold"), bg="#102A43", fg="#A9C3DD"
        ).pack(side="bottom", pady=20)

    def create_dashboard(self):
        self.content = tk.Frame(self.root, bg="#F3F6FA")
        self.content.pack(side="left", fill="both", expand=True)
        self.show_dashboard()

    def clear_content(self):
        for widget in self.content.winfo_children():
            widget.destroy()

    def show_dashboard(self):
        self.clear_content()

        tk.Label(
            self.content, text="Dashboard", font=("Segoe UI", 24, "bold"),
            bg="#F3F6FA", fg="#17365D"
        ).pack(anchor="w", padx=30, pady=(30, 5))

        tk.Label(
            self.content,
            text="Welcome to your Face Recognition Attendance System",
            font=("Segoe UI", 11), bg="#F3F6FA", fg="#64748B"
        ).pack(anchor="w", padx=30, pady=(0, 25))

        cards = tk.Frame(self.content, bg="#F3F6FA")
        cards.pack(fill="x", padx=25)

        registered_count = sum(1 for p in self.dataset_dir.iterdir() if p.is_dir())
        card_data = [
            ("Registered Faces", str(registered_count)),
            ("Today's Attendance", "0"),
            ("Recognition Model", "Not Trained")
        ]

        for title, value in card_data:
            card = tk.Frame(
                cards, bg="white",
                highlightbackground="#DCE4EF", highlightthickness=1
            )
            card.pack(side="left", fill="both", expand=True, padx=7)

            tk.Label(
                card, text=title, font=("Segoe UI", 10),
                bg="white", fg="#64748B"
            ).pack(anchor="w", padx=15, pady=(18, 8))

            tk.Label(
                card, text=value, font=("Segoe UI", 17, "bold"),
                bg="white", fg="#17365D"
            ).pack(anchor="w", padx=15, pady=(0, 18))

        info = tk.Frame(self.content, bg="white")
        info.pack(fill="both", expand=True, padx=32, pady=30)

        tk.Label(
            info, text="System Overview", font=("Segoe UI", 15, "bold"),
            bg="white", fg="#17365D"
        ).pack(anchor="w", padx=20, pady=(20, 10))

        tk.Label(
            info,
            text=(
                "This application uses OpenCV, Haar Cascade and "
                "LBPH Face Recognition to identify registered users "
                "and manage attendance records.\n\n"
                "Getting started:\n"
                "1. Register a person's face.\n"
                "2. Train the face recognition model.\n"
                "3. Start live face recognition.\n"
                "4. View and export attendance records."
            ),
            font=("Segoe UI", 11), bg="white", fg="#475569",
            justify="left", wraplength=650
        ).pack(anchor="w", padx=20, pady=10)

    def register_face(self):
        form = tk.Toplevel(self.root)
        form.title("Register New Face")
        form.geometry("440x330")
        form.resizable(False, False)
        form.configure(bg="#F3F6FA")
        form.transient(self.root)
        form.grab_set()

        tk.Label(
            form, text="Register a Person",
            font=("Segoe UI", 19, "bold"),
            bg="#F3F6FA", fg="#17365D"
        ).pack(pady=(22, 8))

        tk.Label(
            form, text="Enter the person's details before starting the webcam.",
            font=("Segoe UI", 9), bg="#F3F6FA", fg="#64748B"
        ).pack(pady=(0, 18))

        fields = tk.Frame(form, bg="#F3F6FA")
        fields.pack(fill="x", padx=40)

        tk.Label(
            fields, text="Full Name", font=("Segoe UI", 10, "bold"),
            bg="#F3F6FA", fg="#17365D"
        ).pack(anchor="w")
        name_entry = ttk.Entry(fields, font=("Segoe UI", 11))
        name_entry.pack(fill="x", ipady=6, pady=(5, 13))

        tk.Label(
            fields, text="Student / Employee ID",
            font=("Segoe UI", 10, "bold"), bg="#F3F6FA", fg="#17365D"
        ).pack(anchor="w")
        id_entry = ttk.Entry(fields, font=("Segoe UI", 11))
        id_entry.pack(fill="x", ipady=6, pady=(5, 16))

        def start_registration():
            person_name = name_entry.get().strip()
            person_id = id_entry.get().strip()

            if not person_name or not person_id:
                messagebox.showwarning(
                    "Missing details", "Please enter both the person's name and ID.",
                    parent=form
                )
                return

            safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", person_id)
            if not safe_id:
                messagebox.showwarning("Invalid ID", "Please enter a valid ID.", parent=form)
                return

            person_dir = self.dataset_dir / safe_id
            if person_dir.exists() and any(person_dir.iterdir()):
                messagebox.showwarning(
                    "ID already registered",
                    "This ID already has face images. Use a different ID.",
                    parent=form
                )
                return

            form.destroy()
            self.capture_face_samples(person_name, person_id, person_dir)

        tk.Button(
            form, text="START FACE CAPTURE",
            command=start_registration,
            font=("Segoe UI", 10, "bold"),
            bg="#17365D", fg="white",
            activebackground="#28527A", activeforeground="white",
            relief="flat", cursor="hand2", padx=20, pady=11
        ).pack(pady=5)

    def capture_face_samples(self, person_name, person_id, person_dir):
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            messagebox.showerror(
                "Camera error",
                "Could not open your webcam. Close other apps using the camera and try again."
            )
            return

        person_dir.mkdir(parents=True, exist_ok=True)
        sample_count = 0
        frame_number = 0
        target_samples = 50
        window_name = "Face Registration - Press Q to stop"

        messagebox.showinfo(
            "Face capture started",
            "Look directly at the camera and slowly move your face slightly "
            "left, right, up and down.\n\n"
            "The system will save 50 face samples. Press Q in the camera window to stop."
        )

        try:
            while sample_count < target_samples:
                success, frame = camera.read()
                if not success:
                    messagebox.showerror("Camera error", "Unable to read a frame from the webcam.")
                    break

                frame_number += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
                )

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 200, 80), 2)

                    # Save every third detected frame to collect varied samples.
                    if frame_number % 3 == 0 and sample_count < target_samples:
                        face_image = gray[y:y + h, x:x + w]
                        face_image = cv2.resize(face_image, (200, 200))
                        sample_count += 1
                        image_path = person_dir / f"{person_id}_{sample_count:03d}.jpg"
                        cv2.imwrite(str(image_path), face_image)

                cv2.putText(
                    frame, f"Person: {person_name}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                cv2.putText(
                    frame, f"Samples: {sample_count}/{target_samples}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 100), 2
                )
                cv2.putText(
                    frame, "Move your face slowly | Press Q to stop", (15, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
                )

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()

        if sample_count == target_samples:
            messagebox.showinfo(
                "Registration successful",
                f"{person_name} (ID: {person_id}) has been registered successfully.\n"
                f"{sample_count} face images saved in:\n{person_dir}"
            )
        else:
            # Keep any useful samples captured; report the actual count.
            messagebox.showinfo(
                "Face capture stopped",
                f"Captured {sample_count} of {target_samples} samples.\n"
                "You can register this person again with a new ID, or remove the "
                "partial folder before retrying with the same ID."
            )

        self.show_dashboard()

    
    def train_model(self):
        import numpy as np
        import json

        dataset_path = self.dataset_dir
        trainer_path = self.project_dir / "trainer"
        trainer_path.mkdir(parents=True, exist_ok=True)

        faces = []
        labels = []
        label_map = {}

        if not dataset_path.exists():
            messagebox.showerror("Error", "Dataset folder does not exist.")
            return

        student_folders = [
            folder for folder in dataset_path.iterdir()
            if folder.is_dir()
        ]

        if not student_folders:
            messagebox.showwarning(
                "No students",
                "Please register a student before training."
            )
            return

        for label, folder in enumerate(student_folders):
            label_map[str(label)] = folder.name

            for image_file in folder.glob("*.jpg"):
                image = cv2.imread(
                    str(image_file),
                    cv2.IMREAD_GRAYSCALE
                )

                if image is None:
                    continue

                image = cv2.resize(image, (200, 200))
                faces.append(image)
                labels.append(label)

        if len(faces) == 0:
            messagebox.showerror(
                "Training Error",
                "No valid face images were found."
            )
            return

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(
                faces,
                np.array(labels, dtype=np.int32)
            )

            model_path = trainer_path / "face_model.yml"
            recognizer.write(str(model_path))

            labels_path = trainer_path / "labels.json"
            with open(labels_path, "w") as file:
                json.dump(label_map, file, indent=4)

            messagebox.showinfo(
                "Training Successful",
                f"Face recognition model trained successfully!\n\n"
                f"Registered students: {len(student_folders)}\n"
                f"Face images used: {len(faces)}\n\n"
                f"Model saved at:\n{model_path}"
            )

        except Exception as error:
            messagebox.showerror(
                "Training Error",
                f"Could not train the model:\n{error}"
            )
    
    def live_recognition(self):
        import json

        trainer_path = self.project_dir / "trainer"
        model_path = trainer_path / "face_model.yml"
        labels_path = trainer_path / "labels.json"

        # Check whether the model has been trained.
        if not model_path.exists() or not labels_path.exists():
            messagebox.showwarning(
                "Model Not Found",
                "Please train the face recognition model first."
            )
            return

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(str(model_path))

            with open(labels_path, "r") as file:
                label_map = json.load(file)

        except Exception as error:
            messagebox.showerror(
                "Model Error",
                f"Could not load the trained model:\n{error}"
            )
            return

        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not camera.isOpened():
            camera.release()
            messagebox.showerror(
                "Camera Error",
                "Could not open webcam. Close other camera applications and try again."
            )
            return

        messagebox.showinfo(
            "Live Recognition",
            "Live face recognition has started.\n\n"
            "Look at the camera. Press Q to stop."
        )

        confidence_threshold = 75

        try:
            while True:
                success, frame = camera.read()

                if not success:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(100, 100)
                )

                for (x, y, w, h) in faces:
                    face_image = gray[y:y+h, x:x+w]
                    face_image = cv2.resize(face_image, (200, 200))

                    label, confidence = recognizer.predict(face_image)

                    student_id = label_map.get(str(label), "Unknown")

                    if confidence < confidence_threshold:
                        display_text = f"ID: {student_id}"
                        box_color = (0, 200, 0)

                        self.mark_attendance(student_id)
                    else:
                        display_text = "Unknown Person"
                        box_color = (0, 0, 255)

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x+w, y+h),
                        box_color,
                        2
                    )

                    cv2.putText(
                        frame,
                        display_text,
                        (x, y-12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        box_color,
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Confidence: {confidence:.1f}",
                        (x, y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )

                cv2.putText(
                    frame,
                    "Press Q to exit",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Live Face Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()

    
    def mark_attendance(self, student_id):
        """Mark a student's attendance only once per day."""
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")

        file_exists = self.attendance_file.exists()

        # Check whether this student is already marked today.
        if file_exists:
            try:
                with open(self.attendance_file, "r", newline="") as file:
                    reader = csv.DictReader(file)

                    for row in reader:
                        if (
                            row.get("Student ID") == str(student_id)
                            and row.get("Date") == today
                        ):
                            return
            except Exception as error:
                print("Attendance check error:", error)
                return

        # Add a new attendance entry.
        try:
            with open(self.attendance_file, "a", newline="") as file:
                writer = csv.writer(file)

                if not file_exists or self.attendance_file.stat().st_size == 0:
                    writer.writerow(
                        ["Student ID", "Date", "Time", "Status"]
                    )

                writer.writerow(
                    [student_id, today, current_time, "Present"]
                )

            print(f"Attendance marked for {student_id}")

        except Exception as error:
            print("Could not save attendance:", error)

    def show_attendance_records(self):
        """Display saved attendance records in a table."""
        window = tk.Toplevel(self.root)
        window.title("Attendance Records")
        window.geometry("850x550")
        window.configure(bg="#F3F6FA")

        tk.Label(
            window,
            text="Attendance Records",
            font=("Segoe UI", 21, "bold"),
            bg="#F3F6FA",
            fg="#17365D"
        ).pack(anchor="w", padx=25, pady=20)

        table_frame = tk.Frame(window, bg="white")
        table_frame.pack(fill="both", expand=True, padx=25, pady=(0, 20))

        columns = ("Student ID", "Date", "Time", "Status")

        table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings"
        )

        for column in columns:
            table.heading(column, text=column)
            table.column(column, width=160, anchor="center")

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=table.yview
        )

        table.configure(yscrollcommand=scrollbar.set)

        table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        if self.attendance_file.exists():
            try:
                with open(self.attendance_file, "r", newline="") as file:
                    reader = csv.DictReader(file)

                    for row in reader:
                        table.insert(
                            "",
                            "end",
                            values=(
                                row.get("Student ID", ""),
                                row.get("Date", ""),
                                row.get("Time", ""),
                                row.get("Status", "")
                            )
                        )
            except Exception as error:
                messagebox.showerror(
                    "File Error",
                    f"Could not read attendance records:\n{error}",
                    parent=window
                )
        else:
            messagebox.showinfo(
                "No Records",
                "No attendance records are available yet.",
                parent=window
            )

    def export_attendance(self):
        """Export attendance records to a CSV file."""
        if (
            not self.attendance_file.exists()
            or self.attendance_file.stat().st_size == 0
        ):
            messagebox.showwarning(
                "No Records",
                "There are no attendance records to export."
            )
            return

        save_path = filedialog.asksaveasfilename(
            title="Export Attendance Records",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="Attendance_Records.csv"
        )

        if not save_path:
            return

        try:
            import shutil
            shutil.copy2(self.attendance_file, save_path)

            messagebox.showinfo(
                "Export Successful",
                f"Attendance records exported successfully!\n\n"
                f"Saved at:\n{save_path}"
            )

        except Exception as error:
            messagebox.showerror(
                "Export Error",
                f"Could not export attendance records:\n{error}"
            )


    def coming_soon(self):
        messagebox.showinfo(
            "Feature under development",
            "We will implement this feature in the next development steps."
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()

