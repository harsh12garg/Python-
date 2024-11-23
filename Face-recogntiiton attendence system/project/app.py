import os
import cv2
import numpy as np
from datetime import date, datetime
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Flask App Initialization
app = Flask(__name__)
app.secret_key = "your_secret_key"  # For session management

# Constants
nimgs = 10
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
attendance_file = f"Attendance/Attendance-{datetoday}.csv"
model_path = "static/face_recognition_model.pkl"

# Directories Setup
os.makedirs("Attendance", exist_ok=True)
os.makedirs("static/faces", exist_ok=True)

# Create Attendance CSV if not exists
if not os.path.isfile(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Roll,Time\n")

# Load Haarcascade for face detection
cascade_path = "project/haarcascade_frontalface_default.xml"  # Adjust path here
face_detector = cv2.CascadeClassifier(cascade_path)

# Check if the cascade is loaded properly
if face_detector.empty():
    print(
        "[ERROR] Cascade file could not be loaded. Make sure the haarcascade_frontalface_default.xml is present."
    )
    exit()


# Helper Functions
def total_registered_users():
    """Returns the total number of registered users."""
    return len(os.listdir("static/faces"))


def extract_faces(img):
    """Detects faces in an image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))


def train_model():
    """Trains and saves the face recognition model."""
    faces, labels = [], []
    for user in os.listdir("static/faces"):
        user_path = os.path.join("static/faces", user)
        for imgname in os.listdir(user_path):
            img_path = os.path.join(user_path, imgname)
            img = cv2.imread(img_path)
            if img is not None:
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel())
                labels.append(user)

    if faces:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(np.array(faces), np.array(labels))
        joblib.dump(knn, model_path)
        print("[INFO] Model trained and saved.")
    else:
        print("[ERROR] No face data found to train the model.")


def identify_face(facearray):
    """Identifies the face using the trained model."""
    if not os.path.isfile(model_path):
        return None
    model = joblib.load(model_path)
    return model.predict(facearray)[0]


def add_attendance(name):
    """Adds attendance for the identified person."""
    username, userid = name.split("_")
    current_time = datetime.now().strftime("%H:%M:%S")
    with open(attendance_file, "r") as f:
        if userid in f.read():
            return
    with open(attendance_file, "a") as f:
        f.write(f"{username},{userid},{current_time}\n")


def extract_attendance():
    """Extracts attendance data from the CSV file."""
    try:
        df = pd.read_csv(attendance_file)
        names = df["Name"].tolist()
        rolls = df["Roll"].tolist()
        times = df["Time"].tolist()
        return names, rolls, times, len(df)
    except Exception:
        return [], [], [], 0


def get_user_data(user_id):
    """Retrieve user data (name and images) for updating."""
    user_folder = f"static/faces/{user_id}"
    user_name = user_id.split("_")[0]
    user_images = [f"static/faces/{user_id}/{img}" for img in os.listdir(user_folder)]
    return {"username": user_name, "userid": user_id, "images": user_images}


def update_user_data(user_id, new_name, new_id):
    """Update the user data."""
    user_folder = f"static/faces/{user_id}"
    new_user_folder = f"static/faces/{new_name}_{new_id}"
    os.rename(user_folder, new_user_folder)
    # Update attendance file
    df = pd.read_csv(attendance_file)
    df.loc[df["Roll"] == user_id, "Name"] = new_name
    df.loc[df["Roll"] == user_id, "Roll"] = new_id
    df.to_csv(attendance_file, index=False)
    # Train model again after update
    train_model()


def delete_user_data(user_id):
    """Delete the user and their data."""
    user_folder = f"static/faces/{user_id}"
    if os.path.exists(user_folder):
        for img_file in os.listdir(user_folder):
            os.remove(os.path.join(user_folder, img_file))
        os.rmdir(user_folder)
    # Remove user from attendance file
    df = pd.read_csv(attendance_file)
    df = df[df["Roll"] != user_id]
    df.to_csv(attendance_file, index=False)
    # Train model again after deletion
    train_model()


# Routes
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    names, rolls, times, l = extract_attendance()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        totalreg=total_registered_users(),
        datetoday2=datetoday2,
    )


@app.route("/add", methods=["GET", "POST"])
def add_user():
    if request.method == "POST":
        newusername = request.form["newusername"]
        newuserid = request.form["newuserid"]
        user_folder = f"static/faces/{newusername}_{newuserid}"

        if os.path.exists(user_folder):
            return render_template("add.html", mess="User already exists!")

        os.makedirs(user_folder, exist_ok=True)
        cap = cv2.VideoCapture(0)
        i, count = 0, 0
        while i < nimgs:
            ret, frame = cap.read()
            if not ret:
                continue
            faces = extract_faces(frame)
            for x, y, w, h in faces:
                face = frame[y : y + h, x : x + w]
                cv2.imwrite(f"{user_folder}/{i}.jpg", cv2.resize(face, (50, 50)))
                i += 1
            cv2.imshow("Add User", frame)
            if cv2.waitKey(1) == 27:  # Press ESC to stop
                break
        cap.release()
        cv2.destroyAllWindows()

        train_model()
        return redirect(url_for("home"))
    return render_template("add.html")


@app.route("/start", methods=["GET"])
def start_attendance():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = extract_faces(frame)
        for x, y, w, h in faces:
            face = cv2.resize(frame[y : y + h, x : x + w], (50, 50)).reshape(1, -1)
            identified_person = identify_face(face)
            if identified_person:
                add_attendance(identified_person)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 2)
                cv2.putText(
                    frame,
                    identified_person,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

        # Display the frame using Matplotlib (instead of cv2.imshow)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis("off")  # Hide axes
        plt.show(block=False)
        plt.pause(0.1)  # Display the image for a short period before continuing

        if cv2.waitKey(1) == 27:  # Press ESC to stop
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for("home"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == "admin" and password == "admin123":
            session["user"] = username
            return redirect(url_for("home"))
        else:
            return render_template("login.html", message="Invalid credentials")
    return render_template("login.html")


@app.route("/logout")
def logout():
    """Logs out the current user and redirects to the login page."""
    session.pop("user", None)  # Remove the user from the session
    return redirect(url_for("login"))  # Redirect to the login page


@app.route("/update/<user_id>", methods=["GET", "POST"])
@app.route("/update/<user_id>", methods=["GET", "POST"])
def update_user(user_id):
    user_data = get_user_data(user_id)
    if request.method == "POST":
        new_username = request.form["newusername"]
        new_userid = request.form["newuserid"]
        update_user_data(user_id, new_username, new_userid)
        return redirect(url_for("home"))
    return render_template(
        "update_user.html",
        user_name=user_data["username"],
        user_id=user_data["userid"],
        user_images=user_data["images"],
    )


@app.route("/delete/<user_id>", methods=["POST"])
def delete_user(user_id):
    delete_user_data(user_id)
    return redirect(url_for("home"))


@app.route("/reports", methods=["GET", "POST"])
def reports():
    """Handle displaying attendance reports and filtering by date."""
    if request.method == "POST":
        # Get the selected date from the form
        attendance_date = request.form["attendance_date"]

        # Convert the date to the desired format (if needed)
        attendance_date = datetime.strptime(attendance_date, "%Y-%m-%d").strftime(
            "%d-%B-%Y"
        )

        # Filter the attendance records based on the selected date
        names, rolls, times, l = filter_attendance_by_date(attendance_date)

        # Pass filtered data to the template
        return render_template(
            "reports.html",
            names=names,
            rolls=rolls,
            times=times,
            l=l,
            datetoday2=attendance_date,
        )

    # If the request method is GET, show all attendance data
    names, rolls, times, l = get_all_attendance_data()
    datetoday2 = datetime.today().strftime("%d-%B-%Y")

    return render_template(
        "reports.html",
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        datetoday2=datetoday2,
    )


# Function to filter attendance by date (you can modify this to fetch from the database or CSV)
def filter_attendance_by_date(date):
    # Read attendance file or fetch from database
    attendance_file = "Attendance/Attendance-{}.csv".format(date)

    try:
        df = pd.read_csv(attendance_file)
        names = df["Name"].tolist()
        rolls = df["Roll"].tolist()
        times = df["Time"].tolist()
        l = len(df)
        return names, rolls, times, l
    except FileNotFoundError:
        # Handle the case where no records exist for the selected date
        return [], [], [], 0


# Function to get all attendance data (can be updated as needed)
def get_all_attendance_data():
    # For example, load all data (you can customize this)
    attendance_file = "Attendance/Attendance-01_01_2024.csv"  # Example for today's file
    df = pd.read_csv(attendance_file)
    names = df["Name"].tolist()
    rolls = df["Roll"].tolist()
    times = df["Time"].tolist()
    l = len(df)
    return names, rolls, times, l


if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    if not os.path.isfile(model_path):
        print("[INFO] Model not found. Training a new model...")
        train_model()

    app.run(debug=True)
