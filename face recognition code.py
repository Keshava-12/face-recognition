import face_recognition
import cv2
import numpy as np
import os
from datetime import date
import xlrd
from xlutils.copy import copy as xl_copy

# Get the current folder path and images
current_folder = os.getcwd()
image1_path = os.path.join(current_folder, 'nani.png')
image2_path = os.path.join(current_folder, 'bubb.png')

# Webcam reference
video_capture = cv2.VideoCapture(0)

# Load and encode images
person_image = face_recognition.load_image_file(image1_path)
person_face_encoding = face_recognition.face_encodings(person_image)[0]
bhargs_image = face_recognition.load_image_file(image2_path)
bhargs_face_encoding = face_recognition.face_encodings(bhargs_image)[0]

# Known encodings and names
known_face_encodings = [person_face_encoding, bhargs_face_encoding]
known_face_names = ["Dhash", "Bhargavi"]

# Initialize variables
face_locations, face_names = [], []
process_this_frame, attendance_taken = True, {}

# Open Excel file
rb = xlrd.open_workbook('attendance_excel.xls', formatting_info=True)
wb = xl_copy(rb)
inp = input('Current subject lecture name: ')
sheet1 = wb.add_sheet(inp)
sheet1.write(0, 0, 'Name/Date')
sheet1.write(0, 1, str(date.today()))
row = 1

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Find all face locations in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        face_names = []
        for face_location in face_locations:
            face_encoding = face_recognition.face_encodings(rgb_small_frame, [face_location])[0]
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            if name!= "Unknown" and not attendance_taken.get(name):
                sheet1.write(row, 0, name)
                sheet1.write(row, 1, "Present")
                row += 1
                attendance_taken[name] = True
                print(f"Attendance taken for {name}")
                wb.save('attendance_excel.xls')

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0,
                    (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
