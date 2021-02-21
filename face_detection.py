import cv2, time

# dummy function
def calculate_bpm(timestamp, xy):
  return 100

if __name__ == '__main__':
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
  video_capture = cv2.VideoCapture(0)
  draw_border = True
  start_time = None

  while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.05, 5)
    for (x, y, w, h) in faces:
      if draw_border:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      # TODO handle case where multiple "faces" are detected
      
      curr_time = time.time()
      if start_time:
        curr_time -= start_time
      else:
        start_time = time.time()
        curr_time = 0
      # This is a dummy function for now
      bpm = calculate_bpm(curr_time, (x, y))

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  video_capture.release()
  cv2.destroyAllWindows()
