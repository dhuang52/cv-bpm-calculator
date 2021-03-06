import cv2, time

from dataprocessor.bpmcalc import BPMCalc
from threading import Thread

class WebCam:
  """
  Read frames from webcam in separate thread
  """
  def __init__(self, device_index):
    self._video_capture = cv2.VideoCapture(device_index)
    _, self._frame = self._video_capture.read()
    self._stop = False
  
  def start(self):
    Thread(target=self._update, args=()).start()

  def get_frame(self):
    return self._frame
  
  def stop(self):
    self._stop = True

  def _update(self):
    while True:
      if self._stop:
        return
      _, self._frame = self._video_capture.read()


if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    video_capture = cv2.VideoCapture(0)
    webcam = WebCam(0)
    webcam.start()
    draw_border = True
    start_time = None
    calc_time = None
    bpm = None
    bpm_calc = BPMCalc()

    while True:
        frame = webcam.get_frame()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            if draw_border:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            curr_time = time.time()
            if start_time:
                curr_time -= start_time
            else:
                start_time = time.time()
                curr_time = calc_time = 0

            bpm_calc.send_data(curr_time, (x, y))

            if(curr_time - calc_time > 2):
                bpm = bpm_calc.calculate_bpm()
                # print(bpm, "\tcalculation took ", (time.time() - curr_time - start_time), " seconds")
                calc_time = curr_time

        if bpm:
          cv2.putText(frame, f'Estimated bpm: {bpm}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
          cv2.putText(frame, f'Estimating bpm...', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Video', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
          bpm_calc.reset()
          bpm = None

    webcam.stop()
    cv2.destroyAllWindows()
