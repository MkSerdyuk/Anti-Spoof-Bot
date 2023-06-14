import cv2
import numpy as np
from antispoof.antispoofing import antispoof
from antispoof.detection import face_detector, mtcnn_detection
from antispoof.models import mobilenet_antispoof


class AntiSpoof:
    FRAMES_PER_CALCULATION = 7
    PADDING = 10

    def __init__(self):
        self.model = antispoof.Antispoof(mobilenet_antispoof.MobileNetAntispoof())
        self.detector = face_detector.FaceDetector(mtcnn_detection.MTCNNDetector())
        self.frame_counter = 0
        self.logits_sum = np.array([[0, 0]], dtype=float)
        self.prediction_sum = -1

    def get_processed_video_path(self, vid_path: str) -> str | None:
        vcap = cv2.VideoCapture(vid_path)

        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vcap.get(cv2.CAP_PROP_FPS))

        response_path = vid_path.replace(".", ".result.")

        response = cv2.VideoWriter(
            response_path,
            cv2.VideoWriter_fourcc(*"DIVX"),
            fps,
            (width, height),
        )

        frame_counter = 0
        logits_sum = np.array([[0, 0]], dtype=float)
        prediction_sum = -1

        face_not_found = True

        while vcap.isOpened():
            ret, frame = vcap.read()

            if not ret:
                break

            try:
                faces = self.detector.detect_face(frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                for x0, y0, x1, y1 in faces:
                    x0 -= self.PADDING
                    y0 -= 3 * self.PADDING
                    x1 += self.PADDING
                    y1 += self.PADDING

                    shape = np.shape(frame_rgb)

                    if shape[0] > shape[1]:  # определяем ориентацию видео
                        prediction, logits = self.model.model_predict(
                            np.copy(frame[y0:y1, x0:x1, :])
                        )
                    else:
                        prediction, logits = self.model.model_predict(
                            np.copy(frame[x0:x1, y0:y1, :])
                        )

                    if self.FRAMES_PER_CALCULATION > 1:
                        frame_counter += 1
                        logits_sum += logits
                        logits = logits_sum / frame_counter

                        if self.FRAMES_PER_CALCULATION == frame_counter:
                            prediction_sum = np.argmax(logits_sum, axis=1)
                            logits_sum = np.array([[0, 0]], dtype=float)
                            frame_counter = 1

                        prediction = prediction_sum

                    if prediction:
                        if prediction != -1:
                            color = (0, 255, 0)
                            prediction = "REAL"
                        else:
                            color = (0, 255, 255)
                            prediction = "LOADING..."
                    else:
                        color = (0, 0, 255)
                        prediction = "SPOOF"

                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

                    (w, h), _ = cv2.getTextSize(
                        prediction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1
                    )
                    cv2.rectangle(frame, (x0, y0 - h), (x0 + w, y0), color, -1)
                    cv2.putText(
                        frame,
                        prediction,
                        (x0, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        1,
                    )

                    (w, h), _ = cv2.getTextSize(
                        "real", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                    )
                    cv2.putText(
                        frame,
                        f"real {logits[0][1]}",
                        (x0, y1 + h),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    cv2.putText(
                        frame,
                        f"spoof {logits[0][0]}",
                        (x0, y1 + 2 * h),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

                    face_not_found = False

            except:
                prediction_sum = -1
                frame_counter = 1

            response.write(frame)

        response.release()
        vcap.release()

        if face_not_found:
            return None

        return response_path

    def get_processed_photo_path(self, photo_path: str) -> str | None:
        frame = cv2.imread(photo_path)

        response_path = photo_path.replace(".", ".result.")

        frame_counter = 0
        logits_sum = np.array([[0, 0]], dtype=float)

        face_not_found = True

        try:
            faces = self.detector.detect_face(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for x0, y0, x1, y1 in faces:
                x0 -= self.PADDING
                y0 -= 3 * self.PADDING
                x1 += self.PADDING
                y1 += self.PADDING

                shape = np.shape(frame_rgb)

                if shape[0] > shape[1]:  # определяем ориентацию видео
                    prediction, logits = self.model.model_predict(
                        np.copy(frame[y0:y1, x0:x1, :])
                    )
                else:
                    prediction, logits = self.model.model_predict(
                        np.copy(frame[x0:x1, y0:y1, :])
                    )

                if prediction:
                    if prediction != -1:
                        color = (0, 255, 0)
                        prediction = "REAL"
                    else:
                        color = (0, 255, 255)
                        prediction = "LOADING..."
                else:
                    color = (0, 0, 255)
                    prediction = "SPOOF"

                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

                (w, h), _ = cv2.getTextSize(
                    prediction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1
                )
                cv2.rectangle(frame, (x0, y0 - h), (x0 + w, y0), color, -1)
                cv2.putText(
                    frame,
                    prediction,
                    (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    1,
                )

                (w, h), _ = cv2.getTextSize("real", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.putText(
                    frame,
                    f"real {logits[0][1]}",
                    (x0, y1 + h),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    frame,
                    f"spoof {logits[0][0]}",
                    (x0, y1 + 2 * h),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

                face_not_found = False

        except:
            pass

        if face_not_found:
            return None

        cv2.imwrite(response_path, frame)

        return response_path
