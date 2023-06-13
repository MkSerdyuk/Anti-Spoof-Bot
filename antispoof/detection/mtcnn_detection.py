from facenet_pytorch import MTCNN


class MTCNNDetector:
    def __init__(self):
        self.mtcnn = MTCNN(select_largest=True)

    def detect_face(self, frame):
        faces, _ = self.mtcnn.detect(frame)
        if len(faces) == 0:
            return -1
        faces = [map(int, face) for face in faces]
        return faces
