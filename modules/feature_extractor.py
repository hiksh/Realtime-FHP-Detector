class FeatureExtractor:
  def __init__(self):
    # A space to data for learning
    self.raw_data_history = []

  def extract_from_dots(self, dots_dict):
    """Form
    {
    "nose": [0.51, 0.32, -0.15],
    "l_eye": [0.53, 0.28, -0.12],
    "r_eye": [0.49, 0.28, -0.12],
    "l_shld": [0.65, 0.55, 0.05],
    "r_shld": [0.35, 0.55, 0.05]
    }
    """
    # 사진 하나씩 받아서 저런 형태로 받는데 여기서 특징 딴 다음 데이터 계속 누적시켜 저장시키기
    # 이후 train model에서 훈련 시키면 됨
    vector = []
    for key in dots_dict.keys():
      vector.extend(dots_dict[key])
    
    self.raw_data_history.append(vector)
    return vector

  def extract_from_image(self, image_tensor):
        # Transform image into feature (CNN)
        # Resize, Normalization (/255.0)
        processed_img = image_tensor / 255.0
        return processed_img

if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    dot_vector = FeatureExtractor.extract_from_dots({
    "nose": [0.51, 0.32, -0.15],
    "l_eye": [0.53, 0.28, -0.12],
    "r_eye": [0.49, 0.28, -0.12],
    "l_shld": [0.65, 0.55, 0.05],
    "r_shld": [0.35, 0.55, 0.05]
    })

