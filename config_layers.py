config = [(1, "Focus", [64, 3, 1]),
          (1, "Conv", [128, 3, 2]),
          (3, "C3", 128, True),
          (1, "Conv", [256, 3, 2]),
          (6, "C3", 256, True),
          (1, "Conv", [512, 3, 2]),
          (9, "C3", 512, True),
          (1, "Conv", [1024, 3, 2]),
          (3, "C3", 1024, True),
          (1, "SPPF", [1024, 5]),
          (1, "Conv", [512, 1, 1]),
          (1, "Upsample", [None, 2, 'nearest']),
          (3, "C3", 512, False),
          (1, "Conv", [256, 1, 1]),
          (1, "Upsample", [None, 2, 'nearest']),
          (3, "C3", 256, False),
          (1, "ScalePrediction"),
          (1, "Conv", [256, 3, 2]),
          (3, "C3", 512, False),
          (1, "ScalePrediction"),
          (1, "Conv", [512, 3, 2]),
          (3, "C3", 1024, False),
          (1, "ScalePrediction")]

ANCHORS = [[(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
           [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
           [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)]
           ]

IMAGE_SIZE = 416
S = [IMAGE_SIZE // 8, IMAGE_SIZE // 16, IMAGE_SIZE // 32]
