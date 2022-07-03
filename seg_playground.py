import requests
from transformers import (
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
    SegformerConfig,
    SegformerModel,
)
from PIL import Image
import torch.nn.functional as F

if __name__ == '__main__':
    model_name = "nvidia/mit-b0"

    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=10)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = feature_extractor(images=image, return_tensors="pt")
    print(inputs["pixel_values"].size())
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height, width)
    print(logits.size())

    upsampled = F.interpolate(
        logits,
        size=image.size[::-1],  # (height, width)
        mode='bilinear',
        align_corners=False
    )
    print(upsampled.size())

    #
    config = SegformerConfig(num_labels=10)
    seg_model = SegformerModel(config)
    outputs = seg_model(**inputs)
    print(outputs.last_hidden_state.size())
