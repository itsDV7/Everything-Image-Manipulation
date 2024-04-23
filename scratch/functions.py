from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoImageProcessor, DetrImageProcessor, DetrForObjectDetection, BlipForQuestionAnswering
import torch

def generate_image_caption(image_path):
    image = Image.open(image_path).convert('RGB')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)

    inputs = processor(images=image, return_tensors="pt").to(device="cuda")
    output = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption

def detect_objects(image_path):
    image = Image.open(image_path).convert('RGB')

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        detections += ' {}'.format(model.config.id2label[int(label)])
        detections += ' {}\n'.format(float(score))

    return detections


def image_question_answer(image_path, question):
    image = Image.open(image_path).convert('RGB')

    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to("cuda")

    inputs = processor(image, question, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)

    return answer


print(generate_image_caption("dogs_playing.jpg"))
print(detect_objects("dogs_playing.jpg"))
print(image_question_answer("dogs_playing.jpg", "Which dog has the ball?"))
