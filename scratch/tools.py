from langchain.tools import BaseTool
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoImageProcessor, DetrForObjectDetection, BlipForQuestionAnswering
import torch

class ImageCaptioningTool(BaseTool):
    name = "Image Captioning Tool / Image Captioner"
    description = """Use this tool when a path to an image is given and you are asked to describe the image.
    This tool will return a caption about the image in string format.
    The tool should only be used to provide a general description about the image and not to answer any specific questions with context of the image."""

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(device)

        inputs = processor(images=image, return_tensors="pt").to(device="cuda")
        output = model.generate(**inputs, max_new_tokens=50)

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class ObjectDetectionTool(BaseTool):
    name = "Object Detection Tool / Object Detector"
    description = """Use this tool when a path to an image is given and you are asked to detect the objects in that image.
    This tool will return a list of all the detected objects, their bounding box coordinates, and the confidence score of the model.
    Each element in the list in the format: [x1, y1, x2, y2] class_name confidence_score.
    The tool can also be used to check the existence of an objet in the image BUT you MUST use the image question and answering tool to confirm the answer."""

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

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

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class ImageQuestionAnswerTool(BaseTool):
    name = "Image question answering tool"
    description = """Use this tool when a path to an image is given and you are asked about any contextual information about the image.
    This tool will return an answer string according to the question asked by the user, keeping in mind the context provided by the image.
    This tool MUST NOT be used to provide a general description about the image."""

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to("cuda")

        inputs = processor(image, question, return_tensors="pt").to("cuda")

        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)

        return answer
        
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
