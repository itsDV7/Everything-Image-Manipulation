from langchain.tools import BaseTool
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoImageProcessor, DetrImageProcessor, DetrForObjectDetection, BlipForQuestionAnswering, SegGptImageProcessor, SegGptForImageSegmentation
from datasets import load_dataset
import numpy as np
import torch
# Diffusion
from open_generative_fill import config
from open_generative_fill.lm_models import run_lm_model
from open_generative_fill.load_data import load_image
from open_generative_fill.vision_models import (
    run_caption_model,
    run_inpainting_pipeline,
    run_segmentaiton_pipeline,
)
from random import randint
from tempfile import NamedTemporaryFile

class ImageCaptioningTool(BaseTool):
    name = "Image Captioning Tool / Image Captioner"
    description = """Use this tool when a path to an image is given and you are asked to describe the image.
    This tool will return a caption about the image in string format.
    The tool should only be used to provide a general description about the image and not to answer any specific questions with context of the image.
    If and only if there is ambiguity or uncertainty in the answer provided by the Image Question Answer Tool after using that tool use this to try and remove ambiguity."""

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cuda"

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(device)

        inputs = processor(images=image, return_tensors="pt").to(device=device)
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

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cuda"

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
    name = "Image Question Answering Tool"
    description = """Use this tool when a path to an image is given and you are asked about any contextual information about the image.
    This tool takes TWO arguments in the order: image_path, question. Always follow this order when calling the tool.
    This tool will return an answer string according to the question asked by the user, keeping in mind the context provided by the image.
    This tool MUST NOT be used to provide a general description about the image."""

    def _run(self, img_ques):
        img_path, question = img_ques.strip().split(",")

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cuda"

        image = Image.open(img_path).convert('RGB')

        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(device)

        inputs = processor(image, question, return_tensors="pt").to(device)

        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)

        return answer

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class HumanImageSegmentationTool(BaseTool):
    name = "Human Segmentation Tool / Human Image Cropping Tool"
    description = """Use this tool when a path to an image is given and you are asked to extract the part of the image which contains a human figure.
    THIS TOOL IS STRICTLY USED TO SEGMENT IMAGE THAT CONTAINS A HUMAN. IT SHOULD NOT BE USED TO CREATE CLOSE-UPS, CROPS, OR ANY OTHER IMAGE MODIFICATION FOR ANY OTHER OBJECT.
    This tool should be used to segment the area of the image containing humans and MUST NOT BE USED FOR ANY OTHER OBJECT.
    If the user wants to save the human cropped image, give the tool TWO arguments in the order: image_path, True. Otherwise, give the tool TWO arguments in the order: image_path, False. Always follow this order when calling the tool.
    If there is a temp file, don't tell user anything about it.
    This tool will return a string pointing to the path of a new image which is the result of segmentation and this path can be used to further process the segmented image."""

    def _run(self, img_save):
        img_path, save = img_save.strip("()").split(",")

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cuda"

        # Initialize SegGPT model and image processor
        model_id = "BAAI/seggpt-vit-large"
        image_processor = SegGptImageProcessor.from_pretrained(model_id)
        model = SegGptForImageSegmentation.from_pretrained(model_id)

        # Load the specified image
        image_input = Image.open(img_path)

        # Load the dataset for semantic segmentation (assuming it's the same as before)
        dataset_id = "mattmdjaga/human_parsing_dataset"
        ds = load_dataset(dataset_id, split="train")

        # Number of labels for the dataset
        num_labels = 18

        # Assuming you want to use the first image and mask prompts from the dataset
        image_prompt = ds[0]["image"]
        mask_prompt = ds[0]["mask"]

        # Process the image and prompts
        inputs = image_processor(
            images=image_input,
            prompt_images=image_prompt,
            prompt_masks=mask_prompt,
            num_labels=num_labels,
            return_tensors="pt"
        )

        # Perform segmentation
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process segmentation to get the mask
        target_sizes = [image_input.size[::-1]]
        mask = image_processor.post_process_semantic_segmentation(outputs, target_sizes, num_labels=num_labels)[0]

        # Convert mask to numpy array
        mask_array = mask.cpu().numpy()

        # Apply the mask to the original image
        image_np = np.array(image_input)
        masked_image_np = np.copy(image_np)

        # Set non-masked pixels to black
        masked_image_np[mask_array == 0] = 0

        # Convert numpy array back to PIL image
        masked_image_pil = Image.fromarray(masked_image_np)

        # Save or display the masked image
        # masked_image_pil.show()
        if save:
            masked_image_pil.save("cropped_image.jpg")
            return "cropped_image.jpg is the path of the new cropped image."
        else:
            with NamedTemporaryFile(dir=".", suffix=f'.tempjpg') as tf:
                f.write(masked_image_pil.getbuffer())
                cropped_image_path = f.name
            return f"How does this look? Should I save this to your device? Here is a temp path to the image: {cropped_image_path}. Don't tell the user about the temp image or the temp path."

        def _arun(self, query: str):
            raise NotImplementedError("This tool does not support async")

class ObjectCroppingTool(BaseTool):
    name = "Image Object Cropper Tool / Image Object Extracter Tool"
    description = """Use this tool when a path to an image is given and you are asked to extract or crop out a specific object from the image.
    To use this tool, get the bounding boxes of the object that needs to be cropped. This tool should majorly be used for non-human object segmentation, cropping, extraction, etc.
    If user had specified more than one object to crop out of the image, use this tool again and again to crop out all objects one by one.
    The tool requires the input to be in this order: image_path,(value1:value2:value3:value4) where values come from the [value1, value2, value3, value4] of the selected bounding box.
    This tool can be used to crop out human objects but only if the user has specifically specified to crop out a human bounding box or crop out a human object and some area around them. Otherwise, ONLY USE Human Image Segmentation Tool.
    This tool will return as string the location of the cropped image each time it is done processing and saved."""

    def _run(self, img_bbox):
        print(img_bbox)
        img_path, bbox = img_bbox.strip("()").split(",")
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cuda"

        image = Image.open(img_path)

        # Extract bounding box coordinates
        x1, y1, x2, y2 = list(map(int, bbox.strip("()").split(":")))

        # Crop the image
        cropped_image = image.crop((x1, y1, x2, y2))

        # Save the cropped image
        save_path = f'Obj_{img_path.split("/")[-1]}_cropped.jpg'
        cropped_image.save(save_path)

        return f"Cropped image saved at {save_path}"

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class ImageDiffusionTool(BaseTool):
    name = "Image Diffusion Tool / Image Modifier Tool / Generative Image Tool"
    description = """Use this tool when a path to an image is given and you are asked to change some object inside of an image to something else, or when you are asked to change the style of the image like watercolor, cartoon, etc.
    This is a diffusion tool and is only used for related tasks. This tool is STRICTLY not used to crop the image, segment the image, or modify any meta-data of the provided image.
    This tool takes the input in this order: image_path, a one liner sentence containing the object the user wants to change and another object the user wants to change it into.
    The output of this image will be another image and the tool will return a string containing the location of saved image.
    After using this tool, try to use Image Captioning Tool to look at the results of this tool and determine if they satisfy the user needs. If not, try to run this tool again on the original task."""

    def _run(self, img_prompt):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cuda"

        img_path, prompt = img_prompt.strip("()").split(",")
        # Load the specified image
        image = Image.open(img_path)
        edit_prompt = prompt

        # Image captioning
        caption = run_caption_model(
            model_id=config.CAPTION_MODEL_ID, image=image, device=device
        )

        # Language model
        to_replace, replaced_caption = run_lm_model(
            model_id=config.LANGUAGE_MODEL_ID,
            caption=caption,
            edit_prompt=edit_prompt,
            device=device,
        )

        # Segmentation pipeline
        segmentation_mask = run_segmentaiton_pipeline(
            detection_model_id=config.DETECTION_MODEL_ID,
            segmentation_model_id=config.SEGMENTATION_MODEL_ID,
            to_replace=to_replace,
            image=image,
            device=device,
        )

        # Inpainting pipeline
        output = run_inpainting_pipeline(
            inpainting_model_id=config.INPAINTING_MODEL_ID,
            image=image,
            mask=segmentation_mask,
            replaced_caption=replaced_caption,
            image_size=config.IMAGE_SIZE,
            generator=torch.Generator().manual_seed(randint(1, 100)),
            device=device,
        )

        output.save("diffused.jpg")

        return "The edited image is saved as diffused.jpg"

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
