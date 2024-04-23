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


def image_segmentation(image_path):
    # import torch
    from datasets import load_dataset
    # from transformers import SegGptImageProcessor, SegGptForImageSegmentation
    #
    # model_id = "BAAI/seggpt-vit-large"
    # image_processor = SegGptImageProcessor.from_pretrained(model_id)
    # model = SegGptForImageSegmentation.from_pretrained(model_id)
    #
    # dataset_id = "mattmdjaga/human_parsing_dataset"
    # ds = load_dataset(dataset_id, split="train")
    #
    # num_labels = 18
    #
    # image_input = ds[4]["image"]
    # ground_truth = ds[4]["mask"]
    # image_prompt = ds[29]["image"]
    # mask_prompt = ds[29]["mask"]
    #
    # inputs = image_processor(
    #     images=image_input,
    #     prompt_images=image_prompt,
    #     prompt_masks=mask_prompt,
    #     num_labels=num_labels,
    #     return_tensors="pt"
    # )
    #
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #
    # target_sizes = [image_input.size[::-1]]
    # mask = image_processor.post_process_semantic_segmentation(outputs, target_sizes, num_labels=num_labels)[0]

    import torch
    from PIL import Image
    from transformers import SegGptImageProcessor, SegGptForImageSegmentation

    # Initialize SegGPT model and image processor
    model_id = "BAAI/seggpt-vit-large"
    image_processor = SegGptImageProcessor.from_pretrained(model_id)
    model = SegGptForImageSegmentation.from_pretrained(model_id)

    # Load the specified image
    image_input = Image.open(image_path)

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


    from PIL import Image
    import numpy as np

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
    masked_image_pil.save("masked_image.jpg")  # Save the masked image to a file if needed

    # Now you can display the segmented mask overlaying the original image as before
    # from PIL import Image
    # import cv2
    # import numpy as np
    # import torch
    #
    # # Load the image
    # # image_input = Image.open("dogs_playing.jpg")
    #
    # # Convert PIL image to NumPy array
    # image_np = np.array(image_input)
    #
    # # Convert NumPy array to PyTorch tensor and permute dimensions
    # image_tensor = torch.tensor(image_np).permute(2, 0, 1)
    #
    # # Convert mask to numpy array
    # mask_array = mask.cpu().numpy()
    #
    # # Convert mask to 3-channel image
    # mask_image = np.zeros_like(image_np)
    # mask_image[:, :, 0] = mask_array * 255
    #
    # # Add mask overlay to original image
    # overlay = cv2.addWeighted(image_np, 1, mask_image.astype(np.uint8), 0.5, 0)
    #
    # # Display the image with mask overlay
    # cv2.imshow('Mask Overlay', overlay)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def diffusion(image_path):
    from open_generative_fill import config
    from open_generative_fill.lm_models import run_lm_model
    from open_generative_fill.load_data import load_image
    from open_generative_fill.vision_models import (
        run_caption_model,
        run_inpainting_pipeline,
        run_segmentaiton_pipeline,
    )
    from random import randint

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the specified image
    image = Image.open(image_path)

    edit_prompt = "make me a cartoon"

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

    # print(output)
    try:
        print("Saved?")
        output.save("diffused.jpg")
    except Exception (e):
        print("No save", e)
    finally:
        print("Got to end.")

def cropper(image_path):
    image = Image.open(image_path)

    # Extract bounding box coordinates
    x1, y1, x2, y2 = [273, 97, 517, 495]

    # Crop the image
    cropped_image = image.crop((x1, y1, x2, y2))

    # Save the cropped image
    cropped_image.save('img_cropper.jpg')

    print(f"Cropped image saved at img_cropper.jpg")


# Post-process the output
# Convert the output tensor back to an image format and save or display it



# print(generate_image_caption("dogs_playing.jpg"))
# print(detect_objects("dogs_playing.jpg"))
# print(image_question_answer("dogs_playing.jpg", "Which dog has the ball?"))
# print(image_segmentation("human.jpg, True"))
print(diffusion("cropped_image.jpg"))
# print(cropper("dogs_playing.jpg"))
