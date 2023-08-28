import gradio as gr
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import openpyxl
import os

# Load the pre-trained EfficientNet-B7 model
model = models.efficientnet_b7(pretrained=True)
model.eval()

# Define the transformations to be applied to the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_house_area(image_file):
    total_area_sqm = 0
    predicted_areas = []
    
    # Load the input image from the Gradio Image component
    img = Image.fromarray(image_file)

    image_file_name = "single_image.jpg"

    if img.format == "PNG":
        img = img.convert("RGB")

    img_transformed = transform(img)
    img_transformed_batch = torch.unsqueeze(img_transformed, 0)

    with torch.no_grad():
        output = model(img_transformed_batch)

    softmax = torch.nn.Softmax(dim=1)
    output_probs = softmax(output)
    predicted_class = torch.argmax(output_probs)

    predicted_area_sqm = 0
    if predicted_class in [861, 648, 594, 894, 799, 896, 454]:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        edges = cv2.Canny(mask, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > max_area:
                max_area = area

        pixels_per_meter = 300
        predicted_area_sqm = (max_area + 10) / (2 * pixels_per_meter ** 2)
    else:
        predicted_area_sqft = predicted_class.item()
        predicted_area_sqm = predicted_area_sqft * 0.092903 / 4.2

    total_area_sqm += predicted_area_sqm
    predicted_areas.append(predicted_area_sqm)

    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.cell(row=1, column=1).value = "Image File"
    worksheet.cell(row=1, column=2).value = "Predicted Area (sqm)"
    worksheet.cell(row=2, column=1).value = image_file_name
    worksheet.cell(row=2, column=2).value = predicted_area_sqm

    temp_file = "predicted_area.xlsx"
    workbook.save(temp_file)

    return f"Predicted house square footage: {predicted_area_sqm:.2f} square meters", temp_file

inputs = [
    gr.inputs.Image(label="Image")
]

outputs = [
    gr.outputs.Textbox(label="Predicted House Square Footage"),
    gr.outputs.File(label="Excel Printed Result"),
]

interface = gr.Interface(
    fn=predict_house_area,
    inputs=inputs,
    outputs=outputs,
    title="House Square Predictor",
    allow_flagging="never"  # Disable flag button
)

if __name__ == "__main__":
    interface.launch()
