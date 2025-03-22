import streamlit as st
import torch
from PIL import Image
import io
from transformers import AutoProcessor, AutoModelForVision2Seq
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

# Set page configuration
st.set_page_config(
    page_title="Document OCR Extractor",
    page_icon="📄",
    layout="wide"
)

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define OCR pipeline function
def ocr_pipeline(image):
    # Load model and processor
    processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
    model = AutoModelForVision2Seq.from_pretrained(
        "ds4sd/SmolDocling-256M-preview",
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    
    # Prepare input message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this page to docling."}
            ]
        },
    ]
    
    # Process input
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    
    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    trimmed_generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    
    # Decode output
    doctags = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=False)[0].lstrip()
    
    # Create Docling document
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument(name="Document")
    doc.load_from_doctags(doctags_doc)
    
    return doc.export_to_markdown()

# Cache model loading to improve performance
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
    model = AutoModelForVision2Seq.from_pretrained(
        "ds4sd/SmolDocling-256M-preview",
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    return processor, model

# Main application
def main():
    # Application header
    st.title("Document OCR Text Extractor")
    st.markdown("Upload an image containing text to extract its content.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "tiff", "pdf"])
    
    # Process the uploaded image
    if uploaded_file is not None:
        # Display a spinner during processing
        with st.spinner("Processing the image..."):
            # Read the image
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Display the uploaded image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_column_width=True)
            
            # Extract text using OCR pipeline
            result = ocr_pipeline(image)
            
            # Display the extracted text
            with col2:
                st.subheader("Extracted Text")
                st.markdown(result, unsafe_allow_html=True)
            
            # Provide download button for the text
            st.download_button(
                label="Download Extracted Text",
                data=result,
                file_name="extracted_text.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()