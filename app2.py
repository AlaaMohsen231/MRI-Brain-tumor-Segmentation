import streamlit as st
# import tensorflow as tf
from tensorflow import keras
from keras import models
print(keras.__verson__)
from keras.models import load_model
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import os
from transformers import pipeline  # Hugging Face Transformers


medical_chatbot = pipeline("question-answering", model="deepset/bert-base-cased-squad2")
text_generator = pipeline("text-generation", model="gpt2")

resunet=load_model("brain_tumor_segmentation_model.h5")

# @st.cache(allow_output_mutation=True)
# def load_model():
#     interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
#     interpreter.allocate_tensors()
#     return interpreter

# def preprocess_image(image, target_size):
#     # Ensure the image is grayscale and convert to RGB (3 channels)
#     if image.mode != 'RGB':
#         image = image.convert('RGB')  # Convert grayscale to RGB (3 channels)
    
#     # Resize image to the expected size from the model
#     image = image.resize(target_size)
    
#     # Normalize to [0, 1] if needed (depends on your model training)
#     image = np.array(image).astype(np.float32) / 255.0
    
#     # Expand dimensions to add batch dimension
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
    
#     return image

# def predict(image, interpreter):
#     # Get input and output details
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     # Set the input tensor
#     interpreter.set_tensor(input_details[0]['index'], image)

#     # Run inference
#     interpreter.invoke()

#     # Get the output
#     output = interpreter.get_tensor(output_details[0]['index'])
#     return output

    # Preprocessing function
def preprocess_image(image):
    """Preprocess the uploaded MRI image for prediction."""
    img = load_img(image, target_size=(128, 128), color_mode="grayscale")
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Post-processing function
def postprocess_prediction(prediction):
    """Convert prediction mask back to original image size."""
    prediction = prediction.squeeze()  # Remove batch dimension
    prediction = (prediction > 0.5).astype(np.uint8) * 255  # Binary mask
    return prediction


    # Prediction function
def predict_segmentation(image):
    # Preprocess the image
    img = preprocess_image(image)

    model = resunet

    # Make prediction
    prediction = model.predict(img)

    # Post-process the prediction
    segmented_mask = postprocess_prediction(prediction)

    # Generate text based on whether tumor is detected or not
    # generated_text = generate_text(has_tumor)

    # Resize back to original image size
    original_img = load_img(image)
    segmented_mask = cv2.resize(segmented_mask, (original_img.width, original_img.height))

    # Return the original image and the segmented mask
    return  segmented_mask

def generate_text(has_tumor):
    """Generate text based on tumor detection using Hugging Face."""
    if has_tumor:
        prompt = "Based on the brain tumor segmentation analysis,A tumor was detected. We recommend that you consult with a neurologist to discuss further diagnostic tests and treatment options."
    else:
        prompt = "The brain tumor segmentation analysis did not reveal any evidence of a tumor. However, we recommend that you continue to monitor your symptoms and follow up with your doctor as advised."

    # generated_text = text_generator(prompt, max_length=50, num_return_sequences=1)
    return prompt
    # return generated_text[0]["generated_text"]


def answer_question(question, context):
    """Use Hugging Face model to answer medical questions."""
    result = medical_chatbot(question=question, context=context)
    return result["answer"]

medical_context = """
Brain tumors are abnormal growths of cells in the brain. Tumors can be benign (non-cancerous) or malignant (cancerous).
Common symptoms of brain tumors include headaches, seizures, memory problems, and changes in behavior.
Treatment options depend on the type and size of the tumor, and can include surgery, radiation therapy, or chemotherapy.

Brain tumors are abnormal growths of tissue within the brain. They can be benign (non-cancerous) or malignant (cancerous). The severity and treatment options for brain tumors vary widely depending on several factors, including:

Type: There are numerous types of brain tumors, each with its own characteristics and prognosis. Some common types include gliomas, meningiomas, and pituitary tumors.
Location: The location of a brain tumor within the brain can significantly impact its symptoms and treatment options. For example, a tumor located in the motor cortex may affect movement, while one in the visual cortex may impact vision.
Size: Larger tumors are often associated with more severe symptoms and may require more aggressive treatment.
Grade: The grade of a brain tumor refers to its aggressiveness and the likelihood of its recurrence. Higher-grade tumors are generally more malignant and difficult to treat.
Symptoms of brain tumors can vary widely depending on their location and size. Some common symptoms include:

Headaches
Seizures
Changes in vision
Weakness or numbness
Difficulty speaking or understanding speech
Balance problems
Personality changes
Diagnosis of brain tumors typically involves a combination of imaging tests, such as MRI and CT scans, and sometimes a biopsy.

Treatment options for brain tumors depend on the type, location, size, and grade of the tumor. Common treatment options include:

Surgery: Surgical removal of the tumor is often the preferred treatment option, especially for benign tumors and some malignant tumors.
Radiation therapy: Radiation therapy uses high-energy rays to kill cancer cells. It may be used after surgery to reduce the risk of recurrence or as the primary treatment for some tumors.
Chemotherapy: Chemotherapy uses drugs to kill cancer cells. It may be used in combination with surgery or radiation therapy for certain types of brain tumors.   
Targeted therapy: Targeted therapy uses drugs that specifically target the genetic changes that occur in cancer cells. This type of treatment is becoming increasingly important for treating brain tumors.
Prognosis for brain tumors varies widely depending on the factors mentioned above. While some brain tumors can be cured, others are more difficult to treat and may lead to a poorer outcome. Early diagnosis and treatment can improve the prognosis for many patients with brain tumors.
"""

def segmentation():
    # Streamlit app layout
    st.title("Brain Tumor Segmentation")
    st.write("Upload an MRI image to detect the tumor using a quantized ResUNet model.")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["tif"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        image_resized = image.resize((300, int(image.height * (300 / image.width))))
        st.image(image_resized,caption='Uploaded MRI Image')
        # st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        # Load the TFLite model
        # interpreter = load_model()
        
        # Get the correct input shape for resizing (ignoring batch size)
        # input_details = interpreter.get_input_details()
        # input_shape = input_details[0]['shape'][1:3]  # Get height and width

        # # Preprocess image to the correct size
        # preprocessed_image = preprocess_image(image, input_shape)

        # Predict
        st.write("Classifying...")
        output = predict_segmentation(image, resunet)

        # Post-process output (thresholding for segmentation)
        output_image = np.squeeze(output)  # Remove batch dimension
        
        # Check the output range (should be between 0 and 1 for segmentation masks)
        st.write("Model output range:", output_image.min(), output_image.max())

        # Apply thresholding if the output is in the range [0, 1]
        if output_image.max() <= 1.0:
            output_image = (output_image > 0.5).astype(np.uint8)  # Thresholding for binary mask
        else:
            st.write("Unexpected output range! Check the model output.")

        # st.download_button("Download file", output_image)

        tumor=False
        if output_image.max() > 0:
            tumor=True

        # Display output (segmentation mask)
        # output_image *=255
        # image_resized = output_image.resize((300, int(image.height * (300 / image.width))))
        # st.image(image_resized, caption='Segmented Tumor Area')
        st.image(output_image * 255, caption='Segmented Tumor Area', use_column_width=300)


        #Text generation
        st.write("Advice")
        text=generate_text(tumor)
        st.write(f"{text}")


        # Medical chatbot
        st.write("Medical Chatbot")
        question = st.text_input("Ask a medical question about brain tumors")
        if question:
            answer = answer_question(question, medical_context)
            st.write(f"Chatbot answer: {answer}")

# --------------------------------------------------------------
st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")

# Custom CSS to change background color and add styling
page_bg_color = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #F5F5F5;  /* Light grey background */
}

[data-testid="stSidebar"] {
    background-color: #31333F;  /* Dark background for the sidebar */
    color: white;
}
div[data-testid="stSidebar"] .css-1v3fvcr, div[data-testid="stSidebar"] .css-1inhw11 {
    color: white !important;  /* Force radio button text color to white */
}
h1 {
    color: #0066CC;  /* Custom header color */
}

</style>
"""

# Inject CSS code in Streamlit
st.markdown(page_bg_color, unsafe_allow_html=True)

def home():
    st.title("Home")
    st.write("Welcome to the Brain Tumor Segmentation App.")
    image = Image.open("home.jpg")
    image_resized = image.resize((300, int(image.height * (300 / image.width))))
    st.image(image_resized)


def about():
    st.title("About")

    col1, col2, col3 = st.columns(3)
    col1.write("Marina Safwat")
    col2.write("Rehab Hamdy")
    col3.write("Alaa Mohsen")
    # st.write("""
    # \nMarina Safwat
    # \nRehab Hamdy
    # \nAlaa Mohsen
    # """)
def contact():
    st.title("Contact us")
    st.write("")

def navbar():
    # Sidebar Navbar
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "Segmentation", "About"])

    # Conditional rendering of pages based on selection
    if selection == "Home":
        home()
    elif selection == "Segmentation":
        segmentation()
    elif selection == "About":
        about()
# pg = st.navigation(
#     st.Page(f"{home()}", title="Home", url_path="home", default=True),
#     st.Page(f"{segmentation()}", title="Segmentation", url_path="segmentation"),
#     st.Page(f"{about()}", title="About", url_path="about"))
# pg.run()
tab1, tab2,tab3, tab4 = st.tabs(["Home", "Segmentation","About","Contact us"])
# tab1.write("this is tab 1")
# tab2.write("this is tab 2")

with tab1:
    home()
with tab2:
    segmentation()
with tab3:
    about()
with tab4:
    contact()
# Call the navbar function to display the navigation
# navbar()

# ---------------------------------
