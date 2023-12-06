import streamlit as st
import tensorflow as tf
import requests
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie

# --- MODEL LOADING ---
model = tf.keras.models.load_model('model2.h5')

lesion_type_dict = {
    0 : 'Actinic keratoses',
    1 : 'Basal cell carcinoma',
    2 : 'Benign keratosis-like lesions',
    3 : 'Dermatofibroma',
    4 : 'Melanocytic nevi',
    5 : 'Melanoma',
    6 : 'Vascular lesions',
}

lesion_description_dict = {
    0 : "A rough, scaly patch on the skin caused by years of sun exposure. It's often found on the face, ears, lips, back of the hands, forearms, scalp, or neck.",
    1 : "A type of skin cancer that begins in the basal cells. It often appears as a slightly transparent bump on the skin, though it can take other forms.",
    2 : "Commonly known as seborrheic keratoses, these are noncancerous skin growths that vary in color and size. They are usually found on the face, chest, shoulders, or back.",
    3 : "A common, benign skin nodule thats usually small, firm, and red or brown. Its typically found on the legs.",
    4 : "Commonly known as moles, these are usually brown or black skin growths that can appear alone or in groups and often develop during childhood or young adulthood.",
    5 : "The most serious type of skin cancer, which develops in the melanocytes that produce melanin. Melanoma can occur anywhere on the body, in normal skin or in moles that become cancerous.",
    6 : "Abnormal blood vessels in the skin, which can appear as red spots, patches, or slightly raised lesions. They can be congenital or caused by certain skin conditions."
}



# --- PAGE CONFIG  ---
st.set_page_config(page_title="Skin Disease Sorting", page_icon=":mask:", layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_doctor = load_lottieurl("https://lottie.host/d0c230d5-814f-41bf-a814-86049c4ce776/ci59c1Cw0y.json")

# --- HEADER SECTION ---
with st.container():
    st.markdown("<h3 style='text-align: center; color: black;'>Welcome to</h3>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: black;'>Skin Disease Identification App</h1>", unsafe_allow_html=True)
    st_lottie(lottie_doctor, speed=1, height=300, key="doctor")
    st.markdown("<p style='text-align: center; font-size: 20px'>This application uses state-of-the-art machine learning to diagnose skin diseases.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px'>Simply upload an image and let the model diagnose it!</p>", unsafe_allow_html=True)



uploaded_file = st.file_uploader("Choose a skin lesion image...", type="jpg")

# --- ANALYSIS SECTION ---
if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)

    col1, col2, col3= st.columns([1, 1, 1])
    with col2:
        st.image(image.resize((300, 300)), caption='Uploaded Image.', use_column_width=True)
    
    
    image_resized = image.resize((100, 75))
    img_np = np.array(image_resized)
    img_normalized = img_np / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)
    shaping = img_normalized.shape

    predictions = model.predict(img_normalized)
    predictions = np.argmax(predictions, axis = 1)
    st.write("")
    
    # --- RESULT SECTION ---
    # Display the predictions
    st.markdown("<h3 style='text-align: center; font-weight: bold; text-decoration: underline'>Result:</h3>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; font-weight: bold;'>{lesion_type_dict[predictions[0]]}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{lesion_description_dict[predictions[0]]}</h3>", unsafe_allow_html=True)




# --- FOOTER SECTION ---
with st.container():
    st.write("---")
    st.header("Facts:")
    facts = [
        "Skin cancer is the most common cancer globally.",
        "There are mainly three types of skin cancer: basal cell carcinoma, squamous cell carcinoma, and melanoma.",
        "Melanoma is the deadliest form of skin cancer but is less common than the other types.",
        "Ultraviolet (UV) radiation from the sun is the main cause of skin cancer. Tanning lamps and beds are also sources of UV radiation.",
        "Early detection of skin cancer gives a patient a high chance of successful treatment.",
        "Regular self-exams and understanding the ABCDEs of melanoma can help in early detection. (Asymmetry, Border, Color, Diameter, Evolving)",
        "Using broad-spectrum (UVA/UVB) sunscreen and wearing protective clothing can reduce the risk of skin cancer.",
        "While skin cancer is common in many parts of the world, countries with predominantly fair-skinned populations and high UV exposure, such as Australia, have particularly high rates.",
        "Individuals who have had skin cancer once are at a higher risk of getting it again.",
        "People with weakened immune systems, such as organ transplant recipients, are at greater risk.",
        "A family history of skin cancer can increase one's risk."
    ]

    for fact in facts:
        st.markdown(f"- {fact}")

    st.write("Awareness and understanding of these facts can play a significant role in prevention and early detection. Regular check-ups with dermatologists are also crucial for those at higher risk.")
