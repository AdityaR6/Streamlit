import streamlit as st
import cv2
import base64
import numpy as np
from PIL import Image, ImageDraw
from deepface import DeepFace
import text2emotion as te

def take_photo(filename='user_photo.jpg', quality=0.8):
    # ... (The same code from the previous snippet for taking a photo using the webcam)
    # Capture photo using the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Unable to access the webcam.")
        return

    st.write("Capturing your photo... Please wait.")

    ret, frame = cap.read()

    if not ret:
        st.error("Error: Unable to capture the photo.")
        return None  # Return None to indicate that the photo capturing failed

    # Save the image to a file
    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cap.release()

    # Display the captured image
    st.image(frame, caption="Your Captured Photo", use_column_width=True)

    st.success("User photo saved to: " + filename)

    # Analyze emotion from the photo and update emotion1
    global emotion1
    emotion1, gender = analyze_emotion(filename)

    return filename

def analyze_emotion(image_path):
    # Analyze emotion using DeepFace library
    face_analysis = DeepFace.analyze(img_path=image_path)
    emotion1 = face_analysis[0]['dominant_emotion']
    gender = face_analysis[0]['dominant_gender']
    return emotion1, gender

def create_dog_avatar(finalResult):
    # Define a dictionary to map emotions to corresponding facial expressions
    # finalResults = {
    #     "happy": "smile",
    #     "sad": "frown",
    #     "angry": "angry_eyebrows",
    #     "surprise": "wide_open_eyes",
    #     "fear": "raised_eyebrows",
    #     "neutral": "neutral",
    # }

    # Create a new image with a white background
    finalResult = finalResult.lower()
    width, height = 300, 400
    background_color = (255, 255, 255)
    image = Image.new("RGB", (width, height), background_color)

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Draw the dog body (rectangle)
    body_color = (222, 184, 135)  # Sandy Brown color
    body_width, body_height = 200, 100
    body_position = ((width - body_width) // 2, (height - body_height) // 2)
    draw.rectangle([body_position, (body_position[0] + body_width, body_position[1] + body_height)], fill=body_color)

    # Draw the dog ears (triangles)
    ear_color = (139, 69, 19)  # Saddle Brown color
    ear_size = 40
    left_ear_position = (body_position[0] + 20, body_position[1] - ear_size // 2)
    right_ear_position = (body_position[0] + body_width - 20, body_position[1] - ear_size // 2)
    draw.polygon(
        [left_ear_position, (left_ear_position[0] - ear_size // 2, left_ear_position[1] - ear_size),
         (left_ear_position[0] + ear_size // 2, left_ear_position[1] - ear_size)], fill=ear_color
    )
    draw.polygon(
        [right_ear_position, (right_ear_position[0] - ear_size // 2, right_ear_position[1] - ear_size),
         (right_ear_position[0] + ear_size // 2, right_ear_position[1] - ear_size)], fill=ear_color
    )

    # Draw the eyes and mouth (circles)
    eye_color = (0, 0, 0)  # Black color
    eye_size = 10
    eye_position = (width // 2, height // 2)
    draw.ellipse([eye_position[0] - 30, eye_position[1] - 30, eye_position[0] - 30 + eye_size, eye_position[1] - 30 + eye_size], fill=eye_color)
    draw.ellipse([eye_position[0] + 20, eye_position[1] - 30, eye_position[0] + 20 + eye_size, eye_position[1] - 30 + eye_size], fill=eye_color)

    # Draw the facial expression based on the final emotion
    # finalResult = finalResults.get(finalResult)

    if finalResult == "happy":
        # Draw a smile (arc)
        smile_color = (0, 0, 0)  # Black color
        draw.arc([width // 2 - 20, height // 2 + 20, width // 2 + 20, height // 2 + 30], start=0, end=-180, fill=smile_color, width=2)

    elif finalResult == "sad":
        # Draw a frown (arc)
        frown_color = (0, 0, 0)  # Black color
        draw.arc([width // 2 - 20, height // 2 + 10, width // 2 + 20, height // 2 + 30], start=180, end=0, fill=frown_color, width=2)

    elif finalResult == "angry":
        # Draw angry eyebrows (lines)
        eyebrow_color = (0, 0, 0)  # Black color
        draw.line([(eye_position[0] - 30, eye_position[1] - 20), (eye_position[0] - 10, eye_position[1] - 30)], fill=eyebrow_color, width=2)
        draw.line([(eye_position[0] + 10, eye_position[1] - 30), (eye_position[0] + 30, eye_position[1] - 20)], fill=eyebrow_color, width=2)

    elif finalResult == "surprise":
        # Draw wide-open eyes (ellipses)
        draw.ellipse([eye_position[0] - 30, eye_position[1] - 25, eye_position[0] - 20, eye_position[1] - 20], fill=eye_color)
        draw.ellipse([eye_position[0] + 20, eye_position[1] - 25, eye_position[0] + 30, eye_position[1] - 20], fill=eye_color)
        mouth_color = (0, 0, 0)  # Black color
        mouth_size = 15
        mouth_position = (width // 2, height // 2 + 35)  # Adjust the vertical position of the mouth
        draw.ellipse([mouth_position[0] - mouth_size, mouth_position[1] - mouth_size, mouth_position[0] + mouth_size, mouth_position[1] + mouth_size], fill=mouth_color)

    elif finalResult == "fear":
        # Draw eyebrows raised in fear (lines)
        eyebrow_color = (0, 0, 0)  # Black color
        draw.line([(eye_position[0] - 30, eye_position[1] - 25), (eye_position[0] - 10, eye_position[1] - 30)], fill=eyebrow_color, width=2)
        draw.line([(eye_position[0] + 10, eye_position[1] - 30), (eye_position[0] + 30, eye_position[1] - 25)], fill=eyebrow_color, width=2)

    elif finalResult == "neutral":
        # Draw a neutral expression (straight mouth)
        mouth_color = (0, 0, 0)  # Black color
        draw.line([eye_position[0] - 20, eye_position[1] + 20, eye_position[0] + 20, eye_position[1] + 20], fill=mouth_color, width=2)

    # Save the image
    image.save("dog_avatar.png")
    image.show()
    return image

def get_emotion_from_text(text):
    # Analyze emotion using text2emotion library
    # t2e = te()
    emotion_dict = te.get_emotion(text)
    emotion2 = max(emotion_dict, key=emotion_dict.get)
    return emotion2, emotion_dict[emotion2]

# Streamlit app
st.title('_Emotion based Avatar Generator_')

# Placeholder for the avatar image
avatar_image = None

# Initialize emotion1 outside the form_submit_button block
emotion1 = None

# Option to capture user photo using the webcam and analyze emotion
with st.form("photo_and_emotion_form"):
    if st.form_submit_button("Capture Your Photo and Analyze Emotion"):
        user_photo = take_photo()
        if user_photo is not None:
            st.write("Analyzing emotion from the photo... Please wait.")
            # Analyze emotion from the photo and display the result
            emotion1, gender = analyze_emotion(user_photo)
            st.write(f"Detected Emotion from Photo: {emotion1.capitalize()} | Gender: {gender.capitalize()}")
        else:
            st.error("Error: Unable to capture the photo. Please try again.")

# Take input for user emotion from text prompt
with st.form("text_form"):
    user_emotion_text = st.text_input("Enter your current emotion in text:")

    # Analyze emotion from the text and display the result
    if st.form_submit_button("Submit Text"):
        if user_emotion_text:
            emotion2, emotion_score2 = get_emotion_from_text(user_emotion_text)
            st.write(f"Detected Emotion from Text: {emotion2.capitalize()} | Emotion Score: {emotion_score2}")

            # Generate the dog avatar based on the combined emotions from photo and text
            if emotion1 is not None:
                final_emotion = max(emotion1, emotion2, key=lambda e: emotion1[e] + emotion_score2 * emotion1[e])
            else:
                final_emotion = emotion2

            avatar_image = create_dog_avatar(final_emotion)
            st.image(avatar_image, caption="Your Dog Avatar", use_column_width=True)

# Display the generated avatar image
if avatar_image:
    st.image(avatar_image, caption="Your Dog Avatar", use_column_width=True)