import joblib
import gradio as gr
import pandas as pd
import joblib
import os

#Load Model
model = joblib.load("personality_predictor.pkl")

#Predict Function
def predict_personality(
    time_spent_alone,
    stage_fear,
    social_event_attendance,
    going_outside,
    drained_after_socializing,
    friends_circle_size,
    post_frequency
):
    input_df = pd.DataFrame([{
        'Time_spent_Alone' : float(time_spent_alone),
        'Stage_fear' : 1 if stage_fear == 'Yes' else 0,
        'Social_event_attendance' : float(social_event_attendance),
        'Going_outside' : float(going_outside),
        'Drained_after_socializing' : 1 if drained_after_socializing == 'Yes' else 0,
        'Friends_circle_size' : float(friends_circle_size),
        'Post_frequency' : post_frequency,
    }])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    label = "Introvert" if pred == 1 else "Extrovert"
    confidence = round(float(proba[pred])*100, 1)
    emoji = "🧝" if label == "Introvert" else "🎭"
    
    result = f"{emoji} **{label}**\nConfidence: {confidence}%"
    return result

#Gradio UI

with gr.Blocks(title="Personality Predictor") as demo:
    gr.Markdown(
        """
        # 🔮 Personality Predictor
        ### Predict whether someone is an **Introvert** or **Extrovert**
        Fill in the characteristics below and click **Predict**.
        """
    )

    with gr.Row():
        with gr.Column():
            time_spent_alone = gr.Slider(
                minimum=0,
                maximum=11,
                step=0.5,
                value=5,
                label="Time Spent Alone (hrs/day)",
                info="How much time do you spend alone?"
            )

            stage_fear = gr.Radio(
                choices=['Yes', 'No'],
                value='No',
                label="Stage Fear",
                info="Do you have stage fear?"
            )

            social_event_attendance = gr.Slider(
                minimum=0,
                maximum=10,
                step=0.5,
                value=5,
                label="Social Event Attendance (per month)",
                info="How often do you attend social events?"
            )

            going_outside = gr.Slider(
                minimum=0,
                maximum=7,
                step=0.5,
                value=3,
                label="Going Outside (days/weeks)",
                info="How often do you go outside?"
            )

        with gr.Column():
            drained_after_socializing = gr.Radio(
                choices=['Yes', 'No'],
                value='No',
                label="Drained after socializing",
                info="Do you feel drained after socializing?"
            )

            friends_circle_size = gr.Slider(
                minimum=0,
                maximum=20,
                step=1,
                value=8,
                label="Friends Circle Size",
                info="How many friends do you have?"
            )

            post_frequency = gr.Slider(
                minimum=0,
                maximum=10,
                step=0.5,
                value=4,
                label="Post Frequency (per week)",
                info="How often do you post on social media?"
            )

    predict_btn = gr.Button("Predict Personality", variant="primary")

    output = gr.Markdown(label="Result")

    predict_btn.click(
        fn=predict_personality,
        inputs=[
            time_spent_alone,
            stage_fear,
            social_event_attendance,
            going_outside,
            drained_after_socializing,
            friends_circle_size,
            post_frequency,
        ],
        outputs=output,
    )

    gr.Markdown(
        """
        ---
        ### API Usage(POST Request)

        This app also exposes a REST API. Send a POST request to:
        ```
        POST https://achinthamalinda-personality-predictor.hf.space
        ```

        **Request body (JSON):**
        ```json
        {
            "data": [5, "No", 5, 3, "No", 8, 4]
        }

        **Field order:**
        `
        time_spent_alone,
        stage_fear,
        social_event_attendance,
        going_outside,
        drained_after_socializing,
        friends_circle_size,
        post_frequency
        `

        **Response (JSON):**
        ```json
        {
            "data":["**Extrovert**\\nConfidence: 92.3%"]
        }
        ```
        """)

demo.launch()