import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly_express as px

st.set_page_config(layout="wide")

END_POINT = "https://pet-finder-analytic-api.herokuapp.com"


@st.cache
def fetch_options(type: str):
    return requests.get(f'{END_POINT}/v1/options?Type={type}').json()


@st.cache
def fetch_pet_image(type, breed_1, breed_2, color_1, color_2, color_3):
    inputs = {'Type': type, 'Breed1': breed_1, 'Breed2': breed_2, 'Color1': color_1, 'Color2': color_2, 'Color3': color_3}
    params = "".join(f"{k}={v}" if i == 0 else f"&{k}={v}" for i, (k, v) in enumerate(inputs.items()))
    return requests.get(f'{END_POINT}/v1/imagepreview?{params}').content


@st.cache
def model_predict(inputs: dict):
    params = "".join(f"{k}={v}" if i == 0 else f"&{k}={v}" for i, (k, v) in enumerate(inputs.items()))
    return requests.get(f'{END_POINT}/v1/inference?{params}').json()


st.title("Pet Finder Analytic")
st.text("This is a pet finder analytic app.\n"
        "this project is aimed to help pet rescuers to promote their pets so their pet can have a better chance to find a new good home")
st.markdown("---")

form = {}

col_1, col_2, col_3, col_4 = st.columns([2, 2, 1, 1])

form['Name'] = col_2.text_input("Pet Name", "Coco")
form['Quantity'] = col_2.number_input("Quantity", 1)
form['Description'] = col_2.text_area("Description")

form['Type'] = col_3.selectbox("Type", ['Cat', 'Dog'])
options = fetch_options(form['Type'])

form['State'] = col_2.selectbox("State", options['State'])
form['Gender'] = col_3.selectbox("Gender", options['Gender'])

breeds_list = options['Breed']
form['Breed1'] = col_3.selectbox("Breed 1", breeds_list, breeds_list.index('Not Specified'))

breeds_2_list = [breed for breed in breeds_list if breed != form['Breed1']]
form['Breed2'] = col_3.selectbox("Breed 2", breeds_2_list, breeds_2_list.index('Not Specified')) if form['Breed1'] != "Not Specified" else "Not Specified"

colors = options['Color']
form['Age'] = col_4.number_input("Age", 1, None, 12)
form['Color1'] = col_4.selectbox("Color 1", colors, colors.index('Not Specified'))

colors_2 = [color for color in colors if color != form['Color1']]
form['Color2'] = col_4.selectbox("Color 2", colors_2, colors_2.index('Not Specified')) if form['Color1'] != 'Not Specified' else 'Not Specified'

colors_3 = [color for color in colors if color != form['Color1'] and color != form['Color2']]
form['Color3'] = col_4.selectbox("Color 3", colors_3, colors_3.index('Not Specified')) if form['Color2'] != 'Not Specified' else 'Not Specified'

NOT_SPECIFIED = "Not Specified"
_type = form['Type']
if form['Age'] < 12:
    if _type == 'Cat':
        _type = 'Kitten'
    elif _type == 'Dog':
        _type = 'Puppy'

try:
    image = fetch_pet_image(_type, form['Breed1'], form['Breed2'], form['Color1'], form['Color2'], form['Color3'])
    col_1.image(image, use_column_width=True)
except:
    image = "https://i.ibb.co/BKz3jN5/86893299-404-page-d-erreur-non-trouv-concept-et-un-symbole-de-lien-bris-ou-mort-comme-un-chat-chaton.jpg"
    col_1.image(image, use_column_width=True)

col_1, col_2, col_3, col_4 = st.columns([1.5, 1, 1, 1])

col_1.markdown("---")
col_1.subheader("Look How Cute They are!!!")
col_1.markdown(
    """let's make they look more appealing on the post that you will be posting  
    you can upload some pictures of their cute face so more people can be attracted to them and more better upload your cute pet video so people will want to snuggle with them
    """)

form['MaturitySize'] = col_2.selectbox("Maturity Size", options['MaturitySize'])
form['FurLength'] = col_2.selectbox("Fur Length", options['FurLength'])
form['Health'] = col_2.selectbox("Health", options['Health'])

choices = options['Vaccinated']
form['Vaccinated'] = col_3.selectbox("Vaccinated", choices, choices.index('Not Sure'))
form['Dewormed'] = col_3.selectbox("Dewormed", choices, choices.index('Not Sure'))
form['Sterilized'] = col_3.selectbox("Sterilized", choices, choices.index('Not Sure'))

form['Fee'] = col_4.number_input("Fee", 0, None, 0)
form['PhotoAmt'] = col_4.number_input("Photo Amount", 0, None, 0)
form['VideoAmt'] = col_4.number_input("Video Amount", 0, None, 0)

st.subheader("Prediction")

col_1, col_2 = st.columns([1, 2])

result = pd.DataFrame([model_predict(form)['probability']]).T
result.columns = ['Probability']
result['Probability (%)'] = (result['Probability'] * 100) // 1
col_1.plotly_chart(px.bar(result, y='Probability (%)', range_y=[0, 100]), use_container_width=True)

col_2.markdown(f"""
    ## Suggestion
    > The probability of your pet being adopted **{result.Probability.idxmax()}** is around **{result['Probability (%)'].max()}%**""")
if result.Probability.idxmax() == "More Than 3 Month":
    col_2.markdown(f"""
    > You should consider to change the form above to better optimize their chances of being adopted
    > so please help them to find a good home and make them happy
    """)
elif result.Probability.idxmax() == "Today":
    col_2.markdown(f"""
    > **Hold Tight!** your pet will be adopted today thanks for your efforts to help them find a new good home
    """)
else:
    col_2.markdown(f"""
    > You have done great job with the post and your pet is a good candidate to be adopted but you can still improve the form above to better optimize their chances of being adopted faster
    """)
