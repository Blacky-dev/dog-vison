# # %%writefile app.py
# import hashlib
# import os
# os.environ['TFHUB_CACHE_DIR'] =r'C:\Users\Black\OneDrive\Desktop\tf\tfhub_modules'
# handle = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4"
# hashlib.sha1(handle.encode("utf8")).hexdigest()
import streamlit as st
import keras
from keras import layers
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import tempfile
import os
import time
tmpdir = tempfile.mkdtemp()
os.environ['TFHUB_CACHE_DIR'] =r'C:\Users\Black\OneDrive\Desktop\tf\tfhub_modules'
PAGE_CONFIG = {"page_title":"Dog Vision AI","page_icon":"dog","layout":"centered"}
st.beta_set_page_config(**PAGE_CONFIG)
st.set_option('deprecation.showfileUploaderEncoding', False)


st.title("Dog Vision AI 🐶")
st.header("Welcome To Dog Breed Identification 👀")
st.write('')


menu = ["Home","About",'Contact']
choice = st.sidebar.selectbox('Menu',menu)
if choice == 'Home':
    # st.write(" bhdsjcbdsjcjdc")
    def teachable_machine_classification(img, weights_file):
        # Load the model
        # weights_file=r'C:\Users\Black\OneDrive\Desktop\Dog_ai_webapp\20200911-121337-10000-images-mobilenet-v2-Adam_optimizer.h5'
        model = tf.keras.models.load_model(weights_file,
                                    custom_objects={'KerasLayer':hub.KerasLayer})
        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, 512, 512, 3), dtype=np.float32)
        image = img
        #image sizing
        size = (512, 512)
        image = ImageOps.fit(image, size)

        #turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 255)
        print(normalized_image_array)

        # Load the image into the array
        data[0] = normalized_image_array
        print(data)
        # run the inference
        prediction = model.predict(data)
        preds=(np.argsort(prediction))
        print(preds)
        des=(np.flip(preds))
        # print(des)
        print(des[0][0:5])
        top_5=(des[0][0:5])
        #creating y axis array
        sorte=[]
        for i in top_5:
            sorte.append(prediction[0][i])
        print(sorte)

        print(unique_breeds[top_5])
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20,10))
        plotting=(plt.bar(unique_breeds[top_5],sorte,color='grey'))
        plotting[0].set_color('seagreen')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=20)
        st.pyplot()#top_5[prediction])
        max_pred_index=np.argmax(prediction)
        result=np.max(prediction)*100
        result="%.2f" % result
        # print(f'Prediction Confidence...{result} %')
        return max_pred_index,result# return position of the highest probability	
    uploaded_file = st.file_uploader("Choose a Dog Picture To predict its breed ...", type=["jpeg",'jpg'])
    if uploaded_file is not None:
        labels_csv=pd.read_csv(r'C:\Users\Black\OneDrive\Desktop\Dog_ai_webapp\labels (1).csv')
    # labels_csv
        breed_kinds=np.array(labels_csv['breed'])
    # breed_kinds
    # len(breed_kinds)
        unique_breeds=np.unique(breed_kinds)
        image = Image.open(uploaded_file)
    
        st.image(image, caption='Uploaded Dog Image.', use_column_width=True)
        st.write("")
        st.markdown("**Classifying Dog Breed...**")
        bar = st.progress(0)
        latest_iteration = st.empty()
        start = time.time()
        for i in range(1):  
            with st.spinner('Wait for it...'):
                label = teachable_machine_classification(image, r'C:\Users\Black\OneDrive\Desktop\Dog_ai_webapp\20200911-121337-10000-images-mobilenet-v2-Adam_optimizer.h5')
            # label
                st.markdown(f'The Breed of Dog Is **{unique_breeds[label[0]]}.**')
                st.markdown(f'Prediction Confidence-**{label[1]} %.**')
                st.success('Done!')
                st.info('')
        end = time.time()
        print(end-start)
        latest_iteration.text(f'classification Done...{i+1}')
        bar.progress(100)
        time.sleep(0.1)
        st.balloons()
        

        
  # Update the progress bar with each iteration.
            
        
            
            
    
    # print(label)
    # st.write(label)
    # else:
    #     st.write(prediction)
elif choice=='About':
    st.markdown('This is a simple startup project on data-science/Machine-learning which predicts the **Breed** of a **Dog** from a given image.')
    from PIL import Image
    st.markdown('This model is based on **Transfer learning**')
    st.subheader('This is the acc graph after training the model on 10,000 of dog images of different breeds')
    acc_image = Image.open(r'C:\Users\Black\OneDrive\Desktop\Dog_ai_webapp\dog ai acc.jpg')
    st.image(acc_image, caption='Accuracy of model-93.5 % on validation data', use_column_width=True)
    st.subheader('This is the loss graph after training the model on 10,000 of dog images of different breeds')
    loss_image = Image.open(r'C:\Users\Black\OneDrive\Desktop\Dog_ai_webapp\dog ai loss.jpg')
    st.image(loss_image, caption='Loss of model-0.21 min loss on validation data', use_column_width=True)
    labels_csv=pd.read_csv(r'C:\Users\Black\OneDrive\Desktop\Dog_ai_webapp\labels (1).csv')
    breeds=labels_csv['breed'].value_counts()
    # print(breeds)
    import altair as alt
    st.subheader('total no.of different dog breeds on which model is trained on')
    st.bar_chart(breeds,width=300,height=500,use_container_width=True)
else:
    st.subheader('Contact Me:')
    mail_icon=Image.open(r'C:\Users\Black\OneDrive\Desktop\Dog_ai_webapp\mail.png')
    st.image(mail_icon,width=32)
    link = '**Email**:[diffusion00721@gmail.com](https://mail.google.com/mail/u/1/#inbox?compose=new)'
    st.markdown(link, unsafe_allow_html=True)
    git_image = Image.open(r'GitHub-Mark-32px.png')
    st.write('')
    st.image(git_image,width=36)
    st.markdown('**github**:https://github.com/Blacky-dev')


