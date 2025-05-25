import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import pickle
import numpy as np
import os

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

stop_words = set(stopwords.words('english')) - {'not', 'no', 'never', 'none', 'nor'}
lemmatizer = WordNetLemmatizer()

MAX_NB_WORDS = 15000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 300 
FILTER_SIZES = [3,4,5]
NUM_FILTERS = 128
L2_REG = 0.1
DROPOUT_RATE = 0.6

def build_textcnn_model():
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding_layer = Embedding(
        input_dim=MAX_NB_WORDS + 1,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=True
    )(input_layer)
    
    conv_blocks = []
    for sz in FILTER_SIZES:
        conv = Conv1D(NUM_FILTERS, sz, activation='relu', padding='valid', kernel_regularizer=l2(L2_REG))(embedding_layer)
        conv = BatchNormalization()(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    
    concat = Concatenate()(conv_blocks)
    dropout = Dropout(DROPOUT_RATE)(concat)
    
    dense = Dense(64, activation='relu', kernel_regularizer=l2(L2_REG))(dropout)
    dense = Dropout(DROPOUT_RATE)(dense)
    
    output_layer = Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@st.cache_resource
def load_model_and_tokenizer():
    """Load model weights and tokenizers, and add path verification"""
    required_files = {
        'model_weights': 'best_textcnn2.weights.h5',
        'tokenizer': 'tokenizer.pkl'
    }
    
    missing = [k for k, v in required_files.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f"Missing File: {', '.join(missing)}")
    try:
        model = build_textcnn_model()
        model.load_weights(required_files['model_weights'])
        
        with open(required_files['tokenizer'], 'rb') as f:
            tokenizer = pickle.load(f)
            
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Load Fail: {str(e)}")
def data_processing(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", '', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r"[^\w\s-]", '', text)
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if len(w) > 1 and w.isalpha() and w not in stop_words]
    return ' '.join(filtered)

def lemmatization(text):
    return ' '.join([lemmatizer.lemmatize(w, pos='v') for w in text.split()])


def predict_sentiment(text, model, tokenizer):
    try:
        processed = data_processing(text)
        lemmatized = lemmatization(processed)
        seq = tokenizer.texts_to_sequences([lemmatized])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        prediction = model.predict(padded, verbose=0)[0][0]
        return prediction
    except Exception as e:
        raise RuntimeError(f"Prediction failure: {str(e)}")

st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("""
<style>
    .st-emotion-cache-6qob1r { background-color: #f0f2f6; }
    .stTextInput input, .stTextArea textarea { border-radius: 10px!important; }
    .st-b7 { color: #white!important; }
    .progress-container { 
        position: relative; 
        height: 10px; 
        background: #ddd; 
        border-radius: 5px; 
        overflow: hidden;
    }
    .progress-bar { 
        height: 100%; 
        transition: width 0.3s ease; 
    }
    .negative-progress { 
        position: absolute; 
        right: 0;
        transform: scaleX(-1); 
    }
</style>
""", unsafe_allow_html=True)

text = st.text_area("Please Enter The Movie Review（Support Long Text Analysis）：", height=150, 
                   placeholder="Exp：This movie completely exceeded my expectations...")

try:
    model, tokenizer = load_model_and_tokenizer()
except Exception as e:
    st.error(f"Initialization Error: {str(e)}")
    st.error("Please check:\n1. Is the model file in the current directory\n2. Is the file name best_textcnn2.weights.h5 and tokenizer.pkl")
    st.stop()

if st.button("Start Analysis", type="primary"):
    if not text.strip():
        st.warning("Please enter the text to be analyzed")
    else:
        with st.spinner("Analyzing, please wait..."):
            try:
                prediction = predict_sentiment(text, model, tokenizer)
                
                if prediction < 0.5:
                    emoji = "😊"
                    label = "Positive reviews"
                    color = "#1f77b4"
                    confidence = prediction
                else:
                    emoji = "😞"
                    label = "Negative reviews"
                    color = "#ff4b4b"
                    confidence = 1 - prediction
                
                progress_width = confidence * 100
                
                st.markdown(f"""
                <div style="padding:20px; border-radius:10px; background:{color}22;">
                    <h3 style="color:{color}; margin:0;">{emoji} {label}</h3>
                    <div style="margin-top:10px;">
                        <div class="progress-container">
                            <div class="progress-bar" style="width:{progress_width}%; background:{color};"></div>
                        </div>
                        <p style="color:{color}; margin-top:5px; text-align:right;">
                            Confidence: {confidence:.1%}
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

if st.checkbox("Debug information"):
    st.subheader("Preprocessing process")
    sample_text = "This movie was terribly disappointing, despite the good acting."
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Original text:")
        st.code(sample_text)
    
    with col2:
        st.write("Data preprocess：")
        processed = data_processing(sample_text)
        st.code(processed)
    
    with col3:
        st.write("Lemmatization:")
        lemmatized = lemmatization(processed)
        st.code(lemmatized)
    
    st.subheader("System information")
    st.write("TensorFlow version:", tf.__version__)
    st.write("Working directory:", os.getcwd())
    st.write("Directory file list:", [f for f in os.listdir() if f.endswith(('.h5', '.pkl'))])