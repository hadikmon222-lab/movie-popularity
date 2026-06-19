import streamlit as st
import joblib
import pandas as pd
import numpy as np
import random

# Page configuration
st.set_page_config(page_title="Movie Popularity Predictor", page_icon="🎬", layout="centered")

# 1. Load CSV Dataset efficiently with caching
@st.cache_data
def load_movie_dataset():
    try:
        df = pd.read_csv('top_rated_movies.csv')
        df = df.dropna(subset=['title', 'release_date', 'overview'])
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df = df.dropna(subset=['release_date'])
        
        # Extract features matching model requirements
        df['year'] = df['release_date'].dt.year.astype(int)
        df['month'] = df['release_date'].dt.month.astype(int)
        df['day'] = df['release_date'].dt.day.astype(int)
        
        # Unique representation to prevent identical title collision issues
        df['display_title'] = df['title'] + " (" + df['year'].astype(str) + ")"
        return df
    except Exception as e:
        # Falls back cleanly to using your hardcoded dictionary if CSV isn't found
        return None

@st.cache_resource
def load_models():
    try:
        # Load ALL 3 pipeline component weights
        model = joblib.load('best_model.pkl')
        cv = joblib.load('vectorizer.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, cv, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

movies_df = load_movie_dataset()
model, cv, scaler = load_models()

# 2. Your Correct Core Hardcoded Dictionary
sample_movies = {
    "The Shawshank Redemption": ("The Shawshank Redemption", "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.", 1994, 9, 23, 9.3, 24000),
    "The Godfather": ("The Godfather", "A powerful mafia family story of love, loyalty and betrayal.", 1972, 3, 24, 9.2, 18000),
    "Inception": ("Inception", "A thief enters people's dreams to steal secrets from their subconscious.", 2010, 7, 16, 8.8, 35000),
    "The Dark Knight": ("The Dark Knight", "Batman faces the Joker who creates chaos in Gotham City.", 2008, 7, 18, 9.0, 30000),
    "Interstellar": ("Interstellar", "A team of explorers travel through a wormhole in space.", 2014, 11, 7, 8.6, 32000),
    "Parasite": ("Parasite", "A poor family schemes to become employed by a wealthy family.", 2019, 5, 30, 8.5, 15000),
    "The Lion King": ("The Lion King", "A young lion prince flees his kingdom after his father is murdered.", 1994, 6, 24, 8.5, 14000),
}

st.title("🎬 Movie Popularity Predictor")
st.write("Predict whether a movie will have **Low, Medium or High** popularity!")
st.divider()

# 3. Consolidate choice index options smoothly
options = list(sample_movies.keys())
if movies_df is not None:
    # Append remaining database entries avoiding duplicating your preset dictionary entries
    csv_titles = [t for t in movies_df['display_title'].tolist() if t.split(" (")[0] not in sample_movies]
    options.extend(csv_titles)

if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = "The Shawshank Redemption"

# Random Selection Trigger
if st.button("🎲 Surprise Me — Random Movie", use_container_width=True):
    st.session_state.selected_movie = random.choice(options)
    st.rerun()

selected = st.selectbox(
    "🎬 Select or Search a Movie:",
    options,
    index=options.index(st.session_state.selected_movie) if st.session_state.selected_movie in options else 0
)
st.session_state.selected_movie = selected

# 4. Safely unpack features from either data engine source
if st.session_state.selected_movie in sample_movies:
    movie_data = sample_movies[st.session_state.selected_movie]
    movie_title_val = movie_data[0]
    movie_overview_val = movie_data[1]
    release_year_val = int(movie_data[2])
    release_month_val = int(movie_data[3])
    release_day_val = int(movie_data[4])
    vote_average_val = float(movie_data[5])
    vote_count_val = int(movie_data[6])
else:
    row = movies_df[movies_df['display_title'] == st.session_state.selected_movie].iloc[0]
    movie_title_val = str(row['title'])
    movie_overview_val = str(row['overview'])
    release_year_val = int(row['year'])
    release_month_val = int(row['month'])
    release_day_val = int(row['day'])
    vote_average_val = float(row['vote_average'])
    vote_count_val = int(row['vote_count'])

st.divider()

# 5. Build Form Element Layouts
movie_title = st.text_input("Movie Title", value=movie_title_val)
movie_overview = st.text_area("Overview", value=movie_overview_val)

col1, col2 = st.columns(2)

with col1:
    vote_average = st.number_input("Vote Average", min_value=0.0, max_value=10.0, value=min(max(vote_average_val, 0.0), 10.0), step=0.1)
    vote_count = st.number_input("Total Vote Count", min_value=0, max_value=200000, value=vote_count_val, step=10)
    release_year = st.number_input("Release Year", min_value=1900, max_value=2026, value=release_year_val)

with col2:
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    release_month = st.slider("Release Month", min_value=1, max_value=12, value=min(max(release_month_val, 1), 12))
    st.write(f"**{months[release_month-1]}**")
    release_day = st.slider("Release Day", min_value=1, max_value=31, value=min(max(release_day_val, 1), 31))

st.divider()

# 6. ML Extraction & Matrix Array Concatenation
if st.button("🔮 Predict Popularity", type="primary", use_container_width=True):
    if model is not None:
        try:
            # Scale inputs matching original RobustScaler vector mapping constraints
            scaled = scaler.transform([[vote_average, vote_count, 0]])
            vote_average_scaled = scaled[0][0]
            vote_count_scaled = scaled[0][1]

            # Parse Overview block text vectors via CountVectorizer
            overview_cv = cv.transform([movie_overview]).toarray()

            # Array composition (Excluding release_year to prevent structural input vector crashes!)
            numeric_features = np.array([[vote_average_scaled, vote_count_scaled, release_month, release_day]])
            input_data = np.concatenate([numeric_features, overview_cv], axis=1)

            prediction = model.predict(input_data)[0]

            st.divider()
            st.subheader("🎯 Prediction Results")

            if prediction == 0:
                st.success("🟢 Low Popularity")
                st.write(f"📉 **{movie_title}** is predicted to have **Low** popularity.")
            elif prediction == 1:
                st.warning("🟡 Medium Popularity")
                st.write(f"👍 **{movie_title}** is predicted to have **Medium** popularity.")
            else:
                st.error("🔴 High Popularity")
                st.balloons()
                st.write(f"🌟 **{movie_title}** is predicted to have **High** popularity!")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
