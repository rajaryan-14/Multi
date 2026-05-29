import html

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from sklearn.neighbors import NearestNeighbors


@st.cache_resource
def load_data():
    df = pd.read_csv("music_module/filtered_track_df.csv")
    df["genres"] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df


genre_names = ["Dance Pop", "Electronic", "Electropop", "Hip Hop", "Jazz", "K-pop", "Latin", "Pop", "Pop Rap", "R&B", "Rock"]
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]
feature_labels = {
    "acousticness": "Acousticness",
    "danceability": "Danceability",
    "energy": "Energy",
    "instrumentalness": "Instrumentalness",
    "valence": "Valence",
    "tempo": "Tempo",
}
feature_scales = {
    "acousticness": 1.0,
    "danceability": 1.0,
    "energy": 1.0,
    "instrumentalness": 1.0,
    "valence": 1.0,
    "tempo": 244.0,
}
genre_label_map = {name.lower(): name for name in genre_names}

exploded_track_df = load_data()

emotion_presets = {
    "Angry": {"acousticness": 0.2, "danceability": 0.6, "energy": 0.8, "instrumentalness": 0.1, "valence": 0.3, "tempo": 140.0},
    "Disgust": {"acousticness": 0.5, "danceability": 0.3, "energy": 0.4, "instrumentalness": 0.2, "valence": 0.2, "tempo": 100.0},
    "Fear": {"acousticness": 0.7, "danceability": 0.3, "energy": 0.2, "instrumentalness": 0.5, "valence": 0.1, "tempo": 90.0},
    "Happy": {"acousticness": 0.4, "danceability": 0.8, "energy": 0.7, "instrumentalness": 0.1, "valence": 0.9, "tempo": 120.0},
    "Sad": {"acousticness": 0.8, "danceability": 0.2, "energy": 0.2, "instrumentalness": 0.3, "valence": 0.1, "tempo": 80.0},
    "Surprise": {"acousticness": 0.4, "danceability": 0.7, "energy": 0.6, "instrumentalness": 0.2, "valence": 0.7, "tempo": 130.0},
    "Neutral": {"acousticness": 0.5, "danceability": 0.5, "energy": 0.5, "instrumentalness": 0.2, "valence": 0.5, "tempo": 110.0},
}


def has_value(value):
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    text = str(value).strip()
    return text not in {"", "nan", "None"}


def coerce_float(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(number):
        return default
    return number


def clean_genre_label(value):
    if not has_value(value):
        return ""
    text = str(value).strip().lower()
    return genre_label_map.get(text, text.title())


def trim_label(value, limit=34):
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def join_phrases(phrases):
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_data = exploded_track_df[
        (exploded_track_df["genres"] == genre)
        & (exploded_track_df["release_year"] >= start_year)
        & (exploded_track_df["release_year"] <= end_year)
    ]
    genre_data = genre_data.sort_values(by="popularity", ascending=False)[:500]

    if genre_data.empty:
        return [], np.empty((0, len(audio_feats))), [], np.array([])

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    distances, neighbor_indices = neigh.kneighbors([test_feat], n_neighbors=len(genre_data))
    neighbor_indices = neighbor_indices[0]
    distances = distances[0]
    ordered_tracks = genre_data.iloc[neighbor_indices]

    uris = ordered_tracks["uri"].tolist()
    audios = ordered_tracks[audio_feats].to_numpy()
    details = ordered_tracks[
        [
            "name",
            "artists_name",
            "release_year",
            "popularity",
            "preview_url",
            "playlist",
            "genres",
            "release_date",
            "duration_ms",
        ]
        + audio_feats
    ].to_dict("records")
    return uris, audios, details, distances


def render_header(selected_emotion, recommendation_context):
    source = recommendation_context.get("source")
    subtitle = f"Pop first, shaped by {selected_emotion}."
    if has_value(source):
        subtitle = f"Pop first, shaped by {selected_emotion} from {source}."
    st.markdown(
        f"""
        <div class="music-header">
            <div>
                <h1 class="music-title">EXPLORE Mix</h1>
                <p class="music-subtitle">{html.escape(subtitle)}</p>
            </div>
            <div class="mood-chip">Mood: {html.escape(selected_emotion)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def spotify_embed(uri):
    return f"""
    <iframe
        src="https://open.spotify.com/embed/track/{uri}"
        width="100%"
        height="352"
        frameborder="0"
        allowtransparency="true"
        allow="encrypted-media">
    </iframe>
    """


def strongest_target_features(target_profile, limit=3):
    neutral_profile = emotion_presets["Neutral"]
    ranked = sorted(
        audio_feats,
        key=lambda feature: abs(target_profile[feature] - neutral_profile[feature]) / feature_scales[feature],
        reverse=True,
    )
    return ranked[:limit]


def profile_phrase(feature, value):
    if feature == "acousticness":
        return "warmer acoustic texture" if value >= 0.58 else "cleaner produced edge"
    if feature == "danceability":
        return "dance-forward groove" if value >= 0.6 else "steadier laid-back flow"
    if feature == "energy":
        return "higher energy" if value >= 0.58 else "softer energy"
    if feature == "instrumentalness":
        return "more instrumental space" if value >= 0.26 else "vocal-forward feel"
    if feature == "valence":
        return "brighter mood" if value >= 0.58 else "more introspective tone"
    if feature == "tempo":
        return "faster pace" if value >= 118 else "slower pace"
    return feature_labels[feature]


def describe_target_profile(target_profile):
    phrases = [profile_phrase(feature, target_profile[feature]) for feature in strongest_target_features(target_profile)]
    return join_phrases(phrases)


def render_chip_row(items, class_name):
    return "".join(
        f'<span class="{class_name}">{html.escape(str(label))}</span>'
        for label in items
        if has_value(label)
    )


def render_recommendation_story(recommendation_context, selected_emotion, genre, start_year, end_year, target_profile):
    source = recommendation_context.get("source", "Emotion input")
    confidence = coerce_float(recommendation_context.get("confidence"))
    top_moods = recommendation_context.get("top_emotions", [])
    mood_chips = render_chip_row(
        [f'{item["emotion"]} {item["probability"]:.0%}' for item in top_moods[:3]],
        "info-chip",
    )
    tuning_summary = describe_target_profile(target_profile)
    input_excerpt = recommendation_context.get("input_excerpt")

    if confidence > 0:
        copy = (
            f'EXPLORE heard <strong>{html.escape(selected_emotion)}</strong> from <strong>{html.escape(source)}</strong> '
            f'at <strong>{confidence:.1%}</strong>, defaulted to <strong>{html.escape(genre)}</strong>, '
            f'and searched tracks from <strong>{start_year}</strong> to <strong>{end_year}</strong> '
            f'that stay closest to your current sound target: <strong>{html.escape(tuning_summary)}</strong>.'
        )
    else:
        copy = (
            f'EXPLORE is using <strong>{html.escape(selected_emotion)}</strong> as the starting mood, '
            f'searching <strong>{html.escape(genre)}</strong> tracks from <strong>{start_year}</strong> to <strong>{end_year}</strong> '
            f'that stay closest to your current sound target: <strong>{html.escape(tuning_summary)}</strong>.'
        )

    chips = [
        f"Source: {source}",
        f"Mood: {selected_emotion}",
        f"Genre: {genre}",
        f"Years: {start_year}-{end_year}",
    ]

    st.markdown(
        f"""
        <div class="explain-panel">
            <div class="explain-kicker">Recommendation story</div>
            <div class="explain-title">Why these songs showed up</div>
            <p class="explain-copy">{copy}</p>
            <div class="info-chip-row">
                {render_chip_row(chips, "info-chip info-chip--strong")}
                {mood_chips}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if has_value(input_excerpt):
        st.caption(f'Input: "{input_excerpt}"')


def render_mixed_breakdown(breakdown):
    if not breakdown:
        return

    text_summary = breakdown.get("text_summary", {})
    visual_summary = breakdown.get("visual_summary", {})
    final_summary = breakdown.get("final_summary", {})
    visual_mode = breakdown.get("visual_mode", "Visual")
    text_weight = round(coerce_float(breakdown.get("text_weight")) * 100)
    visual_weight = round(coerce_float(breakdown.get("visual_weight")) * 100)

    def summary_block(label, summary, meta_text):
        mood = html.escape(str(summary.get("dominant_emotion", "Neutral")))
        confidence = coerce_float(summary.get("confidence"))
        return (
            f'<div class="mix-card">'
            f'<div class="mix-label">{html.escape(label)}</div>'
            f'<div class="mix-emotion">{mood}</div>'
            f'<div class="mix-meta">{confidence:.1%} confidence</div>'
            f'<div class="mix-note">{html.escape(meta_text)}</div>'
            f"</div>"
        )

    summary_cards = "".join(
        [
            summary_block("Text", text_summary, f"{text_weight}% influence"),
            summary_block(visual_mode, visual_summary, f"{visual_weight}% influence"),
            summary_block("Final mix", final_summary, "Used for recommendations"),
        ]
    )

    st.markdown(
        (
            f'<div class="mix-breakdown">'
            f'<div class="mix-breakdown-kicker">Mixed-mode breakdown</div>'
            f'<div class="mix-breakdown-copy">'
            f'Text contributes <strong>{text_weight}%</strong> and {html.escape(visual_mode)} contributes '
            f'<strong>{visual_weight}%</strong> before the final mood is chosen.'
            f"</div>"
            f'<div class="mix-grid">{summary_cards}</div>'
            f"</div>"
        ),
        unsafe_allow_html=True,
    )

    text_excerpt = breakdown.get("text_excerpt")
    if has_value(text_excerpt):
        st.caption(f'Mixed prompt: "{text_excerpt}"')


def render_song_meta(track, rank, target_profile):
    title = html.escape(str(track.get("name") or "Recommended track"))
    artist = html.escape(str(track.get("artists_name") or "Unknown artist"))
    year = html.escape(str(track.get("release_year") or ""))
    genre_label = clean_genre_label(track.get("genres"))
    popularity = round(coerce_float(track.get("popularity")))
    playlist = trim_label(track.get("playlist"), limit=28) if has_value(track.get("playlist")) else ""

    chips = [f"Match #{rank}", f"Popularity {popularity}"]
    if genre_label:
        chips.append(genre_label)
    if playlist:
        chips.append(playlist)

    reason = build_track_reason(track, target_profile)
    st.markdown(
        f"""
        <div class="song-meta">
            <div class="song-title">{title}</div>
            <div class="song-subtitle">{artist} {year}</div>
            <div class="song-chip-row">{render_chip_row(chips, "song-chip")}</div>
            <div class="song-note">{html.escape(reason)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_track_reason(track, target_profile):
    priority_features = strongest_target_features(target_profile, limit=4)
    closest_features = sorted(
        priority_features,
        key=lambda feature: abs(coerce_float(track.get(feature)) - target_profile[feature]) / feature_scales[feature],
    )[:2]
    phrases = [profile_phrase(feature, target_profile[feature]) for feature in closest_features]
    return f"Why it fits: this track stays close to your {join_phrases(phrases)} settings."


def render_preview_player(track):
    preview_url = track.get("preview_url")
    if has_value(preview_url) and str(preview_url).startswith("http"):
        st.audio(str(preview_url), format="audio/mp3")
    else:
        st.caption("30-second preview unavailable for this track.")


def render_showing_counter(page_start, page_end, total_tracks):
    st.markdown(
        f"""
        <div class="showing-counter">
            <div class="showing-label">Showing</div>
            <div class="showing-value">{page_start}-{page_end}</div>
            <div class="showing-total">{total_tracks} total</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_audio_chart(audio, chart_key):
    df = pd.DataFrame(dict(r=audio[:5], theta=audio_feats[:5]))
    fig = px.line_polar(df, r="r", theta="theta", line_close=True)
    fig.update_traces(
        line_color="#667635",
        fill="toself",
        fillcolor="rgba(102, 118, 53, 0.24)",
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(248, 244, 232, 0)",
        plot_bgcolor="rgba(248, 244, 232, 0)",
        font_color="#1f2716",
        polar=dict(
            bgcolor="#f8f4e8",
            radialaxis=dict(gridcolor="rgba(70, 85, 40, 0.24)", tickfont=dict(size=10)),
            angularaxis=dict(gridcolor="rgba(70, 85, 40, 0.18)"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_track_stats(track):
    tempo = round(coerce_float(track.get("tempo")))
    danceability = coerce_float(track.get("danceability"))
    energy = coerce_float(track.get("energy"))
    valence = coerce_float(track.get("valence"))
    st.markdown(
        f"""
        <div class="track-stats">
            Tempo {tempo} BPM
            <span>•</span>
            Danceability {danceability:.2f}
            <span>•</span>
            Energy {energy:.2f}
            <span>•</span>
            Valence {valence:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_controls(selected_emotion):
    if st.session_state.get("selected_genre") not in genre_names:
        st.session_state.selected_genre = "Pop"

    preset = emotion_presets.get(selected_emotion, emotion_presets["Neutral"])

    st.markdown('<div class="control-heading">Genre</div>', unsafe_allow_html=True)
    genre = st.selectbox("Change genre", genre_names, key="selected_genre")

    st.markdown('<div class="control-heading">Components</div>', unsafe_allow_html=True)
    start_year, end_year = st.slider("Release years", 1990, 2019, (2015, 2019))
    acousticness = st.slider("Acousticness", 0.0, 1.0, preset["acousticness"])
    danceability = st.slider("Danceability", 0.0, 1.0, preset["danceability"])
    energy = st.slider("Energy", 0.0, 1.0, preset["energy"])
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, preset["instrumentalness"])
    valence = st.slider("Valence", 0.0, 1.0, preset["valence"])
    tempo = st.slider("Tempo", 0.0, 244.0, preset["tempo"])

    if st.button("Back to EXPLORE"):
        st.session_state.page = "emotion"
        st.rerun()

    test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
    return genre, start_year, end_year, test_feat


def run():
    selected_emotion = st.session_state.get("detected_emotion", "Neutral")
    selected_emotion = selected_emotion if selected_emotion in emotion_presets else "Neutral"
    recommendation_context = st.session_state.get("recommendation_context", {})

    render_header(selected_emotion, recommendation_context)

    music_col, controls_col = st.columns([3.2, 1.1], gap="large")

    with controls_col:
        genre, start_year, end_year, test_feat = render_controls(selected_emotion)

    current_inputs = [genre, start_year, end_year] + test_feat + [selected_emotion]
    if current_inputs != st.session_state.get("previous_inputs"):
        st.session_state["start_track_i"] = 0
        st.session_state["previous_inputs"] = current_inputs

    uris, audios, details, distances = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)
    tracks_per_page = 6
    tracks = [spotify_embed(uri) for uri in uris]
    target_profile = dict(zip(audio_feats, test_feat))

    with music_col:
        if not tracks:
            st.warning("No songs found for this combination. Try a wider year range or a different genre.")
            return

        render_recommendation_story(
            recommendation_context,
            selected_emotion,
            genre,
            start_year,
            end_year,
            target_profile,
        )

        if recommendation_context.get("source") == "Mixed":
            render_mixed_breakdown(recommendation_context.get("mixed_breakdown"))

        top_action_col, count_col = st.columns([1, 1])
        with top_action_col:
            if st.button("Recommend More Songs", type="primary"):
                if st.session_state["start_track_i"] + tracks_per_page < len(tracks):
                    st.session_state["start_track_i"] += tracks_per_page
                else:
                    st.session_state["start_track_i"] = 0
        with count_col:
            page_start = st.session_state["start_track_i"] + 1
            page_end = min(st.session_state["start_track_i"] + tracks_per_page, len(tracks))
            render_showing_counter(page_start, page_end, len(tracks))

        start = st.session_state["start_track_i"]
        end = start + tracks_per_page
        current_tracks = tracks[start:end]
        current_audios = audios[start:end]
        current_details = details[start:end]
        current_distances = distances[start:end]
        track_cols = st.columns(2)

        for i, (track, audio, detail, distance) in enumerate(zip(current_tracks, current_audios, current_details, current_distances)):
            with track_cols[i % 2]:
                rank = start + i + 1
                render_song_meta(detail, rank, target_profile)
                render_preview_player(detail)
                components.html(track, height=370)
                with st.expander("Audio profile and fit"):
                    st.caption(f"Nearest-neighbor distance inside the selected pool: {coerce_float(distance):.2f}")
                    render_track_stats(detail)
                    render_audio_chart(audio, chart_key=f"audio_profile_{rank}")
