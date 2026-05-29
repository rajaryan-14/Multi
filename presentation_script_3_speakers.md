# Presentation Script for 3 People

## Team Roles

- **Shubhanshu Gupta**: system architecture, CNN, visual pipeline
- **Raj Aryan Singh**: text emotion understanding, multimodal logic
- **Vaanya Sharma**: music recommendation engine, integration, UI

## Suggested total presentation time

8 to 10 minutes

## Delivery style

- Speak slowly.
- Do not read every bullet.
- Use the slide as support, not as the script itself.
- End each slide with a clean handoff sentence.

## Slide-by-slide script

### Slide 1 - Title
**Speaker: Shubhanshu**

"Good morning. We are presenting our major project titled 'Multi-Modal Emotion Recognition and Music Recommendation System.' The project was developed in the Department of Electronics and Communication Engineering at JIIT Noida under the guidance of Prof. Richa Gupta. Our team members are Shubhanshu Gupta, Raj Aryan Singh, and Vaanya Sharma."

Handoff:

"I'll begin with the problem we wanted to solve."

---

### Slide 2 - Problem Statement
**Speaker: Shubhanshu**

"The core problem is that music recommendation platforms are personalized, but they are usually not emotion-aware in real time. Users still have to manually choose songs depending on their mood, and most platforms rely on listening history rather than current emotional state. We felt there was a clear need for a system that can detect human emotion in real time and automatically recommend music that matches that mood."

"This is important because music directly affects emotional state, focus, and well-being. So a real-time emotion-aware system can improve user experience and make recommendations more meaningful."

Handoff:

"Before this semester's work, we already had a baseline system. Raj will briefly explain that."

---

### Slide 3 - Semester 1 Recap: The Baseline
**Speaker: Raj**

"In our baseline, we explored two major streams. For text, we focused on semantic understanding so that the system could understand the emotion behind what the user writes. For images, we used face detection followed by facial emotion classification, so the system could infer emotion from facial expressions."

"However, one challenge with frame-by-frame visual prediction was instability. Independent image frames can create flickering results because emotion is dynamic, not purely static. That motivated us to think beyond single snapshots and move toward a more integrated system."

Handoff:

"This semester, we focused on turning these separate ideas into a connected recommendation pipeline."

---

### Slide 4 - Work Done This Semester
**Speaker: Shubhanshu**

"This semester, our focus was integration and usability. We connected emotion recognition to a music recommendation pipeline, expanded input modes, improved the interface, and built a system that can move from emotion detection to song suggestion in a practical way."

"We also worked on making the demo more interactive through live camera and mixed-mode input, where text and visual signals can both contribute to the final mood."

Handoff:

"Vaanya will now explain how the recommendation engine works after an emotion is detected."

---

### Slide 5 - Integrated Song Recommendation
**Speaker: Vaanya**

"Once the system detects an emotion, the next task is to convert that emotion into a recommendation. Our flow is: detect emotion, map it to a target music vector, perform a nearest-neighbor search, and then return songs that best match that target."

"For example, if the detected emotion is happy, the target vector will favor higher valence and higher energy. If the emotion is sad, the target vector shifts toward lower valence and more subdued characteristics."

"As the emotion input changes, the target vector also changes, so the recommendations adapt accordingly."

Handoff:

"To understand this better, the next slide shows the music features we use."

---

### Slide 6 - Music Feature Parameters
**Speaker: Vaanya**

"We use six core audio features for recommendation. Energy reflects intensity, valence reflects positivity, danceability reflects rhythmic suitability, tempo captures speed in beats per minute, acousticness reflects how organic or unplugged a song feels, and instrumentalness indicates how vocal-heavy or instrument-heavy a track is."

"These six features together give us a compact but expressive representation of the emotional feel of a song."

Handoff:

"Now I'll explain how we use these features to retrieve songs."

---

### Slide 7 - KNN Recommendation Logic
**Speaker: Vaanya**

"In our recommendation logic, each song is treated as a point in a six-dimensional feature space. The detected emotion is converted into a target query point in the same space. Then we compute distance between the query point and all candidate songs and retrieve the closest matches."

"This is essentially a nearest-neighbor recommendation approach. Songs with minimum distance are considered the best emotional match."

Handoff:

"The natural question is why we chose this approach, and that's what the next slide answers."

---

### Slide 8 - Why KNN?
**Speaker: Vaanya**

"We chose this method because it is simple, interpretable, and effective for feature-based retrieval. It does not require a heavy retraining phase for every new query, and it handles the cold-start problem well because it can recommend songs even for a new user based only on current mood."

"It is also easy to justify technically, because we already have a target feature vector and want to find the nearest songs to it."

Handoff:

"Now Raj will explain how we combine different emotion input streams in the system."

---

### Slide 9 - Visual or Demo Slide
**Speaker: Raj**

"This slide represents the integrated user-facing view of the system. The important idea is that the project is not only a backend model pipeline, but also a complete interactive application where users can choose their input mode and move directly from emotion recognition to song recommendation."

Handoff:

"The real strength of the system comes from how different modalities are brought together."

---

### Slide 10 - Multimodal Integration Logic
**Speaker: Raj**

"Our system is multimodal because it can process multiple kinds of emotional input. One stream handles textual input, and another handles visual input such as image, video, or live camera. Each stream produces its own emotion probabilities independently."

"These outputs are then combined through weighted fusion. This makes the system more robust, because if one modality is weak or ambiguous, the other can still contribute meaningfully."

Handoff:

"The next slide shows how this fusion works in a more concrete way."

---

### Slide 11 - Weighted Decision Fusion
**Speaker: Raj**

"In weighted fusion, we allow the contribution of each modality to be controlled. For example, if text is given weight 0.7 and video is given weight 0.3, then the final probability for each emotion is a weighted combination of the two predictions."

"This is useful because sometimes text may reveal internal emotion better, while visual input may reveal external expression better. Instead of forcing a single modality to dominate, we let both contribute."

Handoff:

"Shubhanshu will now explain the video side and why temporal behavior matters."

---

### Slide 12 - Visual or Results Slide
**Speaker: Shubhanshu**

"This stage highlights the visual side of the project and the movement from single-frame analysis toward more realistic temporal understanding. The key idea is that emotion is not always best captured in one image. It often evolves over time."

Handoff:

"That leads directly to our video training concept."

---

### Slide 13 - Video Training: CNN + LSTM
**Speaker: Shubhanshu**

"Conceptually, this slide explains the difference between spatial and temporal learning. A CNN is good at extracting spatial features from individual frames, such as mouth shape, eyebrow movement, and eye-region patterns. An LSTM is useful when we want to model how these features evolve over time."

"This matters because emotion is dynamic. A genuine smile, a frown, or a surprised reaction often becomes clearer across a short sequence than in a single snapshot."

Safe line if questioned:

"In the working integrated demo, temporal behavior is handled through multi-frame analysis and aggregation for practical deployment, while this slide reflects the temporal modeling concept we studied."

Handoff:

"The next slide makes the spatial versus temporal distinction clearer."

---

### Slide 14 - Spatial vs Temporal Logic
**Speaker: Shubhanshu**

"Spatial information tells us what is visible in a single frame: the face structure and expression at that moment. Temporal information tells us how the expression changes across a sequence. That helps distinguish a brief facial change from a more stable emotional pattern."

"In sequence-based thinking, overlapping windows allow multiple short segments to be analyzed rather than relying on one isolated frame."

Handoff:

"Vaanya will now conclude with the system-level takeaway."

---

### Slide 15 - Final System / Demo / Result Slide
**Speaker: Vaanya**

"At this point, the complete value of the project becomes clear. The user can provide emotion through different modalities, the system detects the dominant mood, and then it generates music recommendations that are explainable and adjustable. This makes the system practical, interactive, and easy to demonstrate."

"The project combines emotion recognition, multimodal fusion, recommendation logic, and a full interface into one end-to-end application."

Handoff:

"I'll close with a short summary."

---

### Slide 16 - Thank You
**Speaker: Vaanya**

"To conclude, our project aims to make music recommendation more emotionally intelligent by combining real-time emotion recognition with feature-based song retrieval. It brings together text, vision, multimodal fusion, and recommendation into a working user-facing system. Thank you, and we are ready for your questions."

## Fast backup version if time is cut short

### Speaker 1

- Slide 1
- Slide 2
- Slide 4

### Speaker 2

- Slide 3
- Slide 10
- Slide 11

### Speaker 3

- Slide 5
- Slide 6
- Slide 7
- Slide 8
- Slide 16

Use slide 13 and 14 only if the panel wants deeper technical detail.

## Quick handoff lines

- "With that context, I'll hand over to Raj for the baseline and multimodal logic."
- "I'll hand over to Vaanya to explain the recommendation engine."
- "I'll hand back to Shubhanshu for the visual and temporal side."
- "I'll close with the overall takeaway."

## Very short 10-second self-intros if needed

### Shubhanshu
"I mainly worked on system architecture and the visual emotion pipeline."

### Raj
"I mainly worked on the text understanding and fusion side."

### Vaanya
"I mainly worked on the recommendation engine and integration of the final application."

## Final rehearsal advice

- Keep answers direct.
- Do not argue with the slide if a teacher probes a detail; clarify calmly.
- If asked a tricky implementation question, say what the concept was and what the working prototype currently uses.
- End strong: this is an end-to-end AI system, not just a model in isolation.
