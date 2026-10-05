# Fake-news-detector
# ============================================================
# AI-BASED FAKE NEWS DETECTION SYSTEM
# ============================================================

import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# TRAINING DATA
# ============================================================

news_data = [

    # ---------------- REAL NEWS ----------------
    "The government announced a new education program for students.",
    "The university announced the dates for the upcoming semester examinations.",
    "The weather department issued a warning about heavy rainfall.",
    "Scientists published a study about climate change in a research journal.",
    "The company announced its quarterly financial results.",
    "The health department advised citizens to maintain proper hygiene.",
    "The central bank announced changes to its monetary policy.",
    "The school announced a holiday for students on Monday.",
    "Researchers developed a new method for detecting diseases.",
    "The government launched a digital service for citizens.",

    # ---------------- FAKE NEWS ----------------
    "Scientists confirm that humans can live without sleep forever.",
    "Government will give every citizen one crore rupees tomorrow.",
    "Drinking hot water can make people live for two hundred years.",
    "Scientists discover that the moon is made completely of gold.",
    "A smartphone can charge itself using moonlight.",
    "Eating chocolate guarantees that you will never get sick.",
    "The government announced free cars for every student.",
    "Researchers prove that humans can survive without food for years.",
    "A magical fruit has been discovered that makes people immortal.",
    "Scientists say that looking at the sun directly improves eyesight."
]

# Labels:
# 1 = Real
# 0 = Fake

labels = [
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
]


# ============================================================
# TEXT CONVERSION USING TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X = vectorizer.fit_transform(news_data)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

model = LogisticRegression()

model.fit(X, labels)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def detect_news():

    text = news_text.get("1.0", tk.END).strip()

    if text == "":
        messagebox.showwarning(
            "Warning",
            "Please enter a news headline or article."
        )
        return

    # Convert user text into TF-IDF features
    text_vector = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(text_vector)[0]

    # Probability
    probability = model.predict_proba(text_vector)[0]

    confidence = max(probability) * 100

    # Display result
    if prediction == 1:

        result_label.config(
            text="REAL NEWS",
            fg="green"
        )

        confidence_label.config(
            text=f"Confidence: {confidence:.2f}%"
        )

    else:

        result_label.config(
            text="FAKE NEWS",
            fg="red"
        )

        confidence_label.config(
            text=f"Confidence: {confidence:.2f}%"
        )


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_text():

    news_text.delete("1.0", tk.END)

    result_label.config(
        text="RESULT",
        fg="black"
    )

    confidence_label.config(
        text="Confidence: --"
    )


# ============================================================
# GUI
# ============================================================

window = tk.Tk()

window.title("AI-Based Fake News Detection System")

window.geometry("800x600")

window.configure(bg="#f2f2f2")


# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(
    window,
    text="AI-Based Fake News Detection System",
    font=("Arial", 24, "bold"),
    bg="#f2f2f2"
)

title_label.pack(pady=25)


# ============================================================
# DESCRIPTION
# ============================================================

description = tk.Label(
    window,
    text="Enter a news headline or article below",
    font=("Arial", 14),
    bg="#f2f2f2"
)

description.pack(pady=10)


# ============================================================
# TEXT BOX
# ============================================================

news_text = tk.Text(
    window,
    height=10,
    width=80,
    font=("Arial", 12),
    wrap=tk.WORD
)

news_text.pack(pady=15)


# ============================================================
# BUTTON FRAME
# ============================================================

button_frame = tk.Frame(
    window,
    bg="#f2f2f2"
)

button_frame.pack(pady=10)


# Detect Button

detect_button = tk.Button(
    button_frame,
    text="CHECK NEWS",
    font=("Arial", 14, "bold"),
    bg="#4CAF50",
    fg="white",
    width=15,
    height=2,
    command=detect_news
)

detect_button.grid(
    row=0,
    column=0,
    padx=10
)


# Clear Button

clear_button = tk.Button(
    button_frame,
    text="CLEAR",
    font=("Arial", 14, "bold"),
    bg="#555555",
    fg="white",
    width=15,
    height=2,
    command=clear_text
)

clear_button.grid(
    row=0,
    column=1,
    padx=10
)


# ============================================================
# RESULT
# ============================================================

result_title = tk.Label(
    window,
    text="Prediction:",
    font=("Arial", 16, "bold"),
    bg="#f2f2f2"
)

result_title.pack(pady=(25, 5))


result_label = tk.Label(
    window,
    text="RESULT",
    font=("Arial", 24, "bold"),
    bg="#f2f2f2"
)

result_label.pack()


# ============================================================
# CONFIDENCE
# ============================================================

confidence_label = tk.Label(
    window,
    text="Confidence: --",
    font=("Arial", 14),
    bg="#f2f2f2"
)

confidence_label.pack(pady=10)


# ============================================================
# FOOTER
# ============================================================

footer = tk.Label(
    window,
    text="Machine Learning: TF-IDF + Logistic Regression",
    font=("Arial", 10),
    bg="#f2f2f2"
)

footer.pack(side=tk.BOTTOM, pady=15)


# ============================================================
# START APPLICATION
# ============================================================

print("Fake News Detection System Started!")

window.mainloop()
