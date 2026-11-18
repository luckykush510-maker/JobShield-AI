from flask import Flask, render_template, request
import joblib
import pandas as pd
import re


# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)


# ============================================================
# Load ML Model and TF-IDF Vectorizer
# ============================================================

model = joblib.load("model/fake_job_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# ============================================================
# Load Reported Scam Database
# ============================================================

reported_scams = pd.read_csv(
    "database/reported_scams.csv"
)


# ============================================================
# Find Suspicious Sentence
# ============================================================

def find_suspicious_sentence(text, keywords):

    if not text:
        return None

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_lower = sentence.lower()

        for keyword in keywords:

            if keyword.lower() in sentence_lower:
                return sentence

    return None


# ============================================================
# Find ML Suspicious Sentences
# ============================================================

def find_ml_suspicious_sentences(offer_letter):

    suspicious_sentences = []

    if not offer_letter:
        return suspicious_sentences

    sentences = re.split(
        r'(?<=[.!?])\s+',
        offer_letter
    )

    suspicious_keywords = [

        "registration fee",
        "registration fees",

        "training fee",
        "training fees",

        "security deposit",
        "processing fee",
        "joining fee",

        "urgent payment",
        "upfront payment",

        "immediate joining",

        "bank details",
        "bank information",

        "aadhaar",
        "aadhar",

        "otp",
        "password",

        "guaranteed selection",
        "100% selection",

        "no interview",
        "without interview",
        "without an interview"
    ]

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_lower = sentence.lower()


        # ----------------------------------------------------
        # Ignore sentences which clearly say NO fee
        # ----------------------------------------------------

        if (
            "no registration fee" in sentence_lower
            or "no registration fees" in sentence_lower
            or "no training fee" in sentence_lower
            or "no training fees" in sentence_lower
            or "no security deposit" in sentence_lower
        ):
            continue


        # ----------------------------------------------------
        # First check whether suspicious keyword exists
        # ----------------------------------------------------

        found_keyword = False

        for keyword in suspicious_keywords:

            if keyword.lower() in sentence_lower:

                found_keyword = True
                break

        if not found_keyword:
            continue


        # ----------------------------------------------------
        # ML Prediction for this sentence
        # ----------------------------------------------------

        try:

            sentence_input = vectorizer.transform(
                [sentence]
            )

            prediction = model.predict(
                sentence_input
            )[0]

            print(
                "Sentence:",
                sentence
            )

            print(
                "Prediction:",
                prediction
            )


            if str(prediction).lower() in [
                "fake",
                "1",
                "1.0"
            ]:

                suspicious_sentences.append(
                    sentence
                )

        except Exception as e:

            print(
                "Sentence ML Error:",
                e
            )


    return suspicious_sentences


# ============================================================
# Home Page
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# Check Job
# ============================================================

@app.route(
    "/check",
    methods=["POST"]
)
def check_job():

    # ========================================================
    # Get Form Data
    # ========================================================

    job_title = request.form.get(
        "job_title",
        ""
    )

    company = request.form.get(
        "company",
        ""
    )

    description = request.form.get(
        "description",
        ""
    )

    offer_letter = request.form.get(
        "offer_letter",
        ""
    )

    salary = request.form.get(
        "salary",
        ""
    )

    email = request.form.get(
        "email",
        ""
    )

    phone = request.form.get(
        "phone",
        ""
    )
    # Check whether all fields are empty
    if not any([
    job_title.strip(),
    company.strip(),
    description.strip(),
    offer_letter.strip(),
    salary.strip(),
    email.strip(),
    phone.strip()
    ]):
     return render_template(
        "index.html",
        error="Please enter at least one job detail before checking."
    )

    # ========================================================
    # Clean Values
    # ========================================================

    job_title = job_title.strip()
    company = company.strip()
    description = description.strip()
    offer_letter = offer_letter.strip()
    salary = salary.strip()
    email = email.strip()
    phone = phone.strip()


    # ========================================================
    # Debugging
    # ========================================================

    print(
        "OFFER LETTER:",
        offer_letter
    )

    print(
        "DESCRIPTION:",
        description
    )

    print(
        "EMAIL:",
        email
    )

    print(
        "PHONE:",
        phone
    )


    # ========================================================
    # Risk Score and Reasons
    # ========================================================

    risk_score = 0
    reasons = []


    # ========================================================
    # Check Email in Reported Scam Database
    # ========================================================

    email_match = pd.DataFrame()


    if (
        email
        and "email" in reported_scams.columns
    ):

        email_match = reported_scams[
            reported_scams["email"]
            .astype(str)
            .str.lower()
            .str.strip()
            == email.lower().strip()
        ]


    if not email_match.empty:

        risk_score += 20

        reasons.append(
            "This email was found in the reported scam database."
        )


    # ========================================================
    # Check Phone in Reported Scam Database
    # ========================================================

    phone_match = pd.DataFrame()


    if (
        phone
        and "phone" in reported_scams.columns
    ):

        phone_match = reported_scams[
            reported_scams["phone"]
            .astype(str)
            .str.strip()
            == phone.strip()
        ]


    if not phone_match.empty:

        risk_score += 20

        reasons.append(
            "This phone number was found in the reported scam database."
        )


    # ========================================================
    # Check Company in Reported Scam Database
    # ========================================================

    company_match = pd.DataFrame()


    if (
        company
        and "company" in reported_scams.columns
    ):

        company_match = reported_scams[
            reported_scams["company"]
            .astype(str)
            .str.lower()
            .str.strip()
            == company.lower().strip()
        ]


    if not company_match.empty:

        risk_score += 20

        matched_company = company_match.iloc[0]["company"]

        if "reason" in company_match.columns:

            database_reason = company_match.iloc[0]["reason"]

        else:

            database_reason = "Reported in the scam database."


        reasons.append(
            f'Company "{matched_company}" was found in the '
            f'reported scam database. Reason: {database_reason}'
        )


    # ========================================================
    # Create Text for Keyword Checking
    # ========================================================

    text = (
        job_title
        + " "
        + company
        + " "
        + description
        + " "
        + salary
        + " "
        + offer_letter
    ).lower()


    print(
        "TEXT:",
        text
    )


    # ========================================================
    # ML Prediction - Sentence Level
    # ========================================================

    suspicious_sentences = []

    try:

        suspicious_sentences = find_ml_suspicious_sentences(
            offer_letter
        )

        print(
            "Suspicious Sentences:",
            suspicious_sentences
        )


        if suspicious_sentences:

            risk_score += 40

            for sentence in suspicious_sentences:

                reasons.append(
                    f'Offer Letter suspicious line: '
                    f'"{sentence}" → ML model classified '
                    f'this sentence as suspicious.'
                )

    except Exception as e:

        print(
            "ML Error:",
            e
        )


    # ========================================================
    # REGISTRATION FEE
    # ========================================================

    if (
        (
            "registration fee" in text
            or "registration fees" in text
        )
        and "no registration fee" not in text
        and "no registration fees" not in text
    ):

        risk_score += 25

        matched_sentence = find_suspicious_sentence(
            offer_letter,
            [
                "registration fee",
                "registration fees"
            ]
        )


        if matched_sentence:

            reasons.append(
                f'Offer Letter: "{matched_sentence}" '
                f'→ Registration fee is mentioned.'
            )

        else:

            reasons.append(
                "Registration fee is mentioned."
            )


    # ========================================================
    # TRAINING FEE
    # ========================================================

    if (
        (
            "training fee" in text
            or "training fees" in text
        )
        and "no training fee" not in text
        and "no training fees" not in text
    ):

        risk_score += 20

        matched_sentence = find_suspicious_sentence(
            offer_letter,
            [
                "training fee",
                "training fees"
            ]
        )


        if matched_sentence:

            reasons.append(
                f'Offer Letter: "{matched_sentence}" '
                f'→ Training fee is mentioned.'
            )

        else:

            reasons.append(
                "Training fee is mentioned."
            )


    # ========================================================
    # SECURITY DEPOSIT
    # ========================================================

    if (
        "security deposit" in text
        and "no security deposit" not in text
    ):

        risk_score += 20

        matched_sentence = find_suspicious_sentence(
            offer_letter,
            [
                "security deposit"
            ]
        )


        if matched_sentence:

            reasons.append(
                f'Offer Letter: "{matched_sentence}" '
                f'→ Security deposit is mentioned.'
            )

        else:

            reasons.append(
                "Security deposit is mentioned."
            )


    # ========================================================
    # GUARANTEED SELECTION
    # ========================================================

    if (
        "guaranteed selection" in text
        or "100% selection" in text
    ):

        risk_score += 20

        matched_sentence = find_suspicious_sentence(
            offer_letter,
            [
                "guaranteed selection",
                "100% selection"
            ]
        )


        if matched_sentence:

            reasons.append(
                f'Offer Letter: "{matched_sentence}" '
                f'→ Guaranteed selection is mentioned.'
            )

        else:

            reasons.append(
                "Guaranteed selection is mentioned."
            )


    # ========================================================
    # HIGH SALARY
    # ========================================================

    if (
        "₹50,000" in salary
        or "₹50,000" in text
        or "₹60,000" in salary
        or "₹60,000" in text
        or "₹100,000" in salary
        or "₹100,000" in text
        or "50000" in salary
        or "60000" in salary
        or "100000" in salary
    ):

        risk_score += 15

        reasons.append(
            "The salary claim may be unusually high."
        )


    # ========================================================
    # SENSITIVE INFORMATION
    # ========================================================

    sensitive_keywords = [
        "bank details",
        "bank information",
        "aadhaar",
        "aadhar",
        "otp",
        "password"
    ]


    found_sensitive = False


    for keyword in sensitive_keywords:

        if keyword in text:

            found_sensitive = True
            break


    if found_sensitive:

        risk_score += 15

        matched_sentence = find_suspicious_sentence(
            offer_letter,
            sensitive_keywords
        )


        if matched_sentence:

            reasons.append(
                f'Offer Letter: "{matched_sentence}" '
                f'→ Sensitive personal or account information '
                f'is requested.'
            )

        else:

            reasons.append(
                "Sensitive personal or bank information is requested."
            )


    # ========================================================
    # URGENT JOINING
    # ========================================================

    if (
        "immediate joining" in text
        or "urgent joining" in text
    ):

        risk_score += 15

        matched_sentence = find_suspicious_sentence(
            offer_letter,
            [
                "immediate joining",
                "urgent joining"
            ]
        )


        if matched_sentence:

            reasons.append(
                f'Offer Letter: "{matched_sentence}" '
                f'→ Urgent joining is mentioned.'
            )

        else:

            reasons.append(
                "Urgent joining is mentioned."
            )


    # ========================================================
    # NO INTERVIEW
    # ========================================================

    if (
        "no interview" in text
        or "without interview" in text
        or "without an interview" in text
    ):

        risk_score += 15

        matched_sentence = find_suspicious_sentence(
            offer_letter,
            [
                "no interview",
                "without interview",
                "without an interview"
            ]
        )


        if matched_sentence:

            reasons.append(
                f'Offer Letter: "{matched_sentence}" '
                f'→ Selection without a proper interview '
                f'is mentioned.'
            )

        else:

            reasons.append(
                "Selection without a proper interview is mentioned."
            )


    # ========================================================
    # Maximum Risk Score
    # ========================================================

    risk_score = min(
        risk_score,
        100
    )


    # ========================================================
    # Final Status
    # ========================================================

    if risk_score >= 60:

        status = "High Risk"

    elif risk_score >= 30:

        status = "Suspicious"

    else:

        status = "Likely Genuine"


    # ========================================================
    # No Reasons
    # ========================================================

    if not reasons:

        reasons.append(
            "No major suspicious indicators were found."
        )


    # ========================================================
    # Result Page
    # ========================================================

    return render_template(
        "result.html",
        risk_score=risk_score,
        status=status,
        reasons=reasons,
        job_title=job_title,
        company=company,
        salary=salary,
        email=email,
        phone=phone,
        description=description,
        offer_letter=offer_letter
    )


# ============================================================
# Run Flask App
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )