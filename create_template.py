import pandas as pd

ideas = [
    {
        "idea_id": 1, "title": "MediScan AI",
        "description": "An AI mobile app that detects skin diseases from photos using computer vision. Users upload a photo and get instant diagnosis with treatment recommendations and nearby clinic suggestions.",
        "category": "Healthcare", "source": "Devpost",
        "expert_rank": 1, "expert_novelty": 4,
        "expert_feasibility": 3, "expert_impact": 4,
        "expert_presentation": 4, "advance": 1
    },
    {
        "idea_id": 2, "title": "EcoTrack",
        "description": "A platform tracking individual carbon footprint by analyzing purchase history, travel data, and energy usage. Provides personalized weekly tips and gamified challenges to reduce emissions.",
        "category": "Environment", "source": "Devpost",
        "expert_rank": 2, "expert_novelty": 3,
        "expert_feasibility": 4, "expert_impact": 4,
        "expert_presentation": 3, "advance": 1
    },
    {
        "idea_id": 3, "title": "StudyBuddy",
        "description": "An AI chatbot that creates personalized study plans based on learning style and exam schedule. Generates quizzes automatically, tracks daily progress, and adapts to weak areas.",
        "category": "Education", "source": "Devpost",
        "expert_rank": 3, "expert_novelty": 3,
        "expert_feasibility": 4, "expert_impact": 3,
        "expert_presentation": 3, "advance": 1
    },
    {
        "idea_id": 4, "title": "CropGuard",
        "description": "Drone-based AI system that scans agricultural fields and detects pest infestations and crop diseases early using multispectral imaging. Sends geo-tagged alerts to farmers with treatment recommendations.",
        "category": "Agriculture", "source": "Devpost",
        "expert_rank": 4, "expert_novelty": 4,
        "expert_feasibility": 3, "expert_impact": 4,
        "expert_presentation": 4, "advance": 1
    },
    {
        "idea_id": 5, "title": "SignSpeak",
        "description": "Real-time sign language to speech translator using phone camera and hand gesture recognition AI. Helps deaf and mute individuals communicate naturally in daily situations without an interpreter.",
        "category": "Accessibility", "source": "Devpost",
        "expert_rank": 5, "expert_novelty": 4,
        "expert_feasibility": 3, "expert_impact": 4,
        "expert_presentation": 4, "advance": 1
    },
    {
        "idea_id": 6, "title": "NeuroCoach",
        "description": "AI-powered brain training app for elderly patients showing early Alzheimer signs. Uses adaptive cognitive exercises, speech analysis, and daily mood tracking to slow cognitive decline.",
        "category": "Healthcare", "source": "Devpost",
        "expert_rank": 6, "expert_novelty": 4,
        "expert_feasibility": 3, "expert_impact": 4,
        "expert_presentation": 3, "advance": 1
    },
    {
        "idea_id": 7, "title": "FloodSense",
        "description": "IoT sensor network placed in flood-prone urban areas that predicts flooding 2 hours in advance using water level and rainfall data. Sends automated SMS alerts to residents and emergency services.",
        "category": "Disaster Management", "source": "College Hackathon",
        "expert_rank": 7, "expert_novelty": 3,
        "expert_feasibility": 4, "expert_impact": 4,
        "expert_presentation": 3, "advance": 1
    },
    {
        "idea_id": 8, "title": "LegalEase",
        "description": "An AI tool that simplifies legal documents into plain language for common citizens. Users upload any contract and receive a plain-English summary with key risk highlights and suggested questions for their lawyer.",
        "category": "Legal", "source": "Devpost",
        "expert_rank": 8, "expert_novelty": 3,
        "expert_feasibility": 3, "expert_impact": 3,
        "expert_presentation": 3, "advance": 1
    },
    {
        "idea_id": 9, "title": "SafeWalk",
        "description": "Women safety app using GPS to detect unusual stops or route deviations and automatically alerts emergency contacts. Includes a disguised panic button triggered by shaking the phone.",
        "category": "Safety", "source": "Devpost",
        "expert_rank": 9, "expert_novelty": 3,
        "expert_feasibility": 4, "expert_impact": 4,
        "expert_presentation": 3, "advance": 1
    },
    {
        "idea_id": 10, "title": "SkillBridge",
        "description": "Platform connecting rural youth with urban mentors for skill development through short video lessons and live Q&A. Works on low bandwidth with offline download support and vernacular language options.",
        "category": "Education", "source": "Devpost",
        "expert_rank": 10, "expert_novelty": 3,
        "expert_feasibility": 4, "expert_impact": 4,
        "expert_presentation": 3, "advance": 1
    },

    # ── NOT ADVANCING (advance = 0) ────────────────────
    {
        "idea_id": 11, "title": "WasteWise",
        "description": "Smart dustbin using computer vision to sort recyclable and non-recyclable waste. Municipal corporations can track bin fill levels remotely via a dashboard.",
        "category": "Environment", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 3,
        "expert_feasibility": 2, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 12, "title": "MindSpace",
        "description": "Mental health journaling app using NLP to detect emotional patterns in diary entries and suggest coping strategies. Tracks mood trends over weeks.",
        "category": "Mental Health", "source": "Devpost",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 3, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 13, "title": "RouteIQ",
        "description": "AI-powered public transport optimizer that predicts bus and metro crowd levels using historical data. Suggests least crowded routes and departure times to commuters.",
        "category": "Transport", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 2, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 14, "title": "FarmSense",
        "description": "IoT sensors placed in fields monitor soil moisture and temperature in real time. Sends alerts to farmers via SMS when irrigation is needed.",
        "category": "Agriculture", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 3, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 15, "title": "ParkEase",
        "description": "Mobile app showing real-time parking slot availability in malls and airports using sensors. Users can reserve a spot in advance and get navigation directions.",
        "category": "Transport", "source": "Devpost",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 3, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 16, "title": "QuizBot",
        "description": "A chatbot that generates multiple choice quiz questions from any text the student pastes. Student pastes their notes and gets 10 practice questions instantly.",
        "category": "Education", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 4, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 17, "title": "ExpenseTracker",
        "description": "A mobile app that scans receipts using OCR and automatically categorizes expenses into food, transport, and shopping. Shows monthly spending charts.",
        "category": "Finance", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 4, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 18, "title": "WeatherAlert",
        "description": "Simple weather alert app that sends push notifications when rain or storms are predicted in the user's location. Uses existing weather APIs.",
        "category": "Utility", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 4, "expert_impact": 1,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 19, "title": "TodoList Pro",
        "description": "A task management app with reminders, priority tags, and color coding. Users can create daily to-do lists and mark tasks as complete. Syncs across devices.",
        "category": "Productivity", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 4, "expert_impact": 1,
        "expert_presentation": 1, "advance": 0
    },
    {
        "idea_id": 20, "title": "ContactBook",
        "description": "A digital contact management app where users can store names, phone numbers, and email addresses. Includes search functionality and export to CSV.",
        "category": "Utility", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 4, "expert_impact": 1,
        "expert_presentation": 1, "advance": 0
    },
    {
        "idea_id": 21, "title": "BusTracker",
        "description": "App showing live bus location on a map using GPS tracker installed in buses. Students can see when their college bus will arrive at their stop.",
        "category": "Transport", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 3, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 22, "title": "VoiceNotes",
        "description": "A voice recording app that transcribes speech to text using basic speech recognition. Users can record meetings and get a text summary.",
        "category": "Productivity", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 3, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 23, "title": "LibraryApp",
        "description": "Digital library management system for college libraries. Students can search book availability, reserve books online, and get due date reminders via email.",
        "category": "Education", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 4, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 24, "title": "FoodCalorie",
        "description": "App where users search for any food item and see its calorie count and nutritional breakdown. Includes a daily calorie tracker with pie charts.",
        "category": "Health", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 4, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 25, "title": "OnlineVoting",
        "description": "A secure online voting platform for college elections. Students log in with their ID, cast their vote, and results are displayed in real time after polls close.",
        "category": "Governance", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 3, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 26, "title": "BlindNav",
        "description": "Navigation assistant for visually impaired people using audio cues and haptic feedback. Uses phone camera to detect obstacles and read signboards aloud in real time.",
        "category": "Accessibility", "source": "Devpost",
        "expert_rank": 0, "expert_novelty": 3,
        "expert_feasibility": 2, "expert_impact": 4,
        "expert_presentation": 3, "advance": 0
    },
    {
        "idea_id": 27, "title": "MentalBot",
        "description": "AI chatbot providing basic mental health support for college students. Asks about mood daily and suggests breathing exercises, music, or professional help based on responses.",
        "category": "Mental Health", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 3, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 28, "title": "FakeNewsDetector",
        "description": "Browser extension that checks news article credibility by comparing content with fact-checking databases and flagging unverified claims with a red warning badge.",
        "category": "Media", "source": "Devpost",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 2, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 29, "title": "WaterQuality",
        "description": "Low cost IoT device measuring water pH, turbidity, and contamination levels in rural water sources. Sends data to a government dashboard and alerts when levels are unsafe.",
        "category": "Environment", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 3,
        "expert_feasibility": 2, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 30, "title": "SleepTracker",
        "description": "Mobile app that monitors sleep quality using phone accelerometer. Tracks sleep stages, detects disturbances, and gives a daily sleep quality score with improvement tips.",
        "category": "Health", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 3, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 31, "title": "EyeStrain Guard",
        "description": "Desktop app that monitors screen time and reminds users to take eye breaks using the 20-20-20 rule. Dims screen automatically after long usage periods.",
        "category": "Health", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 4, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 32, "title": "CodeReviewer",
        "description": "AI tool that reviews student code assignments and gives feedback on logic errors, code style, and efficiency. Helps students improve before final submission.",
        "category": "Education", "source": "Devpost",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 3, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 33, "title": "VirtualTryOn",
        "description": "AR app that lets users virtually try on clothing and glasses using front camera. Shows how the item looks on the user before purchasing online.",
        "category": "E-commerce", "source": "Devpost",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 2, "expert_impact": 2,
        "expert_presentation": 3, "advance": 0
    },
    {
        "idea_id": 34, "title": "PlagiarismCheck",
        "description": "Web tool where students paste their assignment text and get a plagiarism percentage score. Highlights copied sentences and shows the original source URL.",
        "category": "Education", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 3, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 35, "title": "CampusLost",
        "description": "College lost and found platform where students post photos of lost items. AI matches lost item descriptions with found item posts using image similarity search.",
        "category": "Utility", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 3, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 36, "title": "AttendanceAI",
        "description": "Face recognition system for classroom attendance. Camera at entrance scans faces as students enter and marks attendance automatically in the college ERP system.",
        "category": "Education", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 3, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 37, "title": "PriceCompare",
        "description": "Browser extension that compares product prices across Amazon, Flipkart, and Meesho and shows the cheapest option with a price history graph.",
        "category": "E-commerce", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 3, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 38, "title": "JobMatch",
        "description": "Platform that matches fresh graduates with internship and job openings based on their skills and location. Sends daily job alerts and tracks application status.",
        "category": "Careers", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 3, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 39, "title": "PhotoEditor",
        "description": "Simple photo editing web app with filters, crop, rotate, and brightness adjustment tools. Works in browser with no download needed.",
        "category": "Utility", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 4, "expert_impact": 1,
        "expert_presentation": 1, "advance": 0
    },
    {
        "idea_id": 40, "title": "NoteShare",
        "description": "Platform where college students upload and download lecture notes by subject and semester. Includes rating system so best notes appear first.",
        "category": "Education", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 4, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 41, "title": "GreenRoof AI",
        "description": "AI system analyzing satellite imagery to identify buildings suitable for rooftop solar panel installation. Calculates expected energy output and ROI for building owners.",
        "category": "Energy", "source": "Devpost",
        "expert_rank": 0, "expert_novelty": 3,
        "expert_feasibility": 2, "expert_impact": 3,
        "expert_presentation": 3, "advance": 0
    },
    {
        "idea_id": 42, "title": "TrafficPredict",
        "description": "Machine learning model predicting traffic congestion 30 minutes in advance for city roads using historical traffic, weather, and event data. Integrated with Google Maps.",
        "category": "Transport", "source": "Devpost",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 2, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 43, "title": "MediRemind",
        "description": "Medication reminder app for elderly patients. Caregivers set up medication schedules and the app sends voice reminders in local language with confirmation tracking.",
        "category": "Healthcare", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 4, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 44, "title": "EduGame",
        "description": "Educational game teaching primary school children math and science concepts through interactive puzzles and cartoon animations. Available in 5 regional languages.",
        "category": "Education", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 3, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 45, "title": "SoilHealth",
        "description": "Portable soil testing kit connected to a smartphone app. Farmers insert the probe into soil and get instant NPK nutrient levels with fertilizer recommendations.",
        "category": "Agriculture", "source": "Devpost",
        "expert_rank": 0, "expert_novelty": 3,
        "expert_feasibility": 2, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 46, "title": "SignatureVerify",
        "description": "AI system verifying handwritten signatures on bank documents to detect forgery. Banks upload scanned signature images and the system flags suspicious ones.",
        "category": "Finance", "source": "Devpost",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 2, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 47, "title": "AQI Monitor",
        "description": "Low-cost air quality monitor using dust and gas sensors placed in schools and hospitals. Sends alerts when AQI crosses dangerous levels and suggests protective measures.",
        "category": "Environment", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 3, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 48, "title": "VirtualLab",
        "description": "Browser-based virtual chemistry lab where students perform experiments in 3D simulation without physical chemicals. Records results and generates lab reports automatically.",
        "category": "Education", "source": "Devpost",
        "expert_rank": 0, "expert_novelty": 3,
        "expert_feasibility": 2, "expert_impact": 3,
        "expert_presentation": 3, "advance": 0
    },
    {
        "idea_id": 49, "title": "CrimeMap",
        "description": "Web dashboard showing crime incident heatmaps for a city using publicly available police report data. Citizens can report incidents and police can track patterns.",
        "category": "Safety", "source": "Devpost",
        "expert_rank": 0, "expert_novelty": 2,
        "expert_feasibility": 3, "expert_impact": 3,
        "expert_presentation": 2, "advance": 0
    },
    {
        "idea_id": 50, "title": "BudgetPlanner",
        "description": "Personal finance app for college students that tracks monthly income and expenses, sets savings goals, and shows budget alerts when overspending in a category.",
        "category": "Finance", "source": "College Hackathon",
        "expert_rank": 0, "expert_novelty": 1,
        "expert_feasibility": 4, "expert_impact": 2,
        "expert_presentation": 2, "advance": 0
    }
]

df = pd.DataFrame(ideas)

# Add empty AI score columns
ai_cols = [
    "qwen_feasibility",
    "qwen_impact", "qwen_presentation",
    "qwen_overall", "qwen_rank",
    "mistral_feasibility",
    "mistral_impact", "mistral_presentation",
    "mistral_overall", "mistral_rank",
    "llama_novelty", "llama_feasibility",
    "llama_impact", "llama_presentation",
    "llama_overall", "llama_rank",
    "llama_feedback",
    "llama_status", "mistral_status", "qwen_status",
    "avg_overall", "final_rank",
    "ai_advance", "agrees_with_expert",
    "llama_accuracy", "mistral_accuracy", "qwen_accuracy", "combined_accuracy",
    "score_spread", "human_review_required",
    "spearman_rank_correlation"
]
for col in ai_cols:
    df[col] = ""

df.to_excel("ideas.xlsx", index=False)

print("=" * 50)
print("✅ ideas.xlsx created with 50 ideas!")
print("=" * 50)
print(f"  Total ideas    : {len(df)}")
print(f"  Should advance : {df['advance'].sum()} ideas")
print(f"  Not advancing  : {(df['advance']==0).sum()} ideas")
print("=" * 50)
