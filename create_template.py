import pandas as pd

# ─────────────────────────────────────────
# SAMPLE DATA — 10 real-style hackathon ideas
# Replace these with your actual collected ideas
# ─────────────────────────────────────────

sample_ideas = [
    {
        "idea_id": 1,
        "title": "MediScan AI",
        "description": "An AI-powered mobile app that detects skin diseases from photos using computer vision. Users upload a photo and receive instant diagnosis with treatment recommendations and nearby clinic suggestions.",
        "category": "Healthcare",
        "source": "Devpost",
        "expert_rank": 1,
        "expert_novelty": 4,
        "expert_feasibility": 3,
        "expert_impact": 4,
        "expert_presentation": 4,
        "advance": 1
    },
    {
        "idea_id": 2,
        "title": "EcoTrack",
        "description": "A platform that tracks individual carbon footprint by analyzing purchase history, travel data, and energy usage. Provides personalized weekly tips and challenges to reduce emissions with gamification.",
        "category": "Environment",
        "source": "Devpost",
        "expert_rank": 2,
        "expert_novelty": 3,
        "expert_feasibility": 4,
        "expert_impact": 4,
        "expert_presentation": 3,
        "advance": 1
    },
    {
        "idea_id": 3,
        "title": "StudyBuddy",
        "description": "An AI chatbot that creates personalized study plans for students based on their learning style, exam schedule, and weak areas. Generates quizzes automatically and tracks daily progress.",
        "category": "Education",
        "source": "Devpost",
        "expert_rank": 3,
        "expert_novelty": 3,
        "expert_feasibility": 4,
        "expert_impact": 3,
        "expert_presentation": 3,
        "advance": 1
    },
    {
        "idea_id": 4,
        "title": "FarmSense",
        "description": "IoT sensors placed in agricultural fields that monitor soil moisture, temperature, and crop health in real time. Sends alerts to farmers via SMS when intervention is needed.",
        "category": "Agriculture",
        "source": "College Hackathon",
        "expert_rank": 0,
        "expert_novelty": 3,
        "expert_feasibility": 3,
        "expert_impact": 4,
        "expert_presentation": 2,
        "advance": 0
    },
    {
        "idea_id": 5,
        "title": "SafeWalk",
        "description": "A women safety app that uses GPS to detect unusual stops or route deviations and automatically alerts emergency contacts. Includes a panic button and live location sharing feature.",
        "category": "Safety",
        "source": "Devpost",
        "expert_rank": 0,
        "expert_novelty": 2,
        "expert_feasibility": 4,
        "expert_impact": 4,
        "expert_presentation": 3,
        "advance": 0
    },
    {
        "idea_id": 6,
        "title": "WasteWise",
        "description": "A smart dustbin system using computer vision to automatically sort recyclable and non-recyclable waste. Municipal corporations can track bin fill levels remotely via a dashboard.",
        "category": "Environment",
        "source": "College Hackathon",
        "expert_rank": 0,
        "expert_novelty": 3,
        "expert_feasibility": 2,
        "expert_impact": 3,
        "expert_presentation": 2,
        "advance": 0
    },
    {
        "idea_id": 7,
        "title": "LegalEase",
        "description": "An AI tool that simplifies legal documents into plain language for common citizens. Users upload any legal contract and receive a plain-English summary with key risk highlights.",
        "category": "Legal",
        "source": "Devpost",
        "expert_rank": 0,
        "expert_novelty": 3,
        "expert_feasibility": 3,
        "expert_impact": 3,
        "expert_presentation": 3,
        "advance": 0
    },
    {
        "idea_id": 8,
        "title": "MindSpace",
        "description": "A mental health journaling app that uses NLP to detect emotional patterns in daily entries and suggests coping strategies. Tracks mood trends over weeks and alerts trusted contacts if distress is detected.",
        "category": "Mental Health",
        "source": "Devpost",
        "expert_rank": 0,
        "expert_novelty": 2,
        "expert_feasibility": 3,
        "expert_impact": 3,
        "expert_presentation": 2,
        "advance": 0
    },
    {
        "idea_id": 9,
        "title": "RouteIQ",
        "description": "An AI-powered public transport optimizer that predicts bus and metro crowd levels using historical data and suggests the least crowded routes and departure times to commuters.",
        "category": "Transport",
        "source": "College Hackathon",
        "expert_rank": 0,
        "expert_novelty": 2,
        "expert_feasibility": 2,
        "expert_impact": 2,
        "expert_presentation": 2,
        "advance": 0
    },
    {
        "idea_id": 10,
        "title": "SkillBridge",
        "description": "A platform connecting rural youth with urban mentors for skill development through short video lessons and live Q&A sessions. Works on low bandwidth connections with offline download support.",
        "category": "Education",
        "source": "Devpost",
        "expert_rank": 0,
        "expert_novelty": 2,
        "expert_feasibility": 3,
        "expert_impact": 3,
        "expert_presentation": 2,
        "advance": 0
    }
]

# ─────────────────────────────────────────
# CREATE DATAFRAME WITH ALL COLUMNS
# ─────────────────────────────────────────

df = pd.DataFrame(sample_ideas)

# Add empty AI score columns — these will be
# filled automatically by scorer.py later
ai_columns = {
    # Qwen3-32B scores
    "qwen3_32b_novelty": "",
    "qwen3_32b_feasibility": "",
    "qwen3_32b_impact": "",
    "qwen3_32b_presentation": "",
    "qwen3_32b_overall": "",
    "qwen3_32b_rank": "",
    "qwen3_32b_feedback": "",

    # DeepSeek scores
    "deepseek_novelty": "",
    "deepseek_feasibility": "",
    "deepseek_impact": "",
    "deepseek_presentation": "",
    "deepseek_overall": "",
    "deepseek_rank": "",
    "deepseek_feedback": "",

    # Llama scores
    "llama_novelty": "",
    "llama_feasibility": "",
    "llama_impact": "",
    "llama_presentation": "",
    "llama_overall": "",
    "llama_rank": "",
    "llama_feedback": "",

    # Final combined columns
    "avg_overall": "",
    "final_rank": "",
    "agrees_with_expert": ""
}

for col, val in ai_columns.items():
    df[col] = val

# ─────────────────────────────────────────
# STYLE AND SAVE AS EXCEL
# ─────────────────────────────────────────

with pd.ExcelWriter("ideas.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Hackathon Ideas")

    # Auto-adjust column widths
    worksheet = writer.sheets["Hackathon Ideas"]
    for col_cells in worksheet.columns:
        max_length = 0
        col_letter = col_cells[0].column_letter
        for cell in col_cells:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        worksheet.column_dimensions[col_letter].width = min(max_length + 4, 50)

print("=" * 50)
print("✅ ideas.xlsx created successfully!")
print("=" * 50)
print(f"📊 Total columns  : {len(df.columns)}")
print(f"📝 Sample ideas   : {len(df)} rows")
print(f"📁 File location  : D:/Final_Year_Project/ideas.xlsx")

