# app.py
import os
import json
import time
from io import BytesIO
from pathlib import Path
from typing import List, Dict

import streamlit as st
import pandas as pd
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# -----------------------
# Configuration & Helpers
# -----------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RECIPES_FILE = DATA_DIR / "saved_recipes.json"

# load saved recipes
def load_saved_recipes() -> List[Dict]:
    if RECIPES_FILE.exists():
        return json.loads(RECIPES_FILE.read_text(encoding="utf-8"))
    return []

def save_all_recipes(recipes: List[Dict]):
    RECIPES_FILE.write_text(json.dumps(recipes, ensure_ascii=False, indent=2), encoding="utf-8")

def add_recipe(recipe_obj: Dict):
    recipes = load_saved_recipes()
    recipes.insert(0, recipe_obj)
    save_all_recipes(recipes)

def delete_recipe(index: int):
    recipes = load_saved_recipes()
    if 0 <= index < len(recipes):
        recipes.pop(index)
        save_all_recipes(recipes)

# PDF generator
def recipe_to_pdf_bytes(recipe_text: str, title: str = "recipe") -> bytes:
    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y = height - 60
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, title)
    pdf.setFont("Helvetica", 10)
    y -= 30
    for line in recipe_text.splitlines():
        pdf.drawString(40, y, line)
        y -= 14
        if y < 50:
            pdf.showPage()
            y = height - 60
    pdf.save()
    buf.seek(0)
    return buf.read()

# Quick naive nutrition estimation (very approximate)
NUTRITION_DB = {
    # calories per 100g and macro percentages rough
    "chicken": {"cal": 165, "protein": 31, "fat": 3.6, "carb": 0},
    "rice": {"cal": 130, "protein": 2.4, "fat": 0.2, "carb": 28},
    "pasta": {"cal": 131, "protein": 5, "fat": 1.1, "carb": 25},
    "tomato": {"cal": 18, "protein": 0.9, "fat": 0.2, "carb": 3.9},
    "onion": {"cal": 40, "protein": 1.1, "fat": 0.1, "carb": 9.3},
    "potato": {"cal": 77, "protein": 2, "fat": 0.1, "carb": 17},
    "cheese": {"cal": 402, "protein": 25, "fat": 33, "carb": 1.3},
    "egg": {"cal": 155, "protein": 13, "fat": 11, "carb": 1.1},
    "milk": {"cal": 42, "protein": 3.4, "fat": 1, "carb": 5},
    "butter": {"cal": 717, "protein": 0.9, "fat": 81, "carb": 0.1},
    "broccoli": {"cal": 34, "protein": 2.8, "fat": 0.4, "carb": 7},
    "quinoa": {"cal": 120, "protein": 4.4, "fat": 1.9, "carb": 21},
    "beans": {"cal": 347, "protein": 21, "fat": 1.2, "carb": 63},
    "avocado": {"cal": 160, "protein": 2, "fat": 15, "carb": 9},
    # add more as needed...
}

def estimate_nutrition(ingredients_list: List[str], servings: int = 2):
    # Very rough: assume 100g per ingredient scaled and average values
    total = {"cal": 0.0, "protein": 0.0, "fat": 0.0, "carb": 0.0}
    for ing in ingredients_list:
        key = ing.strip().lower().split()[0]
        if key in NUTRITION_DB:
            d = NUTRITION_DB[key]
            total["cal"] += d["cal"]
            total["protein"] += d["protein"]
            total["fat"] += d["fat"]
            total["carb"] += d["carb"]
        else:
            # unknown ingredient, assume small default values
            total["cal"] += 20
            total["protein"] += 0.5
            total["fat"] += 0.1
            total["carb"] += 1
    # divide by servings
    per_serv = {k: round(v / max(1, servings), 1) for k, v in total.items()}
    return per_serv

# -----------------------
# Gemini config helper
# -----------------------
def configure_gemini(api_key: str):
    if not api_key:
        raise ValueError("API Key missing")
    genai.configure(api_key=api_key)

# -----------------------
# Streamlit UI & Pages
# -----------------------
st.set_page_config(page_title="AI Recipe Studio", layout="wide", initial_sidebar_state="expanded")

# CSS for nicer dark UI (glass cards, sidebar)
st.markdown(
    """
    <style>
    :root {
      --bg:#0b0d0f;
      --card:#0f1113;
      --muted:#9aa3a5;
      --accent:#3ddc84;
      --glass: rgba(255,255,255,0.03);
    }
    body { background: var(--bg); color: #e9eef0; }
    .stApp { background: linear-gradient(180deg, rgba(15,17,19,1) 0%, rgba(11,13,15,1) 100%); }
    .sidebar .sidebar-content { background: #0f1113; border-right: 1px solid rgba(255,255,255,0.02); }
    .big-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding: 20px; border-radius: 12px; }
    .recipe-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding: 16px; border-radius: 12px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.02);}
    .stButton>button { background: linear-gradient(90deg, #3ddc84, #2fb36b); color: #081010; font-weight: 700; }
    .small-muted { color: #9aa3a5; font-size: 13px; }
    .accent { color: var(--accent); font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header (top bar)
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:18px;padding:8px 0 18px 0;">
      <div style="width:48px;height:48px;background:#0f9b5a;border-radius:12px;display:flex;align-items:center;justify-content:center;font-weight:800;color:#06110b;font-size:20px">AI</div>
      <div>
        <h2 style="margin:0;margin-bottom:4px">AI Recipe Studio</h2>
        <div class="small-muted">Your pocket AI chef — powered by Gemini</div>
      </div>
      <div style="margin-left:auto;color:#9aa3a5">Theme: <span class="accent">Dark</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation
st.sidebar.title("🍴 AI Recipe Studio")
page = st.sidebar.radio("Navigate", ["Discover", "My Recipes", "Meal Planner", "Nutrition", "Shopping List", "Chat", "Settings"])

# Ensure session state pieces
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GOOGLE_API_KEY", "")
if "last_generated" not in st.session_state:
    st.session_state.last_generated = None
if "history" not in st.session_state:
    st.session_state.history = []

# Attempt to configure Gemini if we have key
if st.session_state.api_key:
    try:
        configure_gemini(st.session_state.api_key)
        # pick a strong model
        RECIPE_MODEL_NAME = "models/gemini-2.5-flash"
        recipe_model = genai.GenerativeModel(RECIPE_MODEL_NAME)
        chat_model = genai.GenerativeModel("models/gemini-2.5-pro")  # for chat suggestions
    except Exception as e:
        recipe_model = None
        chat_model = None
        st.sidebar.error("Gemini config failed — open Settings to set or paste API key.")

# ------------ PAGE: Discover ------------
if page == "Discover":
    st.subheader("Discover Recipes")
    with st.container():
        left, right = st.columns([1, 1.1])
        with left:
            with st.form("discover_form"):
                ingredients = st.text_area("🧂 Ingredients (comma-separated)", placeholder="tomato, pasta, garlic, cheese")
                cuisine = st.selectbox("🌍 Cuisine", ["Any", "Indian", "Italian", "Mexican", "Chinese", "American", "Other"])
                diet = st.selectbox("🥗 Diet", ["Any", "Vegetarian", "Vegan", "Non-Vegetarian", "Keto", "Gluten-Free"])
                servings = st.slider("🍽 Servings", 1, 8, 2)
                submit = st.form_submit_button("✨ Generate Recipe")
            if submit:
                if not ingredients.strip():
                    st.warning("Please enter ingredients.")
                else:
                    if not st.session_state.api_key:
                        st.error("Please add your API key in Settings (sidebar).")
                    else:
                        with st.spinner("Asking your AI chef..."):
                            prompt = f"""
                            Create a detailed recipe using these ingredients: {ingredients}.
                            Cuisine: {cuisine}.
                            Dietary preference: {diet}.
                            Servings: {servings}.
                            Please include:
                            - Recipe name
                            - Short description
                            - Complete ingredients list with quantities for the total dish
                            - Step-by-step preparation
                            - Estimated cooking time
                            - Serving suggestions
                            Format the output clearly.
                            """
                            try:
                                resp = recipe_model.generate_content(prompt)
                                recipe_text = resp.text.strip()
                                st.session_state.last_generated = {
                                    "ingredients_input": ingredients,
                                    "cuisine": cuisine,
                                    "diet": diet,
                                    "servings": servings,
                                    "text": recipe_text,
                                    "timestamp": time.time(),
                                }
                                st.success("Recipe generated!")
                            except Exception as e:
                                st.error(f"API error: {e}")

        with right:
            st.markdown("### 🔎 Preview / Save")
            if st.session_state.last_generated:
                r = st.session_state.last_generated
                st.markdown(f"<div class='recipe-card'><b>{r.get('text').splitlines()[0]}</b><br><div class='small-muted'>Cuisine: {r['cuisine']} • Diet: {r['diet']} • Servings: {r['servings']}</div><hr style='opacity:0.06'>{r['text']}</div>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1,1,1])
                with col1:
                    if st.button("💾 Save Recipe"):
                        add_recipe({**r})
                        st.success("Saved to My Recipes.")
                with col2:
                    pdf_bytes = recipe_to_pdf_bytes(r["text"], title=r["text"].splitlines()[0] if r["text"] else "Recipe")
                    st.download_button("📄 Download PDF", data=pdf_bytes, file_name="recipe.pdf", mime="application/pdf")
                with col3:
                    st.download_button("📥 Export JSON", data=json.dumps(r, ensure_ascii=False, indent=2), file_name="recipe.json", mime="application/json")
            else:
                st.info("Generate a recipe on the left to preview and save it here.")

# ------------ PAGE: My Recipes ------------
elif page == "My Recipes":
    st.subheader("📚 My Recipes")
    recipes = load_saved_recipes()
    if not recipes:
        st.info("No saved recipes yet. Generate recipes on Discover and click Save.")
    else:
        search_q = st.text_input("Search by ingredient, cuisine, or name", "")
        filtered = []
        for idx, rec in enumerate(recipes):
            text = (rec.get("text") or "").lower()
            meta = f"{rec.get('cuisine','')} {rec.get('diet','')}".lower()
            if search_q.strip().lower() in (text + meta):
                filtered.append((idx, rec))
            elif search_q.strip() == "":
                filtered.append((idx, rec))
        for idx, rec in filtered:
            with st.expander(f"{rec.get('text','').splitlines()[0] if rec.get('text') else 'Recipe'}"):
                st.markdown(f"<div class='recipe-card'>{rec.get('text')}</div>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1,1,1])
                with col1:
                    if st.button("📄 Download PDF", key=f"pdf_{idx}"):
                        pdf = recipe_to_pdf_bytes(rec.get("text",""), title=rec.get("text","").splitlines()[0])
                        st.download_button("Click to download", data=pdf, file_name="recipe.pdf", mime="application/pdf")
                with col2:
                    if st.button("🗂 Export JSON", key=f"json_{idx}"):
                        st.download_button("Download JSON", data=json.dumps(rec, ensure_ascii=False, indent=2), file_name="recipe.json", mime="application/json")
                with col3:
                    if st.button("🗑 Delete", key=f"del_{idx}"):
                        delete_recipe(idx)
                        st.experimental_rerun()

# ------------ PAGE: Meal Planner ------------
elif page == "Meal Planner":
    st.subheader("🗓️ Weekly Meal Planner")
    servings = st.slider("Default servings per meal", 1, 6, 2)
    generate_plan = st.button("🧾 Generate 7-day Meal Plan")
    if generate_plan:
        if not st.session_state.api_key:
            st.error("Set API key in Settings first.")
        else:
            days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            plan = {}
            with st.spinner("Generating weekly plan..."):
                for d in days:
                    prompt = f"Create a {d} meal (dish name and short recipe) suitable for {servings} servings. Keep it concise with ingredients and 3-4 steps."
                    try:
                        res = recipe_model.generate_content(prompt)
                        plan[d] = res.text.strip()
                    except Exception as e:
                        plan[d] = f"Error: {e}"
            st.session_state.meal_plan = plan
            st.success("Weekly meal plan created.")
    if "meal_plan" in st.session_state:
        for d,txt in st.session_state.meal_plan.items():
            st.markdown(f"### {d}")
            st.markdown(f"<div class='recipe-card'>{txt}</div>", unsafe_allow_html=True)
        # download as CSV/PDF
        if st.button("📥 Download Meal Plan as CSV"):
            df = pd.DataFrame(list(st.session_state.meal_plan.items()), columns=["Day","Recipe"])
            st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="meal_plan.csv", mime="text/csv")
        if st.button("📄 Download Meal Plan as PDF"):
            all_text = "\n\n".join([f"{d}:\n{r}" for d,r in st.session_state.meal_plan.items()])
            pdf = recipe_to_pdf_bytes(all_text, title="Weekly Meal Plan")
            st.download_button("Download PDF", data=pdf, file_name="meal_plan.pdf", mime="application/pdf")

# ------------ PAGE: Nutrition ------------
elif page == "Nutrition":
    st.subheader("📊 Nutrition Analyzer (approximate)")
    sample = st.text_area("Paste recipe ingredients (one per comma) or type ingredients", placeholder="chicken, rice, tomato")
    servings = st.number_input("Servings", min_value=1, max_value=10, value=2)
    if st.button("🔎 Analyze Nutrition"):
        ings = [s.strip() for s in sample.split(",") if s.strip()]
        if not ings:
            st.warning("Enter ingredients.")
        else:
            per = estimate_nutrition(ings, servings)
            st.metric("Calories / serving (approx)", f"{per['cal']} kcal")
            cols = st.columns(3)
            cols[0].progress(min(100, int(per["protein"]*3)))  # simple visual
            cols[0].caption(f"Protein: {per['protein']} g")
            cols[1].progress(min(100, int(per["fat"]*2)))
            cols[1].caption(f"Fat: {per['fat']} g")
            cols[2].progress(min(100, int(per["carb"]*1)))
            cols[2].caption(f"Carbs: {per['carb']} g")

# ------------ PAGE: Shopping List ------------
elif page == "Shopping List":
    st.subheader("🛒 Shopping List Generator")
    recipes = load_saved_recipes()
    if not recipes:
        st.info("Save recipes in My Recipes to combine ingredients into a shopping list.")
    else:
        selected = st.multiselect("Select recipes to build shopping list", options=[rec.get("text","").splitlines()[0] for rec in recipes])
        if st.button("🧾 Build Shopping List"):
            items = {}
            for rec in recipes:
                title = rec.get("text","").splitlines()[0]
                if title in selected:
                    # attempt to parse ingredient lines: naive approach - find lines that contain common separators
                    lines = rec.get("text","").splitlines()
                    for line in lines:
                        if any(keyword in line.lower() for keyword in ["cup","tsp","tbsp","g","kg","ml","slice","pieces","pieces","pinch"]) or "," in line:
                            # split by comma and keep words
                            parts = [p.strip() for p in re_split_commas(line)]
                            for p in parts:
                                if len(p) > 2:
                                    items[p] = items.get(p, 0) + 1
                        elif any(word in line.lower() for word in ["tomato","onion","garlic","chicken","rice","pasta","cheese","egg"]):
                            items[line.strip()] = items.get(line.strip(), 0) + 1
            if items:
                st.markdown("### 🧾 Consolidated Shopping List")
                for k in items.keys():
                    st.write(f"- {k}")
                csv = "\n".join(items.keys()).encode("utf-8")
                st.download_button("Download as TXT", data=csv, file_name="shopping_list.txt", mime="text/plain")
            else:
                st.info("Couldn't extract ingredients automatically. Try adding clearer ingredient lists to saved recipes.")

# ------------ PAGE: Chat ------------
elif page == "Chat":
    st.subheader("💬 Chat with your AI Chef")
    st.info("Ask cooking questions, substitutions, or tips. Uses Gemini for quick chat responses.")
    if not st.session_state.api_key:
        st.error("Add API key in Settings to use Chat.")
    else:
        user_msg = st.text_input("Ask anything about cooking or recipes", "")
        if st.button("Send"):
            if user_msg.strip():
                with st.spinner("Thinking..."):
                    try:
                        chat_prompt = f"You are a helpful chef. Answer briefly and clearly: {user_msg}"
                        chat_resp = chat_model.generate_content(chat_prompt)
                        st.markdown(f"**AI:** {chat_resp.text}")
                    except Exception as e:
                        st.error(f"API error: {e}")
            else:
                st.warning("Type a question first.")

# ------------ PAGE: Settings ------------
elif page == "Settings":
    st.subheader("⚙️ Settings")
    st.markdown("Enter your Gemini API key (it will be stored only in this session). If you prefer, set environment variable `GOOGLE_API_KEY` before starting Streamlit.")
    key = st.text_input("Gemini API Key", value=st.session_state.api_key or "", type="password")
    if st.button("Save API Key"):
        st.session_state.api_key = key.strip()
        try:
            configure_gemini(st.session_state.api_key)
            # reassign models
            recipe_model = genai.GenerativeModel("models/gemini-2.5-flash")
            chat_model = genai.GenerativeModel("models/gemini-2.5-pro")
            st.success("API key saved and Gemini configured.")
        except Exception as e:
            st.error(f"Could not configure Gemini: {e}")
    st.markdown("---")
    st.markdown("**App Info**")
    st.write("AI Recipe Studio — local demo. Saves recipes to `data/saved_recipes.json`.")
    if st.button("Clear saved recipes (delete file)"):
        if RECIPES_FILE.exists():
            RECIPES_FILE.unlink()
            st.success("Saved recipes cleared.")
        else:
            st.info("No saved recipes file found.")

# small footer (hidden by default by CSS above if you remove)
st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
st.caption("")

# -----------------------
# Utility small helpers used above
# -----------------------
import re
def re_split_commas(line):
    # split by commas but keep phrases in parentheses intact -- simple version
    parts = [p.strip() for p in re.split(r',|\band\b', line) if p.strip()]
    return parts
