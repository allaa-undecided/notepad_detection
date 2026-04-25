# Vision-Based Desktop Automation with Dynamic Icon Grounding

## 🚀 Overview

This project implements a vision-based desktop automation system that dynamically locates and interacts with desktop icons using a vision-language model.

The system is designed to be robust to changes in icon position and does not rely on hardcoded coordinates or template matching. Instead, it uses a grounding model (ZonUI-3B) to identify UI elements based on natural language descriptions.

---

## 🎯 Objective

Automatically:
1. Detect the Notepad icon on the desktop using visual grounding
2. Launch Notepad
3. Fetch posts from an API
4. Type content into Notepad
5. Save each post as a `.txt` file
6. Repeat the process dynamically

---

## 🧠 Key Features

- 🔍 **Dynamic Icon Grounding**
  - Detects Notepad icon regardless of position
  - Works even when icons are moved around

- 🧠 **Vision-Language Model**
  - Uses `ZonUI-3B` for text-to-UI grounding
  - No template matching or fixed image dependency

- ✅ **Verification Step**
  - Crops predicted region
  - Confirms it is actually Notepad before clicking
  - Prevents false clicks

- 🖱️ **Desktop Automation**
  - Opens Notepad
  - Types structured content
  - Saves files automatically

- 🌐 **API Integration**
  - Fetches posts from JSONPlaceholder
  - Includes fallback handling for network issues

- 🛡️ **Robustness**
  - Handles missing icons safely
  - Avoids random clicks
  - Validates Notepad launch

---

## ⚙️ Tech Stack

- Python 3.12
- PyTorch (CPU)
- Transformers (Hugging Face)
- ZonUI-3B (Vision-Language Model)
- PyAutoGUI (automation)
- Requests (API)
- PIL (image processing)

