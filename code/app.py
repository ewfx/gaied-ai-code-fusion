from flask import Flask, request, jsonify
from ServiceRequest import (
    extract_text_from_pdf, classify_email_with_gemini, calculate_confidence_score,
    normalize_confidence_scores, clean_gemini_output, parse_gemini_output, process_pdf_folder
)
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Service Request API is running!"

@app.route('/api/extract_text', methods=['POST'])
def extract_text():
    data = request.json
    pdf_path = data.get('pdf_path')
    if not os.path.exists(pdf_path):
        return jsonify({"error": "PDF file not found"}), 404
    text = extract_text_from_pdf(pdf_path)
    return jsonify({"text": text})

@app.route('/api/classify_email', methods=['POST'])
def classify_email():
    data = request.json
    email_text = data.get('email_text')
    if not email_text:
        return jsonify({"error": "Email text is required"}), 400
    gemini_output = classify_email_with_gemini(email_text)
    cleaned_output = clean_gemini_output(gemini_output)
    parsed_output = parse_gemini_output(cleaned_output)
    return jsonify(parsed_output)

@app.route('/api/calculate_confidence', methods=['POST'])
def calculate_confidence():
    data = request.json
    original_text = data.get('original_text')
    generated_text = data.get('generated_text')
    if not original_text or not generated_text:
        return jsonify({"error": "Both original and generated text are required"}), 400
    confidence_score = calculate_confidence_score(original_text, generated_text)
    return jsonify({"confidence_score": confidence_score})

@app.route('/api/process_folder', methods=['POST'])
def process_folder():
    data = request.json
    folder_path = data.get('folder_path')
    if not os.path.exists(folder_path):
        return jsonify({"error": "Folder not found"}), 404
    results_df = process_pdf_folder(folder_path)
    results = results_df.to_dict(orient='records')
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)