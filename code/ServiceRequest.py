import os
import pandas as pd
#import fitz  # PyMuPDF
import pymupdf as fitz
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import numpy as np
#from google.colab import files



# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set your Google Gemini API key
genai.configure(api_key="AIzaSyBvq_YC9e-4UNy84hnQlud5Rh0_4qYTT10")

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    text = ""
    doc = fitz.open(pdf_path)  # Use fitz.open directly
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Classify emails using Google Gemini
def classify_email_with_gemini(email_text):
    """
    Classifies the email content into request type and sub-request type using Google Gemini.
    """
    prompt = f"""
    Analyze the following email and identify the main request type and sub-request type based on the content.
    
    Email: {email_text}
    
    Provide the output in JSON format with the following keys:
    - request_type
    - sub_request_type
    - customer_name
    """
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text

# Step 3: Calculate confidence score using Sentence Transformers
def calculate_confidence_score(original_text, generated_text):
    """
    Calculates the confidence score by comparing the original email text with the generated response.
    Uses Sentence Transformers for semantic similarity.
    """
    # Encode the original text and generated text into embeddings
    original_embedding = model.encode(original_text, convert_to_tensor=True)
    generated_embedding = model.encode(generated_text, convert_to_tensor=True)
    
    # Calculate cosine similarity between the embeddings
    similarity = util.cos_sim(original_embedding, generated_embedding)
    return similarity.item()  # Confidence score between -1 and 1

# Step 4: Normalize confidence scores
def normalize_confidence_scores(scores):
    """
    Normalizes confidence scores to a range of 0.8 to 1.0 for better interpretability.
    """
    scores = np.array(scores)
    normalized_scores = 0.8 + 0.2 * (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    return normalized_scores.tolist()

# Step 5: Clean Gemini output
def clean_gemini_output(gemini_output):
    """
    Cleans the Gemini output by removing unnecessary symbols and formatting.
    """
    # Remove unwanted symbols like ```json, ```, {, and }
    cleaned_output = gemini_output.replace("```json", "").replace("```", "").replace("{", "").replace("}", "").strip()
    return cleaned_output

# Step 6: Parse cleaned output into a dictionary
def parse_gemini_output(cleaned_output):
    """
    Parses the cleaned Gemini output into a dictionary.
    """
    # Split the cleaned output into key-value pairs
    data = {}
    for item in cleaned_output.split(","):
        key, value = item.split(":", 1)
        data[key.strip().strip('"')] = value.strip().strip('"')
    return data

# Step 7: Process PDF folder and classify emails
def process_pdf_folder(folder_path):
    """
    Processes all PDF files in a folder and classifies them using Google Gemini.
    Also, calculates a relevant confidence score using Sentence Transformers and normalizes the scores.
    """
    results = []
    confidence_scores = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            email_text = extract_text_from_pdf(pdf_path)
            gemini_output = classify_email_with_gemini(email_text)
            
            # Clean the Gemini output
            cleaned_output = clean_gemini_output(gemini_output)
            
            # Parse the cleaned output into a dictionary
            parsed_output = parse_gemini_output(cleaned_output)
            
            # Calculate confidence score using Sentence Transformers
            confidence_score = calculate_confidence_score(email_text, cleaned_output)
            confidence_scores.append(confidence_score)
            
            # Append results
            results.append({
                "Request Type": parsed_output.get("request_type", "N/A"),
                "Sub Request Type": parsed_output.get("sub_request_type", "N/A"),
                "Confidence Score": confidence_score,
                "Customer Name": parsed_output.get("customer_name", "N/A")
            })
    
    # Normalize confidence scores
    normalized_scores = normalize_confidence_scores(confidence_scores)
    for i in range(len(results)):
        results[i]["Confidence Score"] = normalized_scores[i]
    
    return pd.DataFrame(results)

# Main function to classify emails
def main():
    # Folder containing PDF emails
    folder_path = "/workspaces/codespaces-jupyter/data/Input "  # Replace with your folder path for email reading
    
    # Process PDFs and classify emails
    results_df = process_pdf_folder(folder_path)
    
    # Save results to CSV
    results_df.to_csv("/workspaces/codespaces-jupyter/data/Output.csv", index=False) ## Results storing path
    print(results_df)

# Run the program
if __name__ == "__main__":
    main()
