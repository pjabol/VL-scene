import os
import argparse
import csv
from openai import OpenAI
import base64
import time

# API configuration
OPENAI_BASE_URL = "" # OpenAI Compatible endpoint, eg. vLLM or API endpoint, https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html or https://platform.openai.com/docs/quickstart
OPENAI_API_KEY = "" # Optional Apikey 
MODEL_NAME = "" # Model name, eg. Qwen/Qwen2.5-VL-7B-Instruct or gpt-4o

# Create OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY),
    base_url=os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL)
)

def analyze_image(image_path, prompt, binary):
    """
    Reads the image, encodes it in base64 and sends it to the model.
    """
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        # Base64 image encoding
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        print(f"Error opening image  {image_path}: {e}")
        return None

    try:
        # Send base64 encoded image as part of file
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                        {"type": "text", "text": f"{prompt}"},
                    ],
                }
            ],
            max_tokens=250 # fit to your needs
        )
        result_text = response.choices[0].message.content.strip()
        if binary:
            try:
                result_val = int(result_text)
                return result_val
            except ValueError:
                print(f"Response processing error {image_path} | Result: {result_text}")
                return f"NaN ({result_text[:1]})"
        else:
            return result_text
    except Exception as e:
        print(f"Image processing error {image_path}: {e}")
        return None

def process_single_file(file_path, csv_writer, start_time, prompt, binary, files=0, ):
    print(f"Image processing {files}: {file_path}", end=" ")
    result = analyze_image(file_path, prompt, binary)
    csv_writer.writerow([os.path.basename(file_path), result, time.time() - start_time])
    print(f"Result: {result}, time spent: {time.time() - start_time}")

def process_folder(folder_path, csv_writer, prompt, binary):
    print(f"Folder processing: {folder_path}")
    image_extensions = ('.jpg', '.jpeg', '.png')
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.lower().endswith(image_extensions)]
    files.sort()
    FILES=0
    if not files:
        print("No JPG/PNG files found in the folder.")
        return
    
    for file_path in files:
        start_time = time.time()
        FILES = FILES + 1
        process_single_file(file_path, csv_writer, start_time, prompt, binary ,FILES)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analysis of images using the LLM model and saving the results to CSV."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single JPG file")
    group.add_argument("--folder", type=str, help="Path to the folder with JPG files")
    parser.add_argument("--output", type=str, default="results.csv", help="Path to the resulting CSV file (default: results.csv)")
    parser.add_argument("--binary", type=str, help="Output binary results to file", default="TRUE")
    # Uncomment the prompt you want to use or pass new prompt as argument
    parser.add_argument("--prompt", type=str, default="""Does the submitted image show a reporter in a television studio? If yes, respond with '1'. If not, respond with '0'. Return only '1' or '0' without providing any additional information. If you are unsure, respond with '0'""",help="Prompt. Czy przesłany obraz przedstawia kadr reportera w studio telewizyjnym?. Jeśli tak, odpowiedz 1, w przeciwnym razie 0. Zwróć wyłącznie wartości 0 lub 1")
    # parser.add_argument("--prompt", type=str, default="""Definition: A reporter (journalist) in a television studio is a person who, in a professional TV environment presents or announces information. They typically address the camera directly, and their role involves hosting the show, introducing upcoming segments, and/or commenting on current events as part of a news broadcast. \nTask: Analyze the submitted image and determine whether it shows a reporter in a television studio announcing a news segment - reply with '1', or not - reply with '0'. Answer with a single number: '1' or '0', without any additional explanation. If you are unsure, answer '0'.""",help="Prompt. Czy przesłany obraz przedstawia kadr reportera w studio telewizyjnym?. Jeśli tak, odpowiedz 1, w przeciwnym razie 0. Zwróć wyłącznie wartości 0 lub 1")
    
    args = parser.parse_args()
    start_time = time.time()

    with open(args.output, mode="a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["filename", "modelEval", "time"])
        if args.file:
            process_single_file(args.file, csv_writer, start_time, args.prompt, args.binary)
        elif args.folder:
            process_folder(args.folder, csv_writer, args.prompt, args.binary)
        csv_writer.writerow([MODEL_NAME,time.time() - start_time])
    
    print(f"Results saved to file: {args.output}")
    print(f"--- All files processed in {time.time() - start_time} seconds ---" )