import logging
from flask import Flask, abort, request
from tempfile import NamedTemporaryFile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from werkzeug.utils import secure_filename
import time, os, json
import shutil
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a formatter with date-time format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create a file handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


def load_model(model_id, torch_dtype,use_auth_token=None, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=False):
    # Define the model directory and model path
    model_dir = './model'
    model_path = os.path.join(model_dir, model_id.replace("/", "_"))
    
    # Step 1: Ensure the local directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Step 2: Check if the model has already been downloaded
    if not os.path.exists(model_path):
        # Download the model to the local directory
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            cache_dir=model_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_safetensors=use_safetensors,
            use_flash_attention_2=use_flash_attention_2,
            use_auth_token=use_auth_token
        )
        # Save the model to the local path
        model.save_pretrained(model_path)
    else:
        print(f"Model '{model_id}' already downloaded.")
    
    # Step 3: Load the model from the local directory
    local_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        use_safetensors=use_safetensors,
        use_flash_attention_2=use_flash_attention_2,
        use_auth_token=use_auth_token
    )
    return local_model

# Download and Load the Whisper model:

# Check if NVIDIA GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# Select your HF model
model_id = "distil-whisper/distil-large-v2"
use_auth_token = "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"
model = load_model(model_id, torch_dtype, use_auth_token=use_auth_token)
# model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=False, use_auth_token="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")
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=200,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"language": "en", "task": "transcribe"},
)

asr_app = Flask(__name__)
asr_app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 Megabytes

UPLOAD_FOLDER = './uploads'
metadata_dir = 'uploads/metadata'
CHUNK_SIZE = 1024 * 1024 * 2  # 2 Megabyte

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'wav', 'mp3'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@asr_app.route("/")
def hello():
    logger.info("Received request at /")
    return "Whisper Hello World!"

@asr_app.route('/transcribe', methods=['POST'])
def transcribe_30sec():
    logger.info("Received POST request at /transcribe")
    if 'file' not in request.files:
        # If the user didn't submit any files, return a 400 (Bad Request) error.
        logger.warning("No files submitted in the request")
        abort(400, 'No file part in the request.')
    
    # file = request.files['file']

    files = request.files.getlist('file')
    results = []
    for file in files:
        if file.filename == '':
            abort(400, 'No selected file.')

        if file and allowed_file(file.filename):
            timestamp = time.time()  # Get the current timestamp
            secure_name = secure_filename(file.filename)
            secure_name_with_timestamp = f"{secure_name}_{timestamp}"

            # Save the file to the 'data/' directory with the secure filename
            new_filename = os.path.join(UPLOAD_FOLDER, secure_name_with_timestamp)

            # If the file already exists, append to it
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the directory exists
            if os.path.exists(new_filename):
                with open(new_filename, 'ab') as f:
                    f.write(file.stream.read(CHUNK_SIZE))
            else:
                file.save(new_filename)

            # Transcribe the file
            logger.info(f"Transcribing file: {new_filename}")
            start_time = time.time()  # Start the timer
            result = pipe(new_filename)  # Use the model to make predictions
            end_time = time.time()  # Stop the timer
            transcribe_time = end_time - start_time  # Calculate the time taken for transcription

            results.append({
                'filename':  file.filename,
                'transcript': result['text'],
                'transcribing_time': f'{transcribe_time:.2f} seconds',
            })
            # Create the metadata
            metadata = {
                'original_name': file.filename,
                'secure_name': secure_name_with_timestamp,
                'timestamp': timestamp,
                'new_location': new_filename,
                'transcribe_time': transcribe_time,  # Add the transcription time to the metadata
                'transcript': result['text'],  # Add the generated transcript to the metadata
            }

            # Save the metadata to a file
            # metadata_dir = 'data/metadata'
            os.makedirs(metadata_dir, exist_ok=True)  # Ensure the directory exists
            metadata_filename = os.path.join(metadata_dir, secure_name_with_timestamp + '.metadata')
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f)

    if not results:
        abort(400, 'Invalid files.')

    logger.info("Transcription completed")
    return {'results': results}

    # for filename, handle in request.files.items():
    #     temp = NamedTemporaryFile(delete=False)  # Keep the file after it's closed
    #     handle.save(temp.name)

    #     # Transcribe the file
    #     logger.info(f"Transcribing file: {filename}")
    #     start_time = time.time()  # Start the timer
    #     result = pipe(temp.name)  # Use the model to make predictions
    #     end_time = time.time()  # Stop the timer
    #     transcribe_time = end_time - start_time  # Calculate the time taken for transcription

    #     results.append({
    #         'filename': filename,
    #         'transcript': result['text'],
    #         'transcribing_time': f'{transcribe_time:.2f} seconds',
    #     })

    #     # Generate a secure version of the original filename
    #     secure_name = secure_filename(filename)
    #     timestamp = time.time()  # Get the current timestamp
    #     secure_name_with_timestamp = f"{secure_name}_{timestamp}"

    #     # Save the file to the 'data/' directory with the secure filename
    #     new_filename = os.path.join('data', secure_name_with_timestamp)
    #     shutil.move(temp.name, new_filename)
    #     # Save the metadata
    #     metadata = {
    #         'original_name': filename,
    #         'secure_name': secure_name_with_timestamp,
    #         'timestamp': timestamp,
    #         'new_location': new_filename,
    #         'transcribe_time': transcribe_time,  # Add the transcription time to the metadata
    #         'transcript': result['text'],  # Add the generated transcript to the metadata
    #     }

    #     # Save the metadata to a file
    #     metadata_dir = 'data/metadata'
    #     os.makedirs(metadata_dir, exist_ok=True)  # Ensure the directory exists
    #     metadata_filename = os.path.join(metadata_dir, secure_name_with_timestamp + '.metadata')
    #     with open(metadata_filename, 'w') as f:
    #         json.dump(metadata, f)

    # logger.info("Transcription completed")
    # return {'results': results}

@asr_app.route('/transcribe-long', methods=['POST'])
def transcribe_long():
    logger.info("Received POST request at /transcribe-long")
    if 'file' not in request.files:
        # If the user didn't submit any files, return a 400 (Bad Request) error.
        logger.warning("No files submitted in the request")
        abort(400, 'No file part in the request.')
    
    # file = request.files['file']

    files = request.files.getlist('file')
    results = []
    for file in files:
        if file.filename == '':
            abort(400, 'No selected file.')

        if file and allowed_file(file.filename):
            timestamp = time.time()  # Get the current timestamp
            secure_name = secure_filename(file.filename)
            secure_name_with_timestamp = f"{secure_name}_{timestamp}"

            # Save the file to the 'data/' directory with the secure filename
            new_filename = os.path.join(UPLOAD_FOLDER, secure_name_with_timestamp)

            # If the file already exists, append to it
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the directory exists
            if os.path.exists(new_filename):
                with open(new_filename, 'ab') as f:
                    f.write(file.stream.read(CHUNK_SIZE))
            else:
                file.save(new_filename)
            logger.info("File Saved")

            # Transcribe the file
            logger.info("Transcribing file: %s", new_filename)
            start_time = time.time()  # Start the timer
            result = pipe(new_filename)  # Use the model to make predictions
            end_time = time.time()  # Stop the timer
            transcribe_time = end_time - start_time  # Calculate the time taken for transcription

            results.append({
                'filename':  file.filename,
                'transcript': result['text'],
                'transcribing_time': f'{transcribe_time:.2f} seconds',
            })
            # Create the metadata
            metadata = {
                'original_name': file.filename,
                'secure_name': secure_name_with_timestamp,
                'timestamp': timestamp,
                'new_location': new_filename,
                'transcribe_time': transcribe_time,  # Add the transcription time to the metadata
                'transcript': result['text'],  # Add the generated transcript to the metadata
            }

            # Save the metadata to a file
            # metadata_dir = 'data/metadata'
            os.makedirs(metadata_dir, exist_ok=True)  # Ensure the directory exists
            metadata_filename = os.path.join(metadata_dir, secure_name_with_timestamp + '.metadata')
            with open(metadata_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f)

    if not results:
        abort(400, 'Invalid files.')

    logger.info("Transcription completed")
    return {'results': results}

if __name__ == "__main__":
    logger.info("Starting the server")
    asr_app.run(host='0.0.0.0', port=5001)  # Replace with your actual host and port