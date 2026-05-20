import os
import sys
import subprocess
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize your app with your bot token
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# Global dictionary to keep track of running training processes
training_processes = {}

@app.command("/train-start")
def handle_train_start(ack, respond, command):
    ack()
    user_id = command["user_id"]
    
    if "fsd_training" in training_processes and training_processes["fsd_training"].poll() is None:
        respond("A training process is already running! Please stop it first using `/train-stop`.")
        return

    respond(f"<@{user_id}> Starting the E2E FSD training process on the MacBook... 🚗💨")
    
    try:
        # Launch the training script as a background subprocess
        process = subprocess.Popen(
            [sys.executable, "train_fsd.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        training_processes["fsd_training"] = process
        respond(f"Training started successfully! Process ID: {process.pid}")
    except Exception as e:
        respond(f"Failed to start training: {str(e)}")

@app.command("/train-stop")
def handle_train_stop(ack, respond, command):
    ack()
    
    if "fsd_training" not in training_processes or training_processes["fsd_training"].poll() is not None:
        respond("There is no training process currently running.")
        return
        
    process = training_processes["fsd_training"]
    respond("Stopping the E2E FSD training process... 🛑")
    
    try:
        process.terminate()
        process.wait(timeout=5)
        respond("Training process has been successfully stopped and resources cleaned up.")
    except Exception as e:
        respond(f"Error stopping the process: {str(e)}")

@app.event("app_mention")
def handle_app_mention_events(body, say):
    say("Hello! I am your FSD Training Bot. Use `/train-start` and `/train-stop` to control training on this MacBook.")

if __name__ == "__main__":
    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    app_token = os.environ.get("SLACK_APP_TOKEN")
    
    if not bot_token or not app_token:
        print("ERROR: Please set SLACK_BOT_TOKEN and SLACK_APP_TOKEN in your .env file or environment.")
        exit(1)
        
    print("⚡️ FSD Training Slack Bot is running!")
    handler = SocketModeHandler(app, app_token)
    handler.start()
