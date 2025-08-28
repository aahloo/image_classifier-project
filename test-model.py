import subprocess
import sys
import os

def run_command(command):

    print(f"Running: {' '.join(command)}")

    result = subprocess.run(command, capture_output = True, text = True)

    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode == 0

def main():

    # test training
    print("*** Testing Training ***")

    train_cmd = [ sys.executable, "train.py", "flowers/", "--epochs", "1", "--hidden_units", "256" ]

    if run_command(train_cmd):
        print("✅ Training completed successfully!")

        # then test prediction
        print("*** Testing Prediction ***")

        predict_cmd = [ sys.executable, "predict.py", "flowers/test/10/image_07104.jpg", "checkpoint.pth", "--top_k", "3" ]

        if run_command(predict_cmd):
            print("✅ Prediction completed successfully!")
        else:
            print("❌ Prediction failed!")
    
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()
