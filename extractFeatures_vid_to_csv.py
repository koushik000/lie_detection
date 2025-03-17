import os
import subprocess
import pandas
import cv2 
import librosa
import numpy

import os
import subprocess
import glob

def extractAU(input_path, output_dir, output_file):
    # Run the FeatureExtraction command
    temp = subprocess.Popen(["echo", "0"], stdout=subprocess.PIPE)
    popen = subprocess.Popen(
        (f"../OpenFace/build/bin/FeatureExtraction -f {input_path} -out_dir {output_dir} -of {output_file}").split(),
        stdin=temp.stdout,
        stdout=subprocess.PIPE
    )
    popen.wait()

    # Adjust file permissions to make sure the user has access
    for file_path in glob.glob(os.path.join(output_dir, '*')):
        os.chmod(file_path, 0o644)  # Set permissions to read/write for the owner and read for others

    # Clean up unwanted output files (e.g., .hog and .avi)
    for file_path in glob.glob(os.path.join(output_dir, '*')):
        if os.path.isfile(file_path) and not file_path.endswith('.csv'):
            print(f"Removing file: {file_path}")
            os.remove(file_path)


def extractMFCC(input_path, output_dir, output_file):
	cam = cv2.VideoCapture(f"{input_path}")
	fps = cam.get(cv2.CAP_PROP_FPS)
	audio, sr = librosa.load(f"{input_path}")
	mfcc = librosa.feature.mfcc(y = audio, sr = sr, hop_length = int(sr // fps))
	mfcc = numpy.transpose(mfcc)
	numpy.savetxt(f"{output_dir}{output_file}", mfcc, delimiter = ",")
	
def main():
	count_truth = 0
	count_lie = 0
	
	print("Extracting Fearutes from Bag-Of-Lies...")
	df = pandas.read_csv("./Annotations.csv")
	for index, row in df.iterrows():
		input_path = row["video"]
		if not os.path.exists(input_path):
			continue
		if row["truth"] == 1:
			if not os.path.exists(f"./Processed/AU/Truth_{count_truth}.csv"):
				extractAU(input_path, "./Processed/AU/", f"Truth_{count_truth}.csv")
			if not os.path.exists(f"./Processed/MFCC/Truth_{count_truth}.csv"):
				extractMFCC(input_path, "./Processed/MFCC/", f"Truth_{count_truth}.csv")
			count_truth = count_truth + 1 
			print("Truth count:", count_truth)
		else:
			if not os.path.exists(f"./Processed/AU/Lie_{count_lie}.csv"):
				extractAU(input_path, "./Processed/AU/", f"Lie_{count_lie}.csv")
			if not os.path.exists(f"./Processed/MFCC/Lie_{count_lie}.csv"):
				extractMFCC(input_path, "./Processed/MFCC/", f"Lie_{count_lie}.csv")
			count_lie = count_lie + 1 
			print("Lie count:", count_lie)		
	print("Successfully extracted AU and MFCC from Bag-Of-Lies!")		
	print(f"Processed {count_truth} videos of truth and {count_lie} videos of lie!")
if __name__ == "__main__":
	main()