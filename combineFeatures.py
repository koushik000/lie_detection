import pandas as pd
import numpy as np

def main():
	count_truth = 139
	count_lie = 123
	maxlines = 0
	for x in range(0, count_truth, 1):
		fname = "Truth_" + str(x)
		df = pd.read_csv(f"./clips/Processed/AU/{fname}.csv")
		col_list = list(filter(lambda x: x[0: 2] == "AU" and x[-1] == "r", list(df)))
		array_AU = df[col_list].to_numpy()
		array_MFCC = np.genfromtxt(f"./clips/Processed/MFCC/{fname}.csv", delimiter = ",")
		if array_AU.shape[0] > array_MFCC.shape[0]:
			dl = array_AU.shape[0] - array_MFCC.shape[0]
			array_AU = array_AU[ : -dl]
		if array_AU.shape[0] < array_MFCC.shape[0]:
			dl = array_MFCC.shape[0] - array_AU.shape[0]
			array_MFCC = array_MFCC[ : -dl]
		# print(array_AU.shape, " ", array_MFCC.shape)
		maxlines = max(maxlines, array_AU.shape[0])	
		array = np.concatenate((array_AU, array_MFCC), axis = 1)
		np.savetxt(f"./Processed/{fname}.csv", array, delimiter = ",")
		print(f"Truth_{x}...")		

	for x in range(0, count_lie, 1):
		fname = "Lie_" + str(x)
		df = pd.read_csv(f"./clips/Processed/AU/{fname}.csv")
		col_list = list(filter(lambda x: x[0: 2] == "AU" and x[-1] == "r", list(df)))
		array_AU = df[col_list].to_numpy()
		array_MFCC = np.genfromtxt(f"./clips/Processed/MFCC/{fname}.csv", delimiter = ",")
		if array_AU.shape[0] > array_MFCC.shape[0]:
			dl = array_AU.shape[0] - array_MFCC.shape[0]
			array_AU = array_AU[ : -dl]
		if array_AU.shape[0] < array_MFCC.shape[0]:
			dl = array_MFCC.shape[0] - array_AU.shape[0]
			array_MFCC = array_MFCC[ : -dl]
		# print(array_AU.shape, " ", array_MFCC.shape)
		maxlines = max(maxlines, array_AU.shape[0])	
		array = np.concatenate((array_AU, array_MFCC), axis = 1)
		np.savetxt(f"./Processed/{fname}.csv", array, delimiter = ",")
		print(f"Lie_{x}...")
	print("maxlines:", maxlines)
if __name__ == "__main__":
	main()