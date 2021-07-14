import matplotlib.pyplot as plt
import numpy as np
import matplotlib
mtl_regressor_colors = {
	'baseline': 'k',
	'model A': 'g',
	'model B': 'b',
	'model C': 'pink',
	'model D': 'm',
	'model E': 'y',
	'model F': 'c',
	'model G': 'r',
}
sequential_classifiers_colors = {
	'RandomForest': 'k',
	'NaiveBayes': 'g',
	'XGB': 'b',
	'MLP-2': 'm',
	'KNN': 'y',
	'SVM-C=10': 'r',
}
mtl_regressor_values ={
	'baseline':[0.5066858,0.51068616,0.50959753,0.50589209,0.50936303,0.50937063,
			0.51050955,0.50932641,0.5105711,0.50859678],
	'model A':[0.49251452,0.49406225,0.49263322,0.49495272,0.4949043,0.49308661,
			0.4978268,0.50040705,0.50265574,0.4975737],
	'model B':[0.50879399,0.51503246,0.53378283,0.52647857,0.53193693,0.52124907,
			0.54266369,0.53678439,0.52413103,0.54844398],
	'model C':[0.55950767,0.58590568,0.56706693,0.5212562,0.56595202,0.55458257,
			0.52014122,0.5739739,0.55285639,0.572278],
	'model D':[0.5000456,0.5026503,0.46554377,0.51175128,0.47472212,0.51595082,
			0.50867557,0.47023051,0.49021847,0.50687116],
	'model E': [0.49865818, 0.48860633, 0.5054452,  0.50562163, 0.51431797, 0.51155237,
             0.50409944, 0.48178311, 0.49975587, 0.49961937],
	'model F':[0.60579786,0.60613677,0.60921017,0.61570767,0.60598339,0.61656607,
			0.60892292,0.61905755,0.62270765,0.59719039],
	'model G':[0.6331153,0.62842234,0.62754079,0.63807126,0.63278273,0.62619656,
			0.61335589,0.63252092,0.63603595,0.64323212]
}
sequential_classifiers = {
	'RandomForest':[0.75528169,0.74119718,0.76408451,0.74647887,0.73591549,0.75880282,
	0.72711268,0.76760563,0.74250441,0.7372134],
	'NaiveBayes':[0.75,0.73415493,0.73591549,0.72887324,0.70774648,0.7165493,
0.68661972,0.72711268,0.71957672,0.70017637],
	'XGB':[0.75704225,0.74471831,0.76408451,0.75704225,0.74119718,0.77288732,
0.72535211,0.76056338,0.74426808,0.73897707],
	'MLP-2':[0.75,0.74295775,0.75,0.75176056,0.73415493,0.77288732,
0.72887324,0.75,0.7372134,0.7319224],
	'KNN':[0.72007042,0.6971831,0.69542254,0.73591549,0.71830986,0.75176056,
0.70950704,0.71830986,0.71604938,0.71604938],
	'SVM-C=10':[0.7693662,0.74471831,0.76408451,0.75,0.73591549,0.77112676,
0.74295775,0.76056338,0.74779541,0.72839506]
}

def plot_sequential_class_results():
	plt.figure(figsize=(15,8))
	for name, results in sequential_classifiers.items():
		runs = range(10)
		plt.plot(runs, results, label=name, color=sequential_classifiers_colors[name])
		plt.grid(True)
		plt.ylabel('Avg Accuracy')
		plt.xlabel('Run')

	plt.legend()
	plt.savefig(f"/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/scripts/plotting/results/sequential_classifiers")


def plot_regressor_test_values():
	plt.figure(figsize=(15,8))
	for name, results in mtl_regressor_values.items():
		runs = range(10)
		plt.plot(runs, results, label=name, color=mtl_regressor_colors[name])
		plt.grid(True)
		plt.ylabel('Avg CCC')
		plt.xlabel('Run')

	plt.legend()
	plt.savefig(f"/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/scripts/plotting/results/regressor_all_models")

if __name__ == "__main__":
	# plot_regressor_test_values()
	plot_sequential_class_results()
