"""
Example of traces extraction for DeepMoji Classifier
"""
import matplotlib.pyplot as plt
import numpy as np
import json

MAPPING = {
"SAD" : [35, 5, 27, 43, 45, 52, 2, 29, 3, 34, 46],
"MAD" : [37, 32, 55, 22, 25, 1, 19],
"FEAR" : [51, 62, 12, 20], # 0
"MISC" : [14, 39, 42],
"FORCE" : [57, 58, 30, 13, 38, 56],
"HAPPY" : [4, 36, 10, 7, 53],
"DEAL"  : [6, 33, 17, 40],
"WINK"  : [50, 9, 54, 31, 44, 15, 26],
"MUSIC" : [11, 48],
"EYES"  : [41, 28, 49],
"LOVE"  : [21, 47, 8, 16, 23, 59, 61, 18, 60], # 24, # 63
"IGNORE" : [63, 24, 0]
}

BACK_MAPPING = dict()

AGGVECTOR = ['LOVE','HAPPY','WINK','DEAL','FORCE','EYES','FEAR','MAD','SAD','MUSIC','MISC']
SMOOTH = 1e-3
ALPHA = 0.99
ARC_TH = 0.1 
ARC_PEAK_TH = 0.3
INTERP = 20

for c, ids in MAPPING.items():
	if c in AGGVECTOR:
		for i in ids:
			BACK_MAPPING[i] = AGGVECTOR.index(c)


banned = open('banned.txt', encoding='utf-8').read().split('\n')

def update_beliefs(probs, universes = None):
    #probs - probabitity distribution for current string
    #univerces - aggregated probabilities for previous text
    #(u+smooth) corresponds to conditional probability to observe u with previous data
    # other part of numerator - probability of u with current data
	if universes is None:
		return probs
	denominator = sum([(u*(1.-ALPHA)+p*ALPHA)*(u+SMOOTH) for p,u in zip(probs, universes)])
	return [(u*(1.-ALPHA)+p*ALPHA)*(u+SMOOTH)/denominator for p,u in zip(probs, universes)]


    
def clean_trace(tale_idx, traces):
	best_arcs = {}
	traces = list(zip(*traces))
	full_weight = sum([sum(row) for row in traces])
	lines = dict()
	for name, row in zip(AGGVECTOR,traces):
		if sum(row)/full_weight>ARC_TH or max(row)>ARC_PEAK_TH:
			best_arcs[name] = row
	return best_arcs

def dump_trace(tale_idx, traces, path=None):
	if path is None:
		path = '{}.png'.format(tale_idx)
	traces = list(zip(*traces))
	full_weight = sum([sum(row) for row in traces])

	lines = dict()
	for name, row in zip(AGGVECTOR,traces):
		if sum(row)/full_weight>ARC_TH or max(row)>ARC_PEAK_TH:
			if INTERP>0:
				row = np.interp(np.linspace(0, 1., INTERP), list(map(lambda x:x/len(row),range(len(row)))), row)
			plt.plot(np.linspace(0, 1., INTERP), row, label=name)
			lines[name] = row.tolist()
	plt.suptitle(f"{tale_idx} {tale_idx}")
	plt.legend()
	if len(lines)>1:
		plt.savefig(path)
	plt.clf()
    
    
universes = None
prev_tale_idx = None
traces = []
with open('data/aib100_translated.emojis.tsv', encoding='utf-8') as ifh:
	for line in ifh:
		chunks = line.strip().split('\t')
		idx = chunks[:3]
		text = chunks[3]
		scores = chunks[4:]
		agg_scores = [0.,]*11
		for i in range(64):
			if i in BACK_MAPPING:
				agg_scores[BACK_MAPPING[i]] += float(scores[i])

		if prev_tale_idx != idx[0]:
			if prev_tale_idx is not None:
				dump_trace(prev_tale_idx, traces)
				pass
			prev_tale_idx = idx[0]
			traces = []
			universes = None
		universes = update_beliefs(agg_scores, universes)
		traces.append( universes )