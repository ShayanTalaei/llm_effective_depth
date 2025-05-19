import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True #Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = '\\usepackage{amsmath}\n\\usepackage{sansmath}\n\\usepackage{bm}' #Force sans-serif math mode (for axes labels)
plt.rcParams['font.family'] = 'sans-serif' # ... for regular text
plt.rcParams['font.sans-serif'] = ['Times New Roman']
#'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here

def sort_zorder(bars):
    for group in zip(*[container for container in bars]):
        # 'group' is a tuple of bar objects at one x position from different groups.
        z = len(group)
        # Sort the bars by height (lowest first).
        for bar in sorted(group, key=lambda b: abs(b.get_height())):
            bar.set_zorder(z)
            z -= 1

def replace_bos(tokens):
    return [t if t != "<|begin_of_text|>" else "BOS" for t in tokens]