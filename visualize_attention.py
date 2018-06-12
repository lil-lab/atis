"""
Used for creating a graph of attention over a fixed number of logits over a
sequence. E.g., attention over an input sequence while generating an output
sequence.
"""
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams


class AttentionGraph():
    """Creates a graph showing attention distributions for inputs and outputs.

    Attributes:
      keys (list of str): keys over which attention is done during generation.
      generated_values (list of str): keeps track of the generated values.
      attentions (list of list of float): keeps track of the probability
          distributions.
    """

    def __init__(self, keys):
        """
        Initializes the attention graph.

        Args:
          keys (list of string): a list of keys over which attention is done
            during generation.
        """
        if not keys:
            raise ValueError("Expected nonempty keys for attention graph.")

        self.keys = keys
        self.generated_values = []
        self.attentions = []

    def add_attention(self, gen_value, probabilities):
        """
        Adds attention scores for all item in `self.keys`.

        Args:
          gen_value (string): a generated value for this timestep.
          probabilities (np.array): probability distribution over the keys. Assumes
            the order of probabilities corresponds to the order of the keys.

        Raises:
          ValueError if `len(probabilities)` is not the same as `len(self.keys)`
          ValueError if `sum(probabilities)` is not 1
        """
        if len(probabilities) != len(self.keys):
            raise ValueError("Length of attention keys is " +
                             str(len(self.keys)) +
                             " but got probabilities of length " +
                             str(len(probabilities)))
#        if sum(probabilities) != 1.0:
#            raise ValueError("Probabilities sum to " +
#                             str(sum(probabilities)) + "; not 1.0")

        self.generated_values.append(gen_value)
        self.attentions.append(probabilities)

    def render(self, filename):
        """
        Renders the attention graph over timesteps.

        Args:
          filename (string): filename to save the figure to.
        """
        figure, axes = plt.subplots()
        graph = np.stack(self.attentions)

        axes.imshow(graph, cmap=plt.cm.Blues, interpolation="nearest")
        axes.xaxis.tick_top()
        axes.set_xticks(range(len(self.keys)))
        axes.set_xticklabels(self.keys)
        plt.setp(axes.get_xticklabels(), rotation=90)
        axes.set_yticks(range(len(self.generated_values)))
        axes.set_yticklabels(self.generated_values)
        axes.set_aspect(1, adjustable='box')
#        axes.grid(b=True, color='w', linestyle='-',linewidth=2,which='minor')
#        plt.minorticks_on()
        plt.tick_params(axis='x',which='both',bottom='off',top='off')
        plt.tick_params(axis='y',which='both',left='off',right='off')

        figure.savefig(filename)

    def render_as_latex(self, filename):
        ofile = open(filename, "w")

        ofile.write("\\documentclass{article}\\usepackage[margin=0.5in]{geometry}\\usepackage{tikz}\\begin{document}\\begin{tikzpicture}[scale=0.25]\\begin{tiny}\\begin{scope}<+->;\n")
        xstart = 0
        ystart = 0
        xend = len(self.keys)
        yend = len(self.generated_values)

        ofile.write("\\draw[step=1cm,gray,very thin] (" + str(xstart) + "," + str(ystart) +") grid (" + str(xend) + ", " + str(yend) + ");\n")

        for i, tok in enumerate(self.keys):
            tok = tok.replace("_", "\_")
            tok = tok.replace("#", "\#")
            ofile.write("\\draw[gray, xshift=" + str(i) + ".5 cm] (0,0.3) -- (0,0) node[below,rotate=90,anchor=east] {" + tok + "};\n")

        for i, tok in enumerate(self.generated_values[::-1]):
            tok = tok.replace("_", "\_")
            tok = tok.replace("#", "\#")
            ofile.write("\\draw[gray, yshift=" + str(i) + ".5 cm] (0.3,0) -- (0,0) node[left] {" + tok + "};\n")

        for i, gentok_atts in enumerate(self.attentions[::-1]):
            for j, val in enumerate(gentok_atts):
                if val < 0.001:
                    val = 0
                ofile.write("\\filldraw[thin,red,opacity=" + "%.2f" % val + "] (" + str(j) + ", " + str(i) + ") rectangle (" + str(j+1)+ "," + str(i+1) + ");\n")

        ofile.write("\\end{scope}\\end{tiny}\\end{tikzpicture}{\end{document}")
