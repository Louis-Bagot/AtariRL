import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def graph(l, xlab, ylab, title, game_name):
    f = plt.figure()
    p = f.add_subplot(111)
    p.plot(l)
    p.set_xlabel(xlab)
    p.set_ylabel(ylab)
    p.set_title(title)
    plt.savefig('../graphic/perf_' + game_name + '.png')
    plt.close(f)
